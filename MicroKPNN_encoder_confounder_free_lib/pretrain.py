# Modified train.py
import torch
import numpy as np
import dcor
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, auc,
    precision_score, recall_score, confusion_matrix
)

def freeze_batchnorm(module):
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        module.eval()
    for child in module.children():
        freeze_batchnorm(child)

def unfreeze_batchnorm(module):
    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        module.train()
    for child in module.children():
        unfreeze_batchnorm(child)

def train_model(
    model, criterion, optimizer, data_loader, data_all_loader, data_val_loader, data_all_val_loader,
    data_test_loader, data_all_test_loader, num_epochs, 
    criterion_classifier, optimizer_classifier, 
    criterion_reconstructor, optimizer_reconstructor, device
):
    """
    Train the model using a three-phase procedure:
      - Phase 1: Drug/Confounder classification training (r_loss) using data_loader.
      - Phase 2: Distillation training (g_loss) with PearsonCorrelationLoss.
      - Phase 3: Reconstruction training (rec_loss) using data_all_loader for pre-training.
      
    Metrics for training, validation, and test phases are stored.
    """
    # Initialize results dictionary to store metric histories.
    results = {
        "train": {
            "gloss_history": [],      # g_loss: distillation phase loss
            "rloss_history": [],       # rec_loss: reconstruction loss
            "dcor_history": [],       # Distance correlation measure
            "rec_loss_history": [],                # Mean squared error for reconstruction
        },
        "val": {
            "rec_loss_history": [],
            "dcor_history": [],
        },
        "test": {
            "rec_loss_history": [],
            "dcor_history": [],
        }
    }

    # Move model and loss functions to device.
    model = model.to(device)
    criterion = criterion.to(device)
    criterion_classifier = criterion_classifier.to(device)
    criterion_reconstructor = criterion_reconstructor.to(device)

    # Begin epoch loop.
    for epoch in range(num_epochs):
        model.train()  # Enable training mode

        # Initialize accumulators for training phase metrics.
        epoch_gloss = 0
        epoch_rloss = 0
        epoch_rec_loss = 0
        hidden_activations_list = []
        targets_list = []

        # Create iterators for the two DataLoaders used in Phase 1 and Phase 3.
        data_iter = iter(data_loader)
        data_all_iter = iter(data_all_loader)

        # Loop over batches until data_all_iter is exhausted.
        while True:
            try:
                # -------- Phase 3: Reconstruction batch --------
                x_all_batch = next(data_all_iter)  # Ignore labels for reconstruction
                x_all_batch = x_all_batch.to(device)

                # -------- Phase 1: Train confounder classifier --------
                try:
                    x_batch, y_batch = next(data_iter)
                except StopIteration:
                    # Restart the iterator if exhausted.
                    data_iter = iter(data_loader)
                    x_batch, y_batch = next(data_iter)
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                model.zero_grad()
                # Freeze encoder parameters temporarily.
                for param in model.encoder.parameters():
                    param.requires_grad = False
                model.encoder.eval()
                encoded_features = model.encoder(x_batch)
                predicted_drug = model.classifier(encoded_features)
                r_loss = criterion_classifier(predicted_drug, y_batch)
                optimizer_classifier.zero_grad()
                r_loss.backward()
                optimizer_classifier.step()
                epoch_rloss += r_loss.item()
                # Unfreeze encoder parameters.
                for param in model.encoder.parameters():
                    param.requires_grad = True
                model.encoder.train()

                # -------- Phase 2: Train distillation (alignment) --------
                model.zero_grad()
                # Freeze classifier parameters.
                for param in model.classifier.parameters():
                    param.requires_grad = False
                model.classifier.eval()
                encoded_features = model.encoder(x_batch)
                # Use sigmoid on the classifier output.
                predicted_drug = torch.sigmoid(model.classifier(encoded_features))
                g_loss = criterion(predicted_drug, y_batch)
                # Save hidden activations and targets for computing distance correlation.
                hidden_activations_list.append(encoded_features.detach().cpu())
                targets_list.append(y_batch.detach().cpu())
                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()
                epoch_gloss += g_loss.item()
                # Unfreeze classifier parameters.
                for param in model.classifier.parameters():
                    param.requires_grad = True
                model.classifier.train()

                # -------- Phase 3: Train encoder & reconstructor --------
                model.zero_grad()
                encoded_features_all = model.encoder(x_all_batch)
                # Changed from disease_classifier to reconstructor
                reconstructed_x = model.reconstructor(encoded_features_all)
                rec_loss = criterion_reconstructor(reconstructed_x, x_all_batch)
                optimizer_reconstructor.zero_grad()
                rec_loss.backward()
                optimizer_reconstructor.step()
                epoch_rec_loss += rec_loss.item()

                # DEBUG
                has_nan = torch.isnan(x_all_batch).any().item()
                if has_nan: 
                    print(f"Batch {epoch} - x_all_batch: {x_all_batch}")
                    exit()
                has_nan = torch.isnan(encoded_features_all).any().item()
                if has_nan: 
                    print(f"Batch {epoch} - encoded_features_all has NaN: {has_nan}")
                    exit()
                has_nan = torch.isnan(reconstructed_x).any().item()
                if has_nan: 
                    print(f"Batch {epoch} - reconstructed_x has NaN: {has_nan}")
                    exit()

            except StopIteration:
                break  # End of epoch

        # Compute training metrics over the epoch.
        avg_gloss = epoch_gloss / len(data_all_loader)
        avg_rloss = epoch_rloss / len(data_all_loader)
        avg_rec_loss = epoch_rec_loss / len(data_all_loader)
        results["train"]["gloss_history"].append(avg_gloss)
        results["train"]["rloss_history"].append(avg_rloss)
        results["train"]["rec_loss_history"].append(avg_rec_loss)

        # Compute distance correlation for training phase.
        hidden_activations_all = torch.cat(hidden_activations_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
        dcor_value = dcor.distance_correlation_sqr(hidden_activations_all.numpy(), targets_all.numpy())
        results["train"]["dcor_history"].append(dcor_value)

        # DEBUG: Print the average loss.
        print(f"Epoch {epoch+1}/{num_epochs} [Train] - g_loss: {avg_gloss:.4f}, r_loss: {avg_rloss:.4f}, rec_loss: {avg_rec_loss:.4f}, dcor: {dcor_value:.4f}")

        # -------------- Validation Phase --------------
        model.eval()
        epoch_val_loss = 0
        val_hidden_activations_list = []
        val_targets_list = []

        with torch.no_grad():
            for x_batch in data_all_val_loader:  # Ignore labels for reconstruction
                x_batch = x_batch.to(device)
                encoded_features = model.encoder(x_batch)
                reconstructed_x = model.reconstructor(encoded_features)
                rec_loss = criterion_reconstructor(reconstructed_x, x_batch)
                epoch_val_loss += rec_loss.item()
                
            for x_batch, y_batch in data_val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                val_hidden_activations_list.append(encoded_features.cpu())
                val_targets_list.append(y_batch.cpu())

        avg_val_loss = epoch_val_loss / len(data_all_val_loader)
        results["val"]["rec_loss_history"].append(avg_val_loss)
        
        val_hidden_activations_all = torch.cat(val_hidden_activations_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)
        val_dcor_value = dcor.distance_correlation_sqr(val_hidden_activations_all.numpy(), val_targets_all.numpy())
        results["val"]["dcor_history"].append(val_dcor_value)

        # DEBUG: Print the average loss.
        print(f"Epoch {epoch+1}/{num_epochs} [Valid] - rec_loss: {avg_val_loss:.4f}, dcor: {dcor_value:.4f}")

        # -------------- Test Phase --------------
        epoch_test_loss = 0
        test_hidden_activations_list = []
        test_targets_list = []

        with torch.no_grad():
            for x_batch in data_all_test_loader:  # Ignore labels for reconstruction
                x_batch = x_batch.to(device)
                encoded_features = model.encoder(x_batch)
                reconstructed_x = model.reconstructor(encoded_features)
                rec_loss = criterion_reconstructor(reconstructed_x, x_batch)
                epoch_test_loss += rec_loss.item()
                
            for x_batch, y_batch in data_test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                test_hidden_activations_list.append(encoded_features.cpu())
                test_targets_list.append(y_batch.cpu())

        avg_test_loss = epoch_test_loss / len(data_all_test_loader)
        results["test"]["rec_loss_history"].append(avg_test_loss)
        
        test_hidden_activations_all = torch.cat(test_hidden_activations_list, dim=0)
        test_targets_all = torch.cat(test_targets_list, dim=0)
        test_dcor_value = dcor.distance_correlation_sqr(test_hidden_activations_all.numpy(), test_targets_all.numpy())
        results["test"]["dcor_history"].append(test_dcor_value)

        # if (epoch + 1) % 50 == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs} [Train] - g_loss: {avg_gloss:.4f}, r_loss: {avg_rloss:.4f}, rec_loss: {avg_rec_loss:.4f}, dcor: {dcor_value:.4f}")
        #     print(f"Epoch {epoch+1}/{num_epochs} [Valid] - rec_loss: {avg_val_loss:.4f}, dcor: {val_dcor_value:.4f}")

    return results