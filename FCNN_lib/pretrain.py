import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix

def train_model(model, data_all_loader, data_all_val_loader, data_all_test_loader, num_epochs, criterion_reconstructor, optimizer_reconstructor, device, pretrained_model_path=None ):
    """
    Train the model while recording metrics for training, validation, and testing.
    """
    results = {
        "train": {
            "rec_loss_history": [],                # Mean squared error for reconstruction
        },
        "val": {
            "rec_loss_history": [],
        },
        "test": {
            "rec_loss_history": [],
        }
    }

    model = model.to(device)
    criterion_reconstructor = criterion_reconstructor.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_rec_loss = 0
        
        # Training phase: Using manual iterator and StopIteration handling.
        data_all_iter = iter(data_all_loader)
        while True:
            try:
                x_batch = next(data_all_iter)
                x_batch = x_batch.to(device)
            except StopIteration:
                break
            # -------- Phase 3: Train encoder & reconstructor --------
            model.zero_grad()
            encoded_features_all = model.encoder(x_batch)
            # Changed from disease_classifier to reconstructor
            reconstructed_x = model.reconstructor(encoded_features_all)
            rec_loss = criterion_reconstructor(reconstructed_x, x_batch)
            optimizer_reconstructor.zero_grad()
            rec_loss.backward()
            optimizer_reconstructor.step()
            epoch_rec_loss += rec_loss.item()
            

            # DEBUG
            has_nan = torch.isnan(x_batch).any().item()
            if has_nan: 
                print(f"Batch {epoch} - x_all_batch: {x_batch}")
                exit()
            has_nan = torch.isnan(encoded_features_all).any().item()
            if has_nan: 
                print(f"Batch {epoch} - encoded_features_all has NaN: {has_nan}")
                exit()
            has_nan = torch.isnan(reconstructed_x).any().item()
            if has_nan: 
                print(f"Batch {epoch} - reconstructed_x has NaN: {has_nan}")
                exit()
        
        avg_rec_loss = epoch_rec_loss / len(data_all_loader)
        results["train"]["rec_loss_history"].append(avg_rec_loss)

         # DEBUG: Print the average loss.
        print(f"Epoch {epoch+1}/{num_epochs} [Train] - rec_loss: {avg_rec_loss:.4f}")
        
        
        # Validation phase.
        model.eval()
        epoch_val_loss = 0
 
        with torch.no_grad():
            for x_batch in data_all_val_loader:  # Ignore labels for reconstruction
                x_batch = x_batch.to(device)
                encoded_features = model.encoder(x_batch)
                reconstructed_x = model.reconstructor(encoded_features)
                rec_loss = criterion_reconstructor(reconstructed_x, x_batch)
                epoch_val_loss += rec_loss.item()
        
        avg_val_loss = epoch_val_loss / len(data_all_val_loader)
        results["val"]["rec_loss_history"].append(avg_val_loss)

        # DEBUG: Print the average loss.
        print(f"Epoch {epoch+1}/{num_epochs} [Valid] - rec_loss: {avg_val_loss:.4f}")
        
        # Test phase.
        epoch_test_loss = 0
        with torch.no_grad():
            for x_batch in data_all_test_loader:  # Ignore labels for reconstruction
                x_batch = x_batch.to(device)
                encoded_features = model.encoder(x_batch)
                reconstructed_x = model.reconstructor(encoded_features)
                rec_loss = criterion_reconstructor(reconstructed_x, x_batch)
                epoch_test_loss += rec_loss.item()
                
        avg_test_loss = epoch_test_loss / len(data_all_test_loader)
        results["test"]["rec_loss_history"].append(avg_test_loss)
        
      
    return results
