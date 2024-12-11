import torch
import numpy as np
import dcor
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix

def train_model(model, criterion, optimizer, data_loader, data_all_loader, data_val_loader, data_all_val_loader,
                data_test_loader, data_all_test_loader, num_epochs, criterion_classifier, optimizer_classifier,
                criterion_disease_classifier, optimizer_disease_classifier, device):
    """
    Train the model with the given data loaders and hyperparameters.
    """
    disease_loss_history = []
    loss_history = []
    dcor_history = []
    train_disease_accs = []
    train_disease_f1s = []
    train_auc_pr_history = []
    train_precisions = []
    train_recalls = []
    train_conf_matrices = []
    # Validation metrics
    val_loss_history = []
    val_accs = []
    val_f1s = []
    val_dcor_history = []
    val_auc_pr_history = []
    val_precisions = []
    val_recalls = []
    val_conf_matrices = []
    # Test metrics
    test_loss_history = []
    test_accs = []
    test_f1s = []
    test_dcor_history = []
    test_auc_pr_history = []
    test_precisions = []
    test_recalls = []
    test_conf_matrices = []

    model = model.to(device)
    criterion = criterion.to(device)
    criterion_classifier = criterion_classifier.to(device)
    criterion_disease_classifier = criterion_disease_classifier.to(device)

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0
        epoch_disease_loss = 0
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_probs = []  # For PR curve
        hidden_activations_list = []
        targets_list = []

        # Create iterators from both data_loaders
        data_iter = iter(data_loader)
        data_all_iter = iter(data_all_loader)

        while True:
            try:
                # Get the next batch from data_all_loader
                x_all_batch, y_all_batch = next(data_all_iter)
                x_all_batch, y_all_batch = x_all_batch.to(device), y_all_batch.to(device)

                # Try to get the next batch from data_loader
                try:
                    x_batch, y_batch = next(data_iter)
                except StopIteration:
                    # If data_loader is exhausted, re-initialize it
                    data_iter = iter(data_loader)
                    x_batch, y_batch = next(data_iter)
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Train drug classification (r_loss)
                for param in model.encoder.parameters():
                    param.requires_grad = False

                encoded_features = model.encoder(x_batch)
                predicted_drug = model.classifier(encoded_features)
                r_loss = criterion_classifier(predicted_drug, y_batch)

                optimizer_classifier.zero_grad()
                r_loss.backward()
                optimizer_classifier.step()

                for param in model.encoder.parameters():
                    param.requires_grad = True

                # Train distiller (g_loss)
                for param in model.classifier.parameters():
                    param.requires_grad = False

                encoded_features = model.encoder(x_batch)
                predicted_drug = model.classifier(encoded_features)
                predicted_drug = torch.sigmoid(predicted_drug)
                g_loss = criterion(predicted_drug, y_batch)

                hidden_activations_list.append(encoded_features.detach().cpu())
                targets_list.append(y_batch.detach().cpu())

                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()

                epoch_loss += g_loss.item()

                for param in model.classifier.parameters():
                    param.requires_grad = True

                # Train encoder & disease classifier (c_loss)
                encoded_features_all = model.encoder(x_all_batch)
                predicted_disease_all = model.disease_classifier(encoded_features_all)
                c_loss = criterion_disease_classifier(predicted_disease_all, y_all_batch)

                optimizer_disease_classifier.zero_grad()
                c_loss.backward()
                optimizer_disease_classifier.step()

                epoch_disease_loss += c_loss.item()

                pred_prob = torch.sigmoid(predicted_disease_all).detach().cpu()
                epoch_train_probs.extend(pred_prob)
                pred_tag = (pred_prob > 0.5).float()
                epoch_train_preds.append(pred_tag.cpu())
                epoch_train_labels.append(y_all_batch.cpu())

            except StopIteration:
                break

        # Compute training metrics
        avg_loss = epoch_loss / len(data_all_loader)
        avg_disease_loss = epoch_disease_loss / len(data_all_loader)
        loss_history.append(avg_loss)
        disease_loss_history.append(avg_disease_loss)

        epoch_train_probs = torch.cat(epoch_train_probs)
        epoch_train_preds = torch.cat(epoch_train_preds)
        epoch_train_labels = torch.cat(epoch_train_labels)

        train_disease_acc = balanced_accuracy_score(epoch_train_labels, epoch_train_preds)
        train_disease_accs.append(train_disease_acc)
        train_disease_f1 = f1_score(epoch_train_labels, epoch_train_preds)
        train_disease_f1s.append(train_disease_f1)
        precision, recall, _ = precision_recall_curve(epoch_train_labels, epoch_train_probs)
        train_auc_pr = auc(recall, precision)
        train_auc_pr_history.append(train_auc_pr)
        train_precision = precision_score(epoch_train_labels, epoch_train_preds)
        train_recall = recall_score(epoch_train_labels, epoch_train_preds)
        # Append to lists if you want to track over epochs
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        # Compute confusion matrix for train phase
        train_conf_matrix = confusion_matrix(epoch_train_labels, epoch_train_preds)
        # Append to the list if tracking over epochs
        train_conf_matrices.append(train_conf_matrix)

        # Distance correlation
        hidden_activations_all = torch.cat(hidden_activations_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
        hidden_activations_np = hidden_activations_all.numpy()
        targets_np = targets_all.numpy()
        dcor_value = dcor.distance_correlation_sqr(hidden_activations_np, targets_np)
        dcor_history.append(dcor_value)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        epoch_val_preds = []
        epoch_val_labels = []
        epoch_val_probs = []  # For PR curve
        val_hidden_activations_list = []
        val_targets_list = []

        with torch.no_grad():
            for x_batch, y_batch in data_all_val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                predicted_disease = model.disease_classifier(encoded_features)
                c_loss = criterion_disease_classifier(predicted_disease, y_batch)
                epoch_val_loss += c_loss.item()
                pred_prob = torch.sigmoid(predicted_disease).detach().cpu()
                epoch_val_probs.extend(pred_prob)
                pred_tag = (pred_prob > 0.5).float()
                epoch_val_preds.append(pred_tag.cpu())
                epoch_val_labels.append(y_batch.cpu())

            for x_batch, y_batch in data_val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                val_hidden_activations_list.append(encoded_features.cpu())
                val_targets_list.append(y_batch.cpu())

        avg_val_loss = epoch_val_loss / len(data_all_val_loader)
        val_loss_history.append(avg_val_loss)

        epoch_val_probs = torch.cat(epoch_val_probs)
        epoch_val_preds = torch.cat(epoch_val_preds)
        epoch_val_labels = torch.cat(epoch_val_labels)

        val_acc = balanced_accuracy_score(epoch_val_labels, epoch_val_preds)
        val_f1 = f1_score(epoch_val_labels, epoch_val_preds)
        precision, recall, _ = precision_recall_curve(epoch_val_labels, epoch_val_probs)
        val_auc_pr = auc(recall, precision)

        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_auc_pr_history.append(val_auc_pr)

        # Validation distance correlation
        val_hidden_activations_all = torch.cat(val_hidden_activations_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)
        val_hidden_activations_np = val_hidden_activations_all.numpy()
        val_targets_np = val_targets_all.numpy()
        val_dcor_value = dcor.distance_correlation_sqr(val_hidden_activations_np, val_targets_np)
        val_dcor_history.append(val_dcor_value)

        val_precision = precision_score(epoch_val_labels, epoch_val_preds)
        val_recall = recall_score(epoch_val_labels, epoch_val_preds)
        # Append to lists if you want to track over epochs
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        # Compute confusion matrix for train phase
        val_conf_matrix = confusion_matrix(epoch_val_labels, epoch_val_preds)
        # Append to the list if tracking over epochs
        val_conf_matrices.append(val_conf_matrix)


        # Test phase
        epoch_test_loss = 0
        epoch_test_preds = []
        epoch_test_labels = []
        epoch_test_probs = []
        test_hidden_activations_list = []
        test_targets_list = []

        with torch.no_grad():
            for x_batch, y_batch in data_all_test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                predicted_disease = model.disease_classifier(encoded_features)
                c_loss = criterion_disease_classifier(predicted_disease, y_batch)
                epoch_test_loss += c_loss.item()

                pred_prob = torch.sigmoid(predicted_disease).detach().cpu()
                epoch_test_probs.extend(pred_prob)
                pred_tag = (pred_prob > 0.5).float()
                epoch_test_preds.append(pred_tag.cpu())
                epoch_test_labels.append(y_batch.cpu())

            for x_batch, y_batch in data_test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                test_hidden_activations_list.append(encoded_features.cpu())
                test_targets_list.append(y_batch.cpu())

        avg_test_loss = epoch_test_loss / len(data_all_test_loader)
        test_loss_history.append(avg_test_loss)

        epoch_test_probs = torch.cat(epoch_test_probs)
        epoch_test_preds = torch.cat(epoch_test_preds)
        epoch_test_labels = torch.cat(epoch_test_labels)

        test_acc = balanced_accuracy_score(epoch_test_labels, epoch_test_preds)
        test_f1 = f1_score(epoch_test_labels, epoch_test_preds)
        precision, recall, _ = precision_recall_curve(epoch_test_labels, epoch_test_probs)
        test_auc_pr = auc(recall, precision)

        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_auc_pr_history.append(test_auc_pr)

        # Test distance correlation
        test_hidden_activations_all = torch.cat(test_hidden_activations_list, dim=0)
        test_targets_all = torch.cat(test_targets_list, dim=0)
        test_hidden_activations_np = test_hidden_activations_all.numpy()
        test_targets_np = test_targets_all.numpy()
        test_dcor_value = dcor.distance_correlation_sqr(test_hidden_activations_np, test_targets_np)
        test_dcor_history.append(test_dcor_value)
        test_precision = precision_score(epoch_test_labels, epoch_test_preds)
        test_recall = recall_score(epoch_test_labels, epoch_test_preds)
        # Append to lists if you want to track over epochs
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        # Compute confusion matrix for train phase
        test_conf_matrix = confusion_matrix(epoch_test_labels, epoch_test_preds)
        # Append to the list if tracking over epochs
        test_conf_matrices.append(test_conf_matrix)


        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, DCor: {dcor_value:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}, Validation F1: {val_f1:.4f}, Val DCor: {val_dcor_value:.4f}')
            print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test DCor: {test_dcor_value:.4f}')

    return (loss_history, dcor_history, disease_loss_history, train_disease_accs,
            train_disease_f1s, train_auc_pr_history, train_precisions, train_recalls, train_conf_matrices,
            val_loss_history, val_accs, val_f1s, val_dcor_history, val_auc_pr_history, val_precisions, val_recalls, val_conf_matrices,
            test_loss_history, test_accs, test_f1s, test_dcor_history, test_auc_pr_history, test_precisions, test_recalls, test_conf_matrices)
