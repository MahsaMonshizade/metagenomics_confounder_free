import torch
import numpy as np
import dcor
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, auc,
    precision_score, recall_score, confusion_matrix
)

def train_model(
    model, criterion, optimizer, data_loader, data_all_loader, data_val_loader, data_all_val_loader,
    data_test_loader, data_all_test_loader, num_epochs, criterion_classifier, optimizer_classifier,
    criterion_disease_classifier, optimizer_disease_classifier, device
):
    """
    Train the model with the given data loaders and hyperparameters.
    """

    # Initialize dictionaries to store training, validation, and test metrics
    results = {
        'train': {
            'gloss_history': [],
            'loss_history': [],
            'dcor_history': [],
            'accuracy': [],
            'f1_score': [],
            'auc_pr': [],
            'precision': [],
            'recall': [],
            'confusion_matrix': []
        },
        'val': {
            'loss_history': [],
            'dcor_history': [],
            'accuracy': [],
            'f1_score': [],
            'auc_pr': [],
            'precision': [],
            'recall': [],
            'confusion_matrix': []
        },
        'test': {
            'loss_history': [],
            'dcor_history': [],
            'accuracy': [],
            'f1_score': [],
            'auc_pr': [],
            'precision': [],
            'recall': [],
            'confusion_matrix': []
        }
    }

    model = model.to(device)
    criterion = criterion.to(device)
    criterion_classifier = criterion_classifier.to(device)
    criterion_disease_classifier = criterion_disease_classifier.to(device)


    for epoch in range(num_epochs):
        model.train()

        epoch_gloss = 0
        epoch_train_loss = 0
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_probs = []
        hidden_activations_list = []
        targets_list = []

        data_iter = iter(data_loader)
        data_all_iter = iter(data_all_loader)

        
        while True:
            try:
                x_all_batch, y_all_batch = next(data_all_iter)
                x_all_batch, y_all_batch = x_all_batch.to(device), y_all_batch.to(device)

                try:
                    x_batch, y_batch = next(data_iter)
                except StopIteration:
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
                predicted_drug = torch.sigmoid(model.classifier(encoded_features))
                g_loss = criterion(predicted_drug, y_batch)
                hidden_activations_list.append(encoded_features.detach().cpu())
                targets_list.append(y_batch.detach().cpu())
                optimizer.zero_grad()
                g_loss.backward()
                optimizer.step()
                epoch_gloss += g_loss.item()
                for param in model.classifier.parameters():
                    param.requires_grad = True

                # Train encoder & disease classifier (c_loss)
                encoded_features_all = model.encoder(x_all_batch)
                predicted_disease_all = model.disease_classifier(encoded_features_all)
                c_loss = criterion_disease_classifier(predicted_disease_all, y_all_batch)
                optimizer_disease_classifier.zero_grad()
                c_loss.backward()
                optimizer_disease_classifier.step()
                epoch_train_loss += c_loss.item()
                pred_prob = torch.sigmoid(predicted_disease_all).detach().cpu()
                epoch_train_probs.extend(pred_prob)
                pred_tag = (pred_prob > 0.5).float()
                epoch_train_preds.append(pred_tag.cpu())
                epoch_train_labels.append(y_all_batch.cpu())

            except StopIteration:
                break

        # Compute training metrics
        avg_gloss = epoch_gloss / len(data_all_loader)
        avg_train_loss = epoch_train_loss / len(data_all_loader)
        results['train']['gloss_history'].append(avg_gloss)
        results['train']['loss_history'].append(avg_train_loss)

        epoch_train_probs = torch.cat(epoch_train_probs)
        epoch_train_preds = torch.cat(epoch_train_preds)
        epoch_train_labels = torch.cat(epoch_train_labels)

        train_acc = balanced_accuracy_score(epoch_train_labels, epoch_train_preds)
        results['train']['accuracy'].append(train_acc)

        train_f1 = f1_score(epoch_train_labels, epoch_train_preds)
        results['train']['f1_score'].append(train_f1)

        precision, recall, _ = precision_recall_curve(epoch_train_labels, epoch_train_probs)
        train_auc_pr = auc(recall, precision)
        results['train']['auc_pr'].append(train_auc_pr)

        train_precision = precision_score(epoch_train_labels, epoch_train_preds)
        train_recall = recall_score(epoch_train_labels, epoch_train_preds)
        results['train']['precision'].append(train_precision)
        results['train']['recall'].append(train_recall)

        train_conf_matrix = confusion_matrix(epoch_train_labels, epoch_train_preds)
        results['train']['confusion_matrix'].append(train_conf_matrix)

        hidden_activations_all = torch.cat(hidden_activations_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
        dcor_value = dcor.distance_correlation_sqr(hidden_activations_all.numpy(), targets_all.numpy())
        results['train']['dcor_history'].append(dcor_value)


        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_preds = []
        epoch_val_labels = []
        epoch_val_probs = []
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
        results['val']['loss_history'].append(avg_val_loss)

        epoch_val_probs = torch.cat(epoch_val_probs)
        epoch_val_preds = torch.cat(epoch_val_preds)
        epoch_val_labels = torch.cat(epoch_val_labels)

        val_acc = balanced_accuracy_score(epoch_val_labels, epoch_val_preds)
        val_f1 = f1_score(epoch_val_labels, epoch_val_preds)
        precision, recall, _ = precision_recall_curve(epoch_val_labels, epoch_val_probs)
        val_auc_pr = auc(recall, precision)

        results['val']['accuracy'].append(val_acc)
        results['val']['f1_score'].append(val_f1)
        results['val']['auc_pr'].append(val_auc_pr)

        val_hidden_activations_all = torch.cat(val_hidden_activations_list, dim=0)
        val_targets_all = torch.cat(val_targets_list, dim=0)
        val_dcor_value = dcor.distance_correlation_sqr(val_hidden_activations_all.numpy(), val_targets_all.numpy())
        results['val']['dcor_history'].append(val_dcor_value)

        val_precision = precision_score(epoch_val_labels, epoch_val_preds)
        val_recall = recall_score(epoch_val_labels, epoch_val_preds)
        results['val']['precision'].append(val_precision)
        results['val']['recall'].append(val_recall)

        val_conf_matrix = confusion_matrix(epoch_val_labels, epoch_val_preds)
        results['val']['confusion_matrix'].append(val_conf_matrix)


        # Test
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
        results['test']['loss_history'].append(avg_test_loss)

        epoch_test_probs = torch.cat(epoch_test_probs)
        epoch_test_preds = torch.cat(epoch_test_preds)
        epoch_test_labels = torch.cat(epoch_test_labels)

        test_acc = balanced_accuracy_score(epoch_test_labels, epoch_test_preds)
        test_f1 = f1_score(epoch_test_labels, epoch_test_preds)
        precision, recall, _ = precision_recall_curve(epoch_test_labels, epoch_test_probs)
        test_auc_pr = auc(recall, precision)

        results['test']['accuracy'].append(test_acc)
        results['test']['f1_score'].append(test_f1)
        results['test']['auc_pr'].append(test_auc_pr)

        test_hidden_activations_all = torch.cat(test_hidden_activations_list, dim=0)
        test_targets_all = torch.cat(test_targets_list, dim=0)
        test_dcor_value = dcor.distance_correlation_sqr(test_hidden_activations_all.numpy(), test_targets_all.numpy())
        results['test']['dcor_history'].append(test_dcor_value)

        test_precision = precision_score(epoch_test_labels, epoch_test_preds)
        test_recall = recall_score(epoch_test_labels, epoch_test_preds)
        results['test']['precision'].append(test_precision)
        results['test']['recall'].append(test_recall)

        test_conf_matrix = confusion_matrix(epoch_test_labels, epoch_test_preds)
        results['test']['confusion_matrix'].append(test_conf_matrix)

        if (epoch + 1) % 50 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_gloss:.4f}, DCor: {dcor_value:.4f}'
            )
            print(
                f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}, '
                f'Validation F1: {val_f1:.4f}, Val DCor: {val_dcor_value:.4f}'
            )
            print(
                f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, '
                f'Test F1: {test_f1:.4f}, Test DCor: {test_dcor_value:.4f}'
            )
        

    return results
