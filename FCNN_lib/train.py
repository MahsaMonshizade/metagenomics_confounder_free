import torch
import numpy as np
import dcor
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_recall_curve, auc,
    precision_score, recall_score, confusion_matrix
)

def train_model(
   model, data_all_loader, data_all_val_loader,
            data_all_test_loader, num_epochs,
            criterion_disease_classifier, optimizer_disease_classifier,
            device
):
    """
    Train the model with the given data loaders and hyperparameters.
    """

    # Initialize dictionaries to store training, validation, and test metrics
    results = {
        'train': {
            'loss_history': [],
            'accuracy': [],
            'f1_score': [],
            'auc_pr': [],
            'precision': [],
            'recall': [],
            'confusion_matrix': []
        },
        'val': {
            'loss_history': [],
            'accuracy': [],
            'f1_score': [],
            'auc_pr': [],
            'precision': [],
            'recall': [],
            'confusion_matrix': []
        },
        'test': {
            'loss_history': [],
            'accuracy': [],
            'f1_score': [],
            'auc_pr': [],
            'precision': [],
            'recall': [],
            'confusion_matrix': []
        }
    }

    model = model.to(device)
    criterion_disease_classifier = criterion_disease_classifier.to(device)

    for epoch in range(num_epochs):
        model.train()

        epoch_train_loss = 0
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_probs = []
       

        data_all_iter = iter(data_all_loader)

        while True:
            try:
                x_all_batch, y_all_batch = next(data_all_iter)
                x_all_batch, y_all_batch = x_all_batch.to(device), y_all_batch.to(device)

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

        avg_train_loss = epoch_train_loss / len(data_all_loader)
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

      
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_preds = []
        epoch_val_labels = []
        epoch_val_probs = []
       

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


        test_precision = precision_score(epoch_test_labels, epoch_test_preds)
        test_recall = recall_score(epoch_test_labels, epoch_test_preds)
        results['test']['precision'].append(test_precision)
        results['test']['recall'].append(test_recall)

        test_conf_matrix = confusion_matrix(epoch_test_labels, epoch_test_preds)
        results['test']['confusion_matrix'].append(test_conf_matrix)

        if (epoch + 1) % 50 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}]'
            )
            print(
                f'train Loss: {avg_train_loss:.4f}, train Acc: {train_acc:.4f}, '
                f'train F1: {train_f1:.4f}'
            )
            print(
                f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}, '
                f'Validation F1: {val_f1:.4f}'
            )
            print(
                f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, '
                f'Test F1: {test_f1:.4f}'
            )

    return results
