import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix

def train_model(model, data_all_loader, data_all_val_loader, data_all_test_loader, num_epochs, criterion_disease_classifier, optimizer_disease_classifier, device,  pretrained_model_path=None):
    """
    Train the model while recording metrics for training, validation, and testing.
    """
    results = {
        "train": {
            "loss_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "val": {
            "loss_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "test": {
            "loss_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "best_test": {
            "epoch": None, 
            "sample_id": [], 
            "pred_probs": [],
            "labels": [],
            "accuracy": 0.0,  
        },  # DELONG: Store best predictions for test set
    }

    model = model.to(device)
    criterion_disease_classifier = criterion_disease_classifier.to(device)

     # Load the pretrained model if provided
    if pretrained_model_path: 
        state_dict = torch.load(pretrained_model_path, map_location=device, weights_only=True)
        
        # Create a filtered state dict containing only encoder and classifier weights
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Only keep parameters for encoder and classifier
            if key.startswith('encoder.'):
                filtered_state_dict[key] = value
        
        # Load the filtered state dictionary
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained [encoder] and [classifier] weights from {pretrained_model_path}")

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_probs = []
        
        # Training phase: Using manual iterator and StopIteration handling.
        data_all_iter = iter(data_all_loader)
        while True:
            try:
                x_batch, y_batch = next(data_all_iter)
            except StopIteration:
                break
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            encoded_features = model.encoder(x_batch)
            predicted_disease = model.disease_classifier(encoded_features)
            loss = criterion_disease_classifier(predicted_disease, y_batch)
            optimizer_disease_classifier.zero_grad()
            loss.backward()
            optimizer_disease_classifier.step()
            epoch_train_loss += loss.item()
            
            prob = torch.sigmoid(predicted_disease).detach().cpu()
            epoch_train_probs.append(prob)
            pred_tag = (prob > 0.5).float()
            epoch_train_preds.append(pred_tag)
            epoch_train_labels.append(y_batch.cpu())
        
        avg_train_loss = epoch_train_loss / len(data_all_loader)
        results["train"]["loss_history"].append(avg_train_loss)
        epoch_train_probs = torch.cat(epoch_train_probs)
        epoch_train_preds = torch.cat(epoch_train_preds)
        epoch_train_labels = torch.cat(epoch_train_labels)
        
        train_acc = balanced_accuracy_score(epoch_train_labels, epoch_train_preds)
        train_f1 = f1_score(epoch_train_labels, epoch_train_preds, zero_division=0)
        precision, recall, _ = precision_recall_curve(epoch_train_labels.view(-1), epoch_train_probs)
        train_auc_pr = auc(recall, precision)
        train_precision = precision_score(epoch_train_labels, epoch_train_preds, zero_division=0)
        train_recall = recall_score(epoch_train_labels, epoch_train_preds, zero_division=0)
        train_conf_matrix = confusion_matrix(epoch_train_labels, epoch_train_preds)
        
        results["train"]["accuracy"].append(train_acc)
        results["train"]["f1_score"].append(train_f1)
        results["train"]["auc_pr"].append(train_auc_pr)
        results["train"]["precision"].append(train_precision)
        results["train"]["recall"].append(train_recall)
        results["train"]["confusion_matrix"].append(train_conf_matrix)
        
        # Validation phase.
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
                loss = criterion_disease_classifier(predicted_disease, y_batch)
                epoch_val_loss += loss.item()
                prob = torch.sigmoid(predicted_disease).detach().cpu()
                epoch_val_probs.append(prob)
                pred_tag = (prob > 0.5).float()
                epoch_val_preds.append(pred_tag)
                epoch_val_labels.append(y_batch.cpu())
        
        avg_val_loss = epoch_val_loss / len(data_all_val_loader)
        results["val"]["loss_history"].append(avg_val_loss)
        epoch_val_probs = torch.cat(epoch_val_probs)
        epoch_val_preds = torch.cat(epoch_val_preds)
        epoch_val_labels = torch.cat(epoch_val_labels)
        
        val_acc = balanced_accuracy_score(epoch_val_labels, epoch_val_preds)
        val_f1 = f1_score(epoch_val_labels, epoch_val_preds, zero_division=0)
        precision, recall, _ = precision_recall_curve(epoch_val_labels.view(-1), epoch_val_probs)
        val_auc_pr = auc(recall, precision)
        val_precision = precision_score(epoch_val_labels, epoch_val_preds, zero_division=0)
        val_recall = recall_score(epoch_val_labels, epoch_val_preds, zero_division=0)
        val_conf_matrix = confusion_matrix(epoch_val_labels, epoch_val_preds)
        
        results["val"]["accuracy"].append(val_acc)
        results["val"]["f1_score"].append(val_f1)
        results["val"]["auc_pr"].append(val_auc_pr)
        results["val"]["precision"].append(val_precision)
        results["val"]["recall"].append(val_recall)
        results["val"]["confusion_matrix"].append(val_conf_matrix)
        
        # Test phase.
        epoch_test_loss = 0
        epoch_test_preds = []
        epoch_test_labels = []
        epoch_test_probs = []
        epoch_test_sample_ids = []  # DELONG: Store sample IDs for test set
        with torch.no_grad():
            for x_batch, y_batch, id_batch in data_all_test_loader: # DELONG: Include sample IDs in test loader
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                encoded_features = model.encoder(x_batch)
                predicted_disease = model.disease_classifier(encoded_features)
                loss = criterion_disease_classifier(predicted_disease, y_batch)
                epoch_test_loss += loss.item()
                prob = torch.sigmoid(predicted_disease).detach().cpu()
                epoch_test_probs.append(prob)
                pred_tag = (prob > 0.5).float()
                epoch_test_preds.append(pred_tag)
                epoch_test_labels.append(y_batch.cpu())
                epoch_test_sample_ids.extend(id_batch) # DELONG: Store sample IDs for test set
                
        avg_test_loss = epoch_test_loss / len(data_all_test_loader)
        results["test"]["loss_history"].append(avg_test_loss)
        epoch_test_probs = torch.cat(epoch_test_probs)
        epoch_test_preds = torch.cat(epoch_test_preds)
        epoch_test_labels = torch.cat(epoch_test_labels)
        
        test_acc = balanced_accuracy_score(epoch_test_labels, epoch_test_preds)
        test_f1 = f1_score(epoch_test_labels, epoch_test_preds, zero_division=0)
        precision, recall, _ = precision_recall_curve(epoch_test_labels.view(-1), epoch_test_probs)
        test_auc_pr = auc(recall, precision)
        test_precision = precision_score(epoch_test_labels, epoch_test_preds, zero_division=0)
        test_recall = recall_score(epoch_test_labels, epoch_test_preds, zero_division=0)
        test_conf_matrix = confusion_matrix(epoch_test_labels, epoch_test_preds)
        
        results["test"]["accuracy"].append(test_acc)
        results["test"]["f1_score"].append(test_f1)
        results["test"]["auc_pr"].append(test_auc_pr)
        results["test"]["precision"].append(test_precision)
        results["test"]["recall"].append(test_recall)
        results["test"]["confusion_matrix"].append(test_conf_matrix)
        
        if test_acc > results["best_test"]["accuracy"]: # DELONG: Update best test resultsAdd commentMore actions
            results["best_test"]["epoch"] = epoch + 1
            results["best_test"]["pred_probs"] = epoch_test_probs
            results["best_test"]["labels"] = epoch_test_labels
            results["best_test"]["accuracy"] = test_acc
            results["best_test"]["sample_id"] = epoch_test_sample_ids
            
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}]  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}]  Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    return results
