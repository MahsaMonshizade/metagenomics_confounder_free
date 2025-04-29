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
    model, optimizer, data_all_loader, data_all_val_loader,
    data_all_test_loader, num_epochs, 
    criterion_classifier,
    criterion_disease_classifier, device
):
    """
    Train the model using a three-phase procedure:
      - Phase 1: Drug/Confounder classification training (r_loss) using data_loader.
      - Phase 2: Distillation training (g_loss) with PearsonCorrelationLoss.
      - Phase 3: Disease classification training (c_loss) using data_all_loader.
      
    Metrics for training, validation, and test phases are stored.
    """
    # Initialize results dictionary to store metric histories.
    results = {
        "train": {
            # "gloss_history": [],      # g_loss: distillation phase loss
            "loss_history": [],       # c_loss: disease classification loss
            # "dcor_history": [],       # Distance correlation measure
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "val": {
            "loss_history": [],
            # "dcor_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        },
        "test": {
            "loss_history": [],
            # "dcor_history": [],
            "accuracy": [],
            "f1_score": [],
            "auc_pr": [],
            "precision": [],
            "recall": [],
            "confusion_matrix": []
        }
    }

    # Move model and loss functions to device.
    model = model.to(device)
    criterion_classifier = criterion_classifier.to(device)
    criterion_disease_classifier = criterion_disease_classifier.to(device)

    # Begin epoch loop.
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_preds_disease = []
        epoch_train_preds_confounder = []
        epoch_train_labels_disease = []
        epoch_train_labels_confounder = []
        epoch_train_probs_disease = []
        epoch_train_probs_confounder = []
        
        # Training phase: Using manual iterator and StopIteration handling.
        data_all_iter = iter(data_all_loader)
        while True:
            try:
                x_batch, y_batch_disease, y_batch_confounder = next(data_all_iter)
            except StopIteration:
                break
            x_batch, y_batch_disease, y_batch_confounder = x_batch.to(device), y_batch_disease.to(device), y_batch_confounder.to(device)
            encoded_features = model.encoder(x_batch)


            predicted_confounder = model.classifier(encoded_features)   
            prob_confounder   = torch.sigmoid(predicted_confounder)  
            pred_tag_confounder   = (prob_confounder > 0.5).float()            
            # 4) prepare true confounder labels
            y_true_conf = (y_batch_confounder.unsqueeze(1)
                        if y_batch_confounder.dim()==1 
                        else y_batch_confounder)          # (B,1)
            # 5) mask & fill NaNs
            nan_mask   = torch.isnan(y_true_conf)              # (B,1) bool
            conf_input = torch.where(nan_mask, pred_tag_confounder, y_true_conf)
            # 6) concat & predict disease
            cat_input    = torch.cat([encoded_features, conf_input], dim=1)  # (B, latent_dim+1)
            predicted_disease = model.disease_classifier(cat_input)
            loss_confounder = criterion_classifier(predicted_confounder, y_batch_confounder)
            loss_disease = criterion_disease_classifier(predicted_disease, y_batch_disease)
            loss = loss_confounder + loss_disease
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
            prob_disease = torch.sigmoid(predicted_disease).detach().cpu()
            epoch_train_probs_disease.append(prob_disease)
            epoch_train_probs_confounder.append(prob_confounder)
            pred_tag_disease = (prob_disease > 0.5).float()
            epoch_train_preds_disease.append(pred_tag_disease)
            epoch_train_preds_confounder.append(pred_tag_confounder)
            epoch_train_labels_disease.append(y_batch_disease.cpu())
            epoch_train_labels_confounder.append(y_batch_confounder.cpu())
        
        avg_train_loss = epoch_train_loss / len(data_all_loader)
        results["train"]["loss_history"].append(avg_train_loss)
        epoch_train_probs_disease = torch.cat(epoch_train_probs_disease)
        epoch_train_probs_confounder = torch.cat(epoch_train_probs_confounder)
        epoch_train_preds_disease = torch.cat(epoch_train_preds_disease)
        epoch_train_preds_confounder = torch.cat(epoch_train_preds_confounder)
        epoch_train_labels_disease = torch.cat(epoch_train_labels_disease)
        epoch_train_labels_confounder = torch.cat(epoch_train_labels_confounder)
        
        train_acc = balanced_accuracy_score(epoch_train_labels_disease, epoch_train_preds_disease)
        train_f1 = f1_score(epoch_train_labels_disease, epoch_train_preds_disease)
        precision, recall, _ = precision_recall_curve(epoch_train_labels_disease, epoch_train_probs_disease)
        train_auc_pr = auc(recall, precision)
        train_precision = precision_score(epoch_train_labels_disease, epoch_train_preds_disease)
        train_recall = recall_score(epoch_train_labels_disease, epoch_train_preds_disease)
        train_conf_matrix = confusion_matrix(epoch_train_labels_disease, epoch_train_preds_disease)
        
        results["train"]["accuracy"].append(train_acc)
        results["train"]["f1_score"].append(train_f1)
        results["train"]["auc_pr"].append(train_auc_pr)
        results["train"]["precision"].append(train_precision)
        results["train"]["recall"].append(train_recall)
        results["train"]["confusion_matrix"].append(train_conf_matrix)
        
        # Validation phase.
        model.eval()
        epoch_val_loss = 0
        epoch_val_preds_disease = []
        epoch_val_preds_confounder = []
        epoch_val_labels_disease = []
        epoch_val_labels_confounder = []
        epoch_val_probs_disease = []
        epoch_val_probs_confounder = []
        with torch.no_grad():
            for x_batch, y_batch_disease, y_batch_confounder in data_all_val_loader:
                x_batch, y_batch_disease, y_batch_confounder = x_batch.to(device), y_batch_disease.to(device), y_batch_confounder.to(device)
                encoded_features = model.encoder(x_batch)

                predicted_confounder = model.classifier(encoded_features)   
                prob_confounder   = torch.sigmoid(predicted_confounder)  
                pred_tag_confounder   = (prob_confounder > 0.5).float()            
                # 4) prepare true confounder labels
                y_true_conf = (y_batch_confounder.unsqueeze(1)
                            if y_batch_confounder.dim()==1 
                            else y_batch_confounder)          # (B,1)
                # 5) mask & fill NaNs
                nan_mask   = torch.isnan(y_true_conf)              # (B,1) bool
                conf_input = torch.where(nan_mask, pred_tag_confounder, y_true_conf)
                # 6) concat & predict disease
                cat_input    = torch.cat([encoded_features, conf_input], dim=1)  # (B, latent_dim+1)

                predicted_disease = model.disease_classifier(cat_input)
                loss_confounder = criterion_classifier(predicted_confounder, y_batch_confounder)
                loss_disease = criterion_disease_classifier(predicted_disease, y_batch_disease)
                loss = loss_confounder + loss_disease
                epoch_val_loss += loss.item()
                prob_disease = torch.sigmoid(predicted_disease).detach().cpu()
                epoch_val_probs_disease.append(prob_disease)
                epoch_val_probs_confounder.append(prob_confounder)
                pred_tag_disease = (prob_disease > 0.5).float()
                pred_tag_confounder = (prob_confounder > 0.5).float()
                epoch_val_preds_disease.append(pred_tag_disease)
                epoch_val_preds_confounder.append(pred_tag_confounder)
                epoch_val_labels_disease.append(y_batch_disease.cpu())
                epoch_val_labels_confounder.append(y_batch_confounder.cpu())
        
        avg_val_loss = epoch_val_loss / len(data_all_val_loader)
        results["val"]["loss_history"].append(avg_val_loss)
        epoch_val_probs_disease = torch.cat(epoch_val_probs_disease)
        epoch_val_probs_confounder = torch.cat(epoch_val_probs_confounder)
        epoch_val_preds_disease = torch.cat(epoch_val_preds_disease)
        epoch_val_preds_confounder = torch.cat(epoch_val_preds_confounder)
        epoch_val_labels_disease = torch.cat(epoch_val_labels_disease)
        epoch_val_labels_confounder = torch.cat(epoch_val_labels_confounder)
        
        val_acc = balanced_accuracy_score(epoch_val_labels_disease, epoch_val_preds_disease)
        val_f1 = f1_score(epoch_val_labels_disease, epoch_val_preds_disease)
        precision, recall, _ = precision_recall_curve(epoch_val_labels_disease, epoch_val_probs_disease)
        val_auc_pr = auc(recall, precision)
        val_precision = precision_score(epoch_val_labels_disease, epoch_val_preds_disease)
        val_recall = recall_score(epoch_val_labels_disease, epoch_val_preds_disease)
        val_conf_matrix = confusion_matrix(epoch_val_labels_disease, epoch_val_preds_disease)
        
        results["val"]["accuracy"].append(val_acc)
        results["val"]["f1_score"].append(val_f1)
        results["val"]["auc_pr"].append(val_auc_pr)
        results["val"]["precision"].append(val_precision)
        results["val"]["recall"].append(val_recall)
        results["val"]["confusion_matrix"].append(val_conf_matrix)
        
        # Test phase.
        epoch_test_loss = 0
        epoch_test_preds_disease = []
        epoch_test_preds_confounder = []
        epoch_test_labels_disease = []
        epoch_test_labels_confounder = []
        epoch_test_probs_disease = []
        epoch_test_probs_confounder = []
        with torch.no_grad():
            for x_batch, y_batch_disease, y_batch_confounder in data_all_test_loader:
                x_batch, y_batch_disease, y_batch_confounder = x_batch.to(device), y_batch_disease.to(device), y_batch_confounder.to(device)
                encoded_features = model.encoder(x_batch)

                predicted_confounder = model.classifier(encoded_features)   
                prob_confounder   = torch.sigmoid(predicted_confounder)  
                pred_tag_confounder   = (prob_confounder > 0.5).float()            
                # 4) prepare true confounder labels
                y_true_conf = (y_batch_confounder.unsqueeze(1)
                            if y_batch_confounder.dim()==1 
                            else y_batch_confounder)          # (B,1)
                # 5) mask & fill NaNs
                nan_mask   = torch.isnan(y_true_conf)              # (B,1) bool
                conf_input = torch.where(nan_mask, pred_tag_confounder, y_true_conf)
                # 6) concat & predict disease
                cat_input    = torch.cat([encoded_features, conf_input], dim=1)  # (B, latent_dim+1)

                predicted_disease = model.disease_classifier(cat_input)
                loss_confounder = criterion_classifier(predicted_confounder, y_batch_confounder)
                loss_disease = criterion_disease_classifier(predicted_disease, y_batch_disease)
                loss = loss_confounder + loss_disease
                epoch_test_loss += loss.item()
                prob_disease = torch.sigmoid(predicted_disease).detach().cpu()
                epoch_test_probs_disease.append(prob_disease)
                epoch_test_probs_confounder.append(prob_confounder)
                pred_tag_disease = (prob_disease > 0.5).float()
                epoch_test_preds_disease.append(pred_tag_disease)
                epoch_test_preds_confounder.append(pred_tag_confounder)
                epoch_test_labels_disease.append(y_batch_disease.cpu())
                epoch_test_labels_confounder.append(y_batch_confounder.cpu())
                
        avg_test_loss = epoch_test_loss / len(data_all_test_loader)
        results["test"]["loss_history"].append(avg_test_loss)
        epoch_test_probs_disease = torch.cat(epoch_test_probs_disease)
        epoch_test_probs_confounder = torch.cat(epoch_test_probs_confounder)
        epoch_test_preds_disease = torch.cat(epoch_test_preds_disease)
        epoch_test_preds_confounder = torch.cat(epoch_test_preds_confounder)
        epoch_test_labels_disease = torch.cat(epoch_test_labels_disease)
        epoch_test_labels_confounder = torch.cat(epoch_test_labels_confounder)
        
        test_acc = balanced_accuracy_score(epoch_test_labels_disease, epoch_test_preds_disease)
        test_f1 = f1_score(epoch_test_labels_disease, epoch_test_preds_disease)
        precision, recall, _ = precision_recall_curve(epoch_test_labels_disease, epoch_test_probs_disease)
        test_auc_pr = auc(recall, precision)
        test_precision = precision_score(epoch_test_labels_disease, epoch_test_preds_disease)
        test_recall = recall_score(epoch_test_labels_disease, epoch_test_preds_disease)
        test_conf_matrix = confusion_matrix(epoch_test_labels_disease, epoch_test_preds_disease)
        
        results["test"]["accuracy"].append(test_acc)
        results["test"]["f1_score"].append(test_f1)
        results["test"]["auc_pr"].append(test_auc_pr)
        results["test"]["precision"].append(test_precision)
        results["test"]["recall"].append(test_recall)
        results["test"]["confusion_matrix"].append(test_conf_matrix)
        
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}]  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}]  Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    return results
