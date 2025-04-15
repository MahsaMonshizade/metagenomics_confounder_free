#!/usr/bin/env python3
import copy
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from config import config
from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model
from sklearn.model_selection import StratifiedKFold

# Use the device as specified in the config.
device = torch.device(config["training"].get("device", "cpu"))

def run_trial(trial_config, num_epochs=50):
    """
    Run a training trial using 5-fold cross-validation over the training data while
    evaluating on an independent test set. Returns the average test recall across all folds.

    NOTE: The independent test dataset is loaded from config's test paths.
    """
    # Extract configuration sections.
    data_cfg = trial_config["data"]
    train_cfg = trial_config["training"]
    model_cfg = trial_config["model"]

    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]

    # Load training data from the training paths.
    merged_train = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"])
    # Load independent test data from test paths.
    merged_test = get_data(data_cfg["test_abundance_path"], data_cfg["test_metadata_path"])

    # Define feature columns (exclude metadata columns and 'SampleID').
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_train.columns if col not in metadata_columns and col != "SampleID"]

    # Dynamically set input_size from the data.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")

    # Overall disease labels for training.
    X = merged_train[feature_columns].values
    merged_train['combined'] = (
        merged_train[disease_col].astype(str) +
        merged_train[confounder_col].astype(str)
    )
    y_all = merged_train["combined"].values

    # Create independent test DataLoaders (these remain fixed across folds).
    x_test_all = torch.tensor(merged_test[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test[disease_col].values, dtype=torch.float32).unsqueeze(1)
    test_data_disease = merged_test[merged_test[disease_col] == 1]
    x_test_disease = torch.tensor(test_data_disease[feature_columns].values, dtype=torch.float32)
    y_test_disease = torch.tensor(test_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
    batch_size = train_cfg["batch_size"]
    # Create test DataLoaders.
    data_test_loader = create_stratified_dataloader(x_test_disease, y_test_disease, batch_size)
    data_all_test_loader = create_stratified_dataloader(x_test_all, y_test_all, batch_size)

    # Use 5-fold stratified cross-validation on the training data.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_test_recalls = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Trial fold {fold+1} out of 5")
        
        # Prepare training and validation subsets.
        train_data = merged_train.iloc[train_index]
        val_data   = merged_train.iloc[val_index]  # (validation data is used only for training dynamics; evaluation is done on the independent test set)
        
        # Overall training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val   = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val   = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        
        # Confounder (drug) classification data: only samples with disease == 1.
        train_data_disease = train_data[train_data[disease_col] == 1]
        val_data_disease   = val_data[val_data[disease_col] == 1]
        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        x_disease_val   = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_val   = torch.tensor(val_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        
        # Create stratified DataLoaders for training and validation.
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        
        # Compute class weights for training.
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)
        
        num_pos_drug = y_disease_train.sum().item()
        num_neg_drug = len(y_disease_train) - num_pos_drug
        pos_weight_value_drug = num_neg_drug / num_pos_drug
        pos_weight_drug = torch.tensor([pos_weight_value_drug], dtype=torch.float32).to(device)
        
        # Build the model.
        model = GAN(
            input_size=input_size,
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"],
            activation=model_cfg["activation"]
        ).to(device)
        
        # Define loss functions.
        criterion = PearsonCorrelationLoss().to(device)
        criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drug).to(device)
        criterion_disease_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_disease).to(device)
        
        # Define optimizers.
        optimizer = optim.AdamW(model.encoder.parameters(), lr=train_cfg["encoder_lr"], weight_decay=train_cfg["weight_decay"])
        optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=train_cfg["classifier_lr"], weight_decay=train_cfg["weight_decay"])
        optimizer_disease_classifier = optim.AdamW(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
            lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
        )
        
        # Train the model using the three-phase training routine.
        Results = train_model(
            model, criterion, optimizer,
            data_loader, data_all_loader,
            data_val_loader, data_all_val_loader,
            data_test_loader, data_all_test_loader,   # Evaluate on the independent test data
            num_epochs,
            criterion_classifier, optimizer_classifier,
            criterion_disease_classifier, optimizer_disease_classifier,
            device
        )
        
        # Get the final test recall from this fold.
        fold_test_recall = Results["test"]["recall"][-1]
        print(f"Fold {fold+1} test recall: {fold_test_recall:.4f}")
        fold_test_recalls.append(fold_test_recall)
    
    # Compute the average test recall over all folds.
    avg_test_recall = np.mean(fold_test_recalls)
    print(f"Average test recall over 5 folds: {avg_test_recall:.4f}")
    return avg_test_recall

def objective(trial):
    """
    Objective function for Optuna: Suggest discrete hyperparameter values,
    update the base configuration, run a training trial using 5-fold cross-validation
    with evaluation on an independent test dataset, and return the average test recall.
    """
    trial_config = copy.deepcopy(config)
    
    # Set hyperparameters via categorical suggestions.
    trial_config["model"]["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", [1, 2, 3])
    trial_config["model"]["num_classifier_layers"] = trial.suggest_categorical("num_classifier_layers", [1, 2, 3])
    trial_config["model"]["dropout_rate"] = trial.suggest_categorical("dropout_rate", [0.0, 0.3, 0.5])
    trial_config["training"]["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 5e-4, 1e-3])
    trial_config["training"]["encoder_lr"] = trial.suggest_categorical("encoder_lr", [1e-5, 1e-4, 5e-4, 1e-3])
    trial_config["training"]["classifier_lr"] = trial.suggest_categorical("classifier_lr", [1e-5, 1e-4, 5e-4, 1e-3])
    trial_config["model"]["activation"] = trial.suggest_categorical("activation", config["tuning"]["activation"])
    
    # Run the trial with a reduced number of epochs for speed.
    avg_test_recall = run_trial(trial_config, num_epochs=50)
    return avg_test_recall

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Final test recall:", best_trial.value)
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
