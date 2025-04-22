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

# Use the device as specified in the configuration.
device = torch.device(config["training"].get("device", "cpu"))

def run_trial(trial_config, num_epochs=50):
    """
    Run a training trial using 5-fold cross-validation on the training set and return the average validation accuracy.

    This function:
      - Loads the training data from the paths in config.
      - Computes the feature columns and thus the input size dynamically.
      - Uses the column names for disease and confounder as given in config.
      - Splits the training data into 5 folds.
      - For each fold, trains the model (using the three-phase training routine) and evaluates the validation performance.
      - Returns the average validation accuracy (from the 'val' branch) across all folds.
    """
    # Extract configuration sections.
    data_cfg = trial_config["data"]
    train_cfg = trial_config["training"]
    model_cfg = trial_config["model"]
    
    # Load merged training data.
    merged_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"])
    
    # Get column names from config (must exist in the config file).
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]
    
    # Determine feature columns (exclude metadata and SampleID).
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Compute the input size dynamically.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")
    
    # Overall disease labels.
    X = merged_data_all[feature_columns].values
    merged_data_all['combined'] = (
        merged_data_all[disease_col].astype(str) +
        merged_data_all[confounder_col].astype(str)
    )
    y_all = merged_data_all["combined"].values
    
    # Prepare to aggregate validation accuracies.
    fold_val_accuracies = []
    
    # Set up 5-fold stratified cross-validation on the training data.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Running fold {fold+1} of 5")
        # Split into training and validation subsets.
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]
        
        # Overall disease training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        
        # Confounder (drug/sex) training data: use only samples where disease == 1.
        train_data_disease = train_data[train_data[disease_col] == 1]
        val_data_disease = val_data[val_data[disease_col] == 1]
        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        x_disease_val = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_val = torch.tensor(val_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        
        batch_size = train_cfg["batch_size"]
        
        # Create stratified DataLoaders.
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        
        # Compute class weights.
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
        # Here, both the confounder branch and disease branch are trained.
        Results = train_model(
            model, criterion, optimizer,
            data_loader, data_all_loader, data_val_loader, data_all_val_loader,
            data_val_loader, data_all_val_loader, num_epochs,
            criterion_classifier, optimizer_classifier,
            criterion_disease_classifier, optimizer_disease_classifier,
            device
        )
        
        # Extract the final validation accuracy from this fold.
        fold_val_acc = Results["val"]["accuracy"][-1]
        print(f"Fold {fold+1} validation accuracy: {fold_val_acc:.4f}")
        fold_val_accuracies.append(fold_val_acc)
    
    # Compute the average validation accuracy over all folds.
    avg_val_accuracy = np.mean(fold_val_accuracies)
    print(f"Average validation accuracy over 5 folds: {avg_val_accuracy:.4f}")
    return avg_val_accuracy

def objective(trial):
    """
    Objective function for Optuna: Suggest discrete hyperparameter values,
    update the base configuration, run a training trial using 5-fold cross-validation
    (evaluating on the validation dataset), and return the average validation accuracy.
    """
    trial_config = copy.deepcopy(config)
    
    trial_config["model"]["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", [1, 2, 3])
    trial_config["model"]["num_classifier_layers"] = trial.suggest_categorical("num_classifier_layers", [1, 2, 3])
    trial_config["model"]["dropout_rate"] = trial.suggest_categorical("dropout_rate", [0.0, 0.3, 0.5])
    # trial_config["training"]["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 5e-4, 1e-3])
    trial_config["training"]["encoder_lr"] = trial.suggest_categorical("encoder_lr", [1e-5, 1e-4, 5e-4, 1e-3])
    trial_config["training"]["classifier_lr"] = trial.suggest_categorical("classifier_lr", [1e-5, 1e-4, 5e-4, 1e-3])
    trial_config["model"]["activation"] = trial.suggest_categorical("activation", config["tuning"]["activation"])
    trial_config["model"]["latent_dim"] = trial.suggest_categorical("latent_dim", config["tuning"]["latent_dim"])
    trial_config["training"]["batch_size"] = trial.suggest_categorical("batch_size", config["tuning"]["batch_size"])
    trial_config["model"]["norm"] = trial.suggest_categorical("norm", config["tuning"]["norm"])
    trial_config["model"]["last_activation"] = trial.suggest_categorical("last_activation", config["tuning"]["last_activation"])
    
    # Run the trial with a reduced number of epochs for speed.
    avg_val_accuracy = run_trial(trial_config, num_epochs=50)
    return avg_val_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Final validation accuracy:", best_trial.value)
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
