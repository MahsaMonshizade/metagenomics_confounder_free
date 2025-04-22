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

# Use the device specified in the configuration.
device = torch.device(config["training"].get("device", "cpu"))

def run_trial(trial_config, num_epochs=10):
    """
    Run a single training trial using 5-fold cross-validation on the training data and return 
    the average validation accuracy across folds.

    This procedure:
      - Loads training data using the paths in the config.
      - Determines the feature columns dynamically (excluding metadata and SampleID).
      - Computes the input size from the training data.
      - Uses StratifiedKFold to split the data into 5 folds.
      - For each fold:
          • Prepares training and validation subsets.
          • Creates DataLoaders from the training and validation splits.
          • Computes class weights.
          • Builds a model (using hyperparameters from trial_config and with the chosen activation function).
          • Trains the model using your three-phase training routine.
          • Extracts the final validation accuracy for that fold.
      - Returns the average validation accuracy over all 5 folds.
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
    
    # Define feature columns (exclude metadata columns and 'SampleID').
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Compute the input size dynamically from the training data.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")
    
    # Overall disease classification.
    X = merged_data_all[feature_columns].values
    y_all = merged_data_all[disease_col].values  # Assumes the disease column is named "disease"
    
    # Set up 5-fold stratified cross-validation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_accuracies = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Running fold {fold+1} of 5")
        
        # Split into training and validation subsets.
        train_data = merged_data_all.iloc[train_index]
        val_data   = merged_data_all.iloc[val_index]
        
        # Overall disease training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val   = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val   = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        
        batch_size = train_cfg["batch_size"]
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        
        # Compute class weights.
        num_pos = y_all_train.sum().item()
        num_neg = len(y_all_train) - num_pos
        pos_weight_value = num_neg / num_pos
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
        
        # Build the model using hyperparameters from trial_config (including activation).
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
        
        # Define the loss function.
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        
        # Define the optimizer for the disease classification branch.
        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )
        
        # Train the model using your training routine.
        # Here, we pass the same DataLoader as both 'val' and 'test', since we are using validation performance.
        Results = train_model(
            model, data_all_loader, data_all_val_loader, data_all_val_loader,
            num_epochs, criterion, optimizer, device
        )
        
        fold_val_acc = Results["val"]["accuracy"][-1]
        print(f"Fold {fold+1} validation accuracy: {fold_val_acc:.4f}")
        val_accuracies.append(fold_val_acc)
    
    avg_val_accuracy = np.mean(val_accuracies)
    print(f"Average validation accuracy over 5 folds: {avg_val_accuracy:.4f}")
    return avg_val_accuracy

def objective(trial):
    """
    Objective function for Optuna: Suggest discrete hyperparameter values,
    update the configuration, run a training trial using 5-fold cross-validation,
    and return the average validation accuracy for maximization.
    """
    trial_config = copy.deepcopy(config)
    
    # Suggest categorical (discrete) hyperparameter values from the tuning space.
    trial_config["model"]["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", config["tuning"]["num_encoder_layers"])
    trial_config["model"]["num_classifier_layers"] = trial.suggest_categorical("num_classifier_layers", config["tuning"]["num_classifier_layers"])
    trial_config["model"]["dropout_rate"] = trial.suggest_categorical("dropout_rate", config["tuning"]["dropout_rate"])
    trial_config["training"]["learning_rate"] = trial.suggest_categorical("learning_rate", config["tuning"]["learning_rate"])
    trial_config["model"]["activation"] = trial.suggest_categorical("activation", config["tuning"]["activation"])
    
    # Run the trial with a reduced number of epochs for speed.
    avg_val_accuracy = run_trial(trial_config, num_epochs=10)
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
