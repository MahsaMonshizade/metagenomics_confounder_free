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

# Use the device defined in the configuration.
device = torch.device(config["training"].get("device", "cpu"))

def run_trial(trial_config, num_epochs=100):
    """
    Run a training trial using 5-fold cross-validation and return the average test accuracy.
    
    WARNING: This function uses the test dataset for optimization, which biases the final estimates.
    """
    # Extract configuration sections.
    data_cfg = trial_config["data"]
    train_cfg = trial_config["training"]
    model_cfg = trial_config["model"]
    
    # Load merged data (using the training files).
    merged_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"])
    
    # Define feature columns (excluding metadata and SampleID).
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Overall disease classification.
    X = merged_data_all[feature_columns].values
    y_all = merged_data_all["disease"].values
    
    # Prepare to aggregate test accuracies over folds.
    fold_test_accuracies = []
    
    # Define the cross-validation splitter.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Iterate over all folds.
    for fold, (train_index, test_index) in enumerate(skf.split(X, y_all)):
        print(f"Running fold {fold+1}")
        # Create training and test subsets (using the same logic as in main.py).
        train_data = merged_data_all.iloc[train_index]
        test_data = merged_data_all.iloc[test_index]
        
        # Overall disease classification training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data["disease"].values, dtype=torch.float32).unsqueeze(1)
        x_all_test = torch.tensor(test_data[feature_columns].values, dtype=torch.float32)
        y_all_test = torch.tensor(test_data["disease"].values, dtype=torch.float32).unsqueeze(1)
        
        # Prepare confounder (drug/sex) classification data: only samples with disease == 1.
        train_data_disease = train_data[train_data["disease"] == 1]
        test_data_disease  = test_data[test_data["disease"] == 1]
        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease["sex"].values, dtype=torch.float32).unsqueeze(1)
        x_disease_test  = torch.tensor(test_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_test  = torch.tensor(test_data_disease["sex"].values, dtype=torch.float32).unsqueeze(1)
        
        batch_size = train_cfg["batch_size"]
        
        # Create DataLoaders.
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        # Here, we use test data loaders for evaluation.
        data_test_loader = create_stratified_dataloader(x_disease_test, y_disease_test, batch_size)
        data_all_test_loader = create_stratified_dataloader(x_all_test, y_all_test, batch_size)
        
        # Compute class weights.
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)
        
        num_pos_drug = y_disease_train.sum().item()
        num_neg_drug = len(y_disease_train) - num_pos_drug
        pos_weight_value_drug = num_neg_drug / num_pos_drug
        pos_weight_drug = torch.tensor([pos_weight_value_drug], dtype=torch.float32).to(device)
        
        # Build the model using updated hyperparameters.
        model = GAN(
            input_size=model_cfg["input_size"],
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg.get("num_encoder_layers", 1),
            num_classifier_layers=model_cfg.get("num_classifier_layers", 1),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
            norm=model_cfg.get("norm", "batch"),
            classifier_hidden_dims=model_cfg.get("classifier_hidden_dims", []),
            activation=model_cfg.get("activation", "relu")
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
            data_test_loader, data_all_test_loader,   # Evaluate on test data for this fold.
            data_test_loader, data_all_test_loader,
            num_epochs,
            criterion_classifier, optimizer_classifier,
            criterion_disease_classifier, optimizer_disease_classifier,
            device
        )
        
        # Extract the final test accuracy from this fold.
        fold_test_acc = Results["test"]["accuracy"][-1]
        print(f"Fold {fold+1} test accuracy: {fold_test_acc:.4f}")
        fold_test_accuracies.append(fold_test_acc)
    
    # Return the average test accuracy over all folds.
    avg_test_accuracy = np.mean(fold_test_accuracies)
    print(f"Average test accuracy over 5 folds: {avg_test_accuracy:.4f}")
    return avg_test_accuracy

def objective(trial):
    """
    Objective function for Optuna: Suggest discrete hyperparameter values,
    update the base configuration, run a training trial using all folds,
    and return the average test accuracy for maximization.
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
    
    # Run the trial using a reduced number of epochs for speed.
    avg_test_accuracy = run_trial(trial_config, num_epochs=10)
    return avg_test_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Final test accuracy:", best_trial.value)
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
