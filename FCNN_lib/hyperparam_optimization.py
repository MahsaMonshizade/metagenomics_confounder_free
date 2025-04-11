import copy
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from config import config
from data_utils import get_data
from models import GAN  # This is your FCNN model (non-confounder-free version)
from utils import create_stratified_dataloader
from train import train_model
from sklearn.model_selection import StratifiedKFold

# Use the device specified in config.
device = torch.device(config["training"].get("device", "cpu"))

def run_trial(trial_config, num_epochs=10):
    """
    Run a single training trial on one fold (from a 5-fold split) using a reduced
    number of epochs, and return the final validation accuracy.
    """
    # Extract configuration sections.
    data_cfg = trial_config["data"]
    train_cfg = trial_config["training"]
    model_cfg = trial_config["model"]
    
    # Load merged training data.
    merged_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"])
    
    # Define feature columns (excluding metadata columns and 'SampleID').
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Overall disease classification.
    X = merged_data_all[feature_columns].values
    y_all = merged_data_all["disease"].values
    
    # Use the first fold from a 5-fold stratification.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X, y_all):
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]
        break  # Use only the first fold.
    
    # Prepare overall disease training data.
    x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
    y_all_train = torch.tensor(train_data["disease"].values, dtype=torch.float32).unsqueeze(1)
    x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
    y_all_val = torch.tensor(val_data["disease"].values, dtype=torch.float32).unsqueeze(1)
    
    batch_size = train_cfg["batch_size"]
    data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
    data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
    
    # Compute class weights.
    num_pos = y_all_train.sum().item()
    num_neg = len(y_all_train) - num_pos
    pos_weight_value = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    
    # Build the model using parameters from the configuration, including activation.
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
    
    # Define loss function for disease classification.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    # Define the optimizer for the disease classification branch.
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # For tuning, we pass the validation loader twice (using validation data for both "val" and "test" sets).
    Results = train_model(
        model, data_all_loader, data_all_val_loader, data_all_val_loader,
        num_epochs, criterion, optimizer, device
    )
    
    # Return the final validation accuracy.
    final_val_accuracy = Results["val"]["accuracy"][-1]
    return final_val_accuracy

def objective(trial):
    """
    Objective function for Optuna: Suggest discrete hyperparameter values,
    update the configuration, run one training trial, and return the final
    validation accuracy for maximization.
    """
    trial_config = copy.deepcopy(config)
    
    # Suggest categorical (discrete) hyperparameters.
    trial_config["model"]["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", config["tuning"]["num_encoder_layers"])
    trial_config["model"]["num_classifier_layers"] = trial.suggest_categorical("num_classifier_layers", config["tuning"]["num_classifier_layers"])
    trial_config["model"]["dropout_rate"] = trial.suggest_categorical("dropout_rate", config["tuning"]["dropout_rate"])
    trial_config["training"]["learning_rate"] = trial.suggest_categorical("learning_rate", config["tuning"]["learning_rate"])
    trial_config["model"]["activation"] = trial.suggest_categorical("activation", config["tuning"]["activation"])
    
    # Run the trial with a reduced number of epochs.
    val_accuracy = run_trial(trial_config, num_epochs=10)
    return val_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Final validation accuracy:", best_trial.value)
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
