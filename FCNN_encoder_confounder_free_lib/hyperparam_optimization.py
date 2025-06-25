import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import optuna
import json

from sklearn.model_selection import StratifiedKFold
from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model
from config import config

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
device = torch.device(config["training"].get("device", "cpu"))

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def run_trial(trial_number=None):
    data_cfg = config["data"]
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]

    # Load and prepare data
    merged_data_all, merged_test_data_all = get_data(
        data_cfg["train_abundance_path"],
        data_cfg["train_metadata_path"],
        data_cfg["test_abundance_path"],
        data_cfg["test_metadata_path"]
    )

    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    input_size = len(feature_columns)

    X = merged_data_all[feature_columns].values
    y_all = merged_data_all[disease_col].values

    x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test_data_all[disease_col].values, dtype=torch.float32).unsqueeze(1)
    idx_test_all = merged_test_data_all["SampleID"].values

    test_data_disease = merged_test_data_all[merged_test_data_all[disease_col] == 1]
    x_test_disease = torch.tensor(test_data_disease[feature_columns].values, dtype=torch.float32)
    y_test_disease = torch.tensor(test_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"\n🔁 Trial {trial_number} — Fold {fold+1}/5")
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]

        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)

        train_data_disease = train_data[train_data[disease_col] == 1]
        val_data_disease = val_data[val_data[disease_col] == 1]
        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        x_disease_val = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_val = torch.tensor(val_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)

        batch_size = config["training"]["batch_size"]
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        data_test_loader = create_stratified_dataloader(x_test_disease, y_test_disease, batch_size)
        data_all_test_loader = create_stratified_dataloader(x_test_all, y_test_all, batch_size, sampleid=idx_test_all)

        # Class weights
        pos_weight_disease = torch.tensor(
            [(len(y_all_train) - y_all_train.sum().item()) / y_all_train.sum().item()],
            dtype=torch.float32
        ).to(device)

        pos_weight_drug = torch.tensor(
            [(len(y_disease_train) - y_disease_train.sum().item()) / y_disease_train.sum().item()],
            dtype=torch.float32
        ).to(device)

        # Build model
        model = GAN(
            input_size=input_size,
            latent_dim=config["model"]["latent_dim"],
            num_encoder_layers=config["model"]["num_encoder_layers"],
            num_classifier_layers=config["model"]["num_classifier_layers"],
            dropout_rate=config["model"]["dropout_rate"],
            norm=config["model"]["norm"],
            classifier_hidden_dims=config["model"]["classifier_hidden_dims"],
            activation=config["model"]["activation"],
            last_activation=config["model"]["last_activation"]
        ).to(device)

        # Loss and optimizers
        criterion = PearsonCorrelationLoss().to(device)
        criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drug).to(device)
        criterion_disease_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_disease).to(device)

        optimizer = optim.AdamW(model.encoder.parameters(), lr=config["training"]["encoder_lr"], weight_decay=config["training"]["weight_decay"])
        optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=config["training"]["classifier_lr"], weight_decay=config["training"]["weight_decay"])
        optimizer_disease_classifier = optim.AdamW(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
            lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
        )

        results = train_model(
            model, criterion, optimizer,
            data_loader, data_all_loader, data_val_loader, data_all_val_loader,
            data_test_loader, data_all_test_loader, config["training"]["num_epochs"],
            criterion_classifier, optimizer_classifier,
            criterion_disease_classifier, optimizer_disease_classifier, device
        )

        val_f1 = results["val"]["f1_score"][-1]
        val_scores.append(val_f1)

    return np.mean(val_scores)

def objective(trial):
    set_seed()

     # Set hyperparameters via categorical suggestions.
    config["model"]["num_encoder_layers"] = trial.suggest_categorical("num_encoder_layers", [1, 2, 3])
    config["model"]["num_classifier_layers"] = trial.suggest_categorical("num_classifier_layers", [0, 1, 2])
    # trial_config["model"]["dropout_rate"] = trial.suggest_categorical("dropout_rate", [0.0, 0.3, 0.5])
    config["training"]["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 2e-5, 2e-4, 2e-3, 5e-5, 5e-4, 5e-3])
    config["training"]["encoder_lr"] = trial.suggest_categorical("encoder_lr", [1e-5, 1e-4, 1e-3, 2e-5, 2e-4, 2e-3, 5e-5, 5e-4, 5e-3])
    config["training"]["classifier_lr"] = trial.suggest_categorical("classifier_lr", [1e-5, 1e-4, 1e-3, 2e-5, 2e-4, 2e-3, 5e-5, 5e-4, 5e-3])
    config["model"]["activation"] = trial.suggest_categorical("activation", config["tuning"]["activation"])
    config["model"]["latent_dim"] = trial.suggest_categorical("latent_dim", config["tuning"]["latent_dim"])
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", config["tuning"]["batch_size"])
    config["model"]["norm"] = trial.suggest_categorical("norm", config["tuning"]["norm"])
    config["model"]["last_activation"] = trial.suggest_categorical("last_activation", config["tuning"]["last_activation"])

    try:
        val_f1_avg = run_trial(trial.number)
        return val_f1_avg
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        return 0.0

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="confounder_free_search")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print("  Value:", study.best_value)
    print("  Params:", study.best_params)

    os.makedirs("Results/FCNN_encoder_confounder_free_plots", exist_ok=True)
    with open("Results/FCNN_encoder_confounder_free_plots/best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
   