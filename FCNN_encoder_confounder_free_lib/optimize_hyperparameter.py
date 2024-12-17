import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, auc
import numpy as np

from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model
import pandas as pd

# If GPU available, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed hyperparameters based on your existing code
input_size = 371
latent_dim = 32
num_layers = 1
batch_size = 64

# File paths (adjust if needed)
train_abundance_path = 'MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv'
train_metadata_path = 'MetaCardis_data/train_T2D_metadata.csv'
test_abundance_path = 'MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv'
test_metadata_path = 'MetaCardis_data/test_T2D_metadata.csv'

# Load data once outside objective for efficiency
merged_data_all = get_data(train_abundance_path, train_metadata_path)
merged_test_data_all = get_data(test_abundance_path, test_metadata_path)

metadata = list(pd.read_csv(train_metadata_path).columns)
feature_columns = [col for col in merged_data_all.columns if col not in metadata and col != 'SampleID']

X = merged_data_all[feature_columns].values
y_all = merged_data_all['PATGROUPFINAL_C'].values

x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
y_test_all = torch.tensor(merged_test_data_all['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

# We'll do a simple train/val split for hyperparam evaluation
train_data, val_data = train_test_split(merged_data_all, test_size=0.2, stratify=merged_data_all['PATGROUPFINAL_C'], random_state=42)

x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
y_all_train = torch.tensor(train_data['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
y_all_val = torch.tensor(val_data['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

train_data_disease = train_data[train_data['PATGROUPFINAL_C'] == 1]
val_data_disease = val_data[val_data['PATGROUPFINAL_C'] == 1]

x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
y_disease_train = torch.tensor(train_data_disease['METFORMIN_C'].values, dtype=torch.float32).unsqueeze(1)

x_disease_val = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
y_disease_val = torch.tensor(val_data_disease['METFORMIN_C'].values, dtype=torch.float32).unsqueeze(1)

data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, batch_size)
data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)

# Disease group data in test set
test_data_disease = merged_test_data_all[merged_test_data_all['PATGROUPFINAL_C'] == 1]
x_test_disease = torch.tensor(test_data_disease[feature_columns].values, dtype=torch.float32)
y_test_disease = torch.tensor(test_data_disease['METFORMIN_C'].values, dtype=torch.float32).unsqueeze(1)

data_test_loader = create_stratified_dataloader(x_test_disease, y_test_disease, batch_size)
data_all_test_loader = create_stratified_dataloader(x_test_all, y_test_all, batch_size)


def objective(trial):
    # Hyperparameter search space
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    encoder_lr = trial.suggest_float("encoder_lr", 1e-5, 1e-2, log=True)
    classifier_lr = trial.suggest_float("classifier_lr", 1e-5, 1e-2, log=True)
    disease_classifier_lr = trial.suggest_float("disease_classifier_lr", 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 50, 200, step=25)
    step_lr_gamma = trial.suggest_float("scheduler_gamma", 0.5, 0.99)
    step_lr_step_size = trial.suggest_int("scheduler_step_size", 10, 50, step=10)
    gradient_clip_val = trial.suggest_float("gradient_clip", 0.0, 1.0)
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 5, 20)

    # Compute positive class weights
    num_pos_disease = y_all_train.sum().item()
    num_neg_disease = len(y_all_train) - num_pos_disease
    pos_weight_value_disease = num_neg_disease / num_pos_disease
    pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)

    num_pos_drug = y_disease_train.sum().item()
    num_neg_drug = len(y_disease_train) - num_pos_drug
    pos_weight_value_drug = num_neg_drug / num_pos_drug
    pos_weight_drug = torch.tensor([pos_weight_value_drug], dtype=torch.float32).to(device)

    model = GAN(input_size, latent_dim=latent_dim, num_layers=num_layers).to(device)
    criterion = PearsonCorrelationLoss().to(device)
    criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drug).to(device)
    criterion_disease_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_disease).to(device)

    # Select optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.encoder.parameters(), lr=encoder_lr)
        optimizer_classifier = optim.Adam(model.classifier.parameters(), lr=classifier_lr)
        optimizer_disease_classifier = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=disease_classifier_lr
        )
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.encoder.parameters(), lr=encoder_lr, momentum=0.9)
        optimizer_classifier = optim.SGD(model.classifier.parameters(), lr=classifier_lr, momentum=0.9)
        optimizer_disease_classifier = optim.SGD(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), 
            lr=disease_classifier_lr, momentum=0.9
        )
    else:  # AdamW
        optimizer = optim.AdamW(model.encoder.parameters(), lr=encoder_lr)
        optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=classifier_lr)
        optimizer_disease_classifier = optim.AdamW(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=disease_classifier_lr
        )

    # Introduce scheduler on disease_classifier optimizer (as example)
    scheduler = StepLR(optimizer_disease_classifier, step_size=step_lr_step_size, gamma=step_lr_gamma)

    # We'll modify the train loop slightly to include gradient clipping and early stopping here
    best_val_f1 = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        # Training loop
        data_all_iter = iter(data_all_loader)
        data_iter = iter(data_loader)

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

                # Train classifier (drug)
                for param in model.encoder.parameters():
                    param.requires_grad = False
                encoded_features = model.encoder(x_batch)
                predicted_drug = model.classifier(encoded_features)
                r_loss = criterion_classifier(predicted_drug, y_batch)
                optimizer_classifier.zero_grad()
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), gradient_clip_val)
                optimizer_classifier.step()
                for param in model.encoder.parameters():
                    param.requires_grad = True

                # Train encoder (g_loss)
                for param in model.classifier.parameters():
                    param.requires_grad = False
                encoded_features = model.encoder(x_batch)
                predicted_drug = torch.sigmoid(model.classifier(encoded_features))
                g_loss = criterion(predicted_drug, y_batch)
                optimizer.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), gradient_clip_val)
                optimizer.step()
                for param in model.classifier.parameters():
                    param.requires_grad = True

                # Train disease classifier
                encoded_features_all = model.encoder(x_all_batch)
                predicted_disease_all = model.disease_classifier(encoded_features_all)
                c_loss = criterion_disease_classifier(predicted_disease_all, y_all_batch)
                optimizer_disease_classifier.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
                    gradient_clip_val
                )
                optimizer_disease_classifier.step()

            except StopIteration:
                break

        # Step scheduler
        scheduler.step()

        # Validation
        model.eval()
        epoch_val_preds = []
        epoch_val_labels = []
        epoch_val_probs = []
        with torch.no_grad():
            for x_val_batch, y_val_batch in data_all_val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                encoded_val = model.encoder(x_val_batch)
                predicted_val = model.disease_classifier(encoded_val)
                prob_val = torch.sigmoid(predicted_val).detach().cpu()
                epoch_val_probs.append(prob_val)
                epoch_val_preds.append((prob_val > 0.5).float())
                epoch_val_labels.append(y_val_batch.detach().cpu())

        epoch_val_probs = torch.cat(epoch_val_probs)
        epoch_val_preds = torch.cat(epoch_val_preds)
        epoch_val_labels = torch.cat(epoch_val_labels)

        val_f1 = f1_score(epoch_val_labels, epoch_val_preds)
        # Use F1 for optimization (could also use balanced accuracy)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Return the negative of val_f1 if you want to minimize, or just return -best_val_f1
    # Optuna by default tries to minimize, so we can return -best_val_f1 to maximize F1.
    return -best_val_f1


if __name__ == "__main__":

    # Make sure directories exist
    os.makedirs('Results', exist_ok=True)

    # We optimize to minimize the negative F1-score
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value: {-trial.value:.4f} (This is the best F1-score)")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can also save study results to a database or a pickle file if needed.

