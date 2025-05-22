import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_utils import get_data
from pretrain_models import preGAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from pretrain import train_model
from config import config
import random
import os

device = torch.device(config["pretrain_training"].get("device", "cpu"))

def convert_sex_to_binary(value):
    if value in ['1', 1, 'Female']:
        return 1
    elif value in ['0', 0, 'Male']:
        return 0
    else:
        raise ValueError(f"Unexpected value for sex: {value}")

def convert_disease_to_binary(value):
    if pd.isna(value):
        return np.nan

    if value in ['1', 1, 'D015179']:
        return 1
    elif value in ['0', 0, 'D006262']:
        return 0
    else:
        raise ValueError(f"Unexpected value for disease: {value}")

def main():


    os.makedirs("Results/FCNN_pretraining", exist_ok=True)

    # Extract configurations.
    model_cfg = config["model"]
    train_cfg = config["pretrain_training"]
    data_cfg = config["pretrain_data"]
    tmp_cfg = config["data"]

    
    # Get the column names for disease and confounder.
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]

    # Load training and test data using the CLR transform.
    merged_data_all, merged_test_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"], data_cfg["test_abundance_path"], data_cfg["test_metadata_path"])

    # TMP: Load the tmp data for preprocessing.
    merged_tmp_data_all = get_data(tmp_cfg["train_abundance_path"], tmp_cfg["train_metadata_path"], 
                                   tmp_cfg["test_abundance_path"], tmp_cfg["test_metadata_path"])[0]
    
    # TMP: Convert confounder column and disease to binary values [move to preprocessing script]. 
    merged_data_all[confounder_col] = merged_data_all[confounder_col].apply(convert_sex_to_binary)
    merged_test_data_all[confounder_col] = merged_test_data_all[confounder_col].apply(convert_sex_to_binary)
    merged_data_all[disease_col] = merged_data_all[disease_col].apply(convert_disease_to_binary)
    merged_test_data_all[disease_col] = merged_test_data_all[disease_col].apply(convert_disease_to_binary)

    # Define feature columns (exclude metadata and SampleID).
    metadata_columns = pd.read_csv(tmp_cfg["train_metadata_path"]).columns.tolist()
    
    # TMP: Preprocessing 'merged_data_all' 
    thr_feature_sum = data_cfg["threshold_feature_sum"]
    print(f'Sample number before filtering: {len(merged_data_all)}')
    print(f'Sample number in test data before filtering: {len(merged_test_data_all)}')
    feature_columns = [col for col in merged_tmp_data_all.columns if col not in metadata_columns and col != "SampleID"]
    row_sums = merged_data_all[feature_columns].sum(axis=1)
    merged_data_all = merged_data_all[row_sums >= thr_feature_sum]
    row_sums_test = merged_test_data_all[feature_columns].sum(axis=1)
    merged_test_data_all = merged_test_data_all[row_sums_test >= thr_feature_sum]
    print(f"Sample number after filtering: {len(merged_data_all)}")
    print(f"Sample number in test data after filtering: {len(merged_test_data_all)}")

    # Dynamically set input_size from the data.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")
    
    # For pre-training, we use all data for reconstruction (no labels needed)
    X = merged_data_all[feature_columns].values
    X_test = merged_test_data_all[feature_columns].values

    # Split data into training and validation sets (80/20 split)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # Convert data to tensors
    x_all_train = torch.tensor(X_train, dtype=torch.float32)
    x_all_val = torch.tensor(X_val, dtype=torch.float32)
    x_all_test = torch.tensor(X_test, dtype=torch.float32)
    
    # Create DataLoaders for reconstruction - using the data itself as both input and target
    batch_size = train_cfg["batch_size"]
    
    # Create data loaders for unlabeled data (reconstruction)
    data_all_loader = DataLoader(
        x_all_train,  # Just the data without labels
        batch_size=batch_size,
        shuffle=True
    )
    data_all_val_loader = DataLoader(
        x_all_val,
        batch_size=batch_size,
        shuffle=False
    )
    data_all_test_loader = DataLoader(
        x_all_test,
        batch_size=batch_size,
        shuffle=False
    )


    model = preGAN(
            input_size=input_size,  # Using the actual number of features
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"]
        ).to(device)

    # Dynamically set input_size from the data.

    criterion_reconstructor = nn.MSELoss().to(device)

    optimizer_reconstructor = optim.AdamW(
        list(model.encoder.parameters()) + list(model.reconstructor.parameters()),
        lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )

    Results = train_model(model, data_all_loader , data_all_val_loader, data_all_test_loader, train_cfg["num_epochs"], criterion_reconstructor, optimizer_reconstructor, device)

    # Save model and features.
    torch.save(model.state_dict(), f"Results/FCNN_pretraining/pretrained_model.pth")
    pd.Series(feature_columns).to_csv("Results/FCNN_pretraining/feature_columns.csv", index=False)

     # Extract the number of epochs actually trained.
    num_epochs_actual = len(Results["train"]["rec_loss_history"])
    epochs = range(1, num_epochs_actual + 1)

    # Plot metric histories.
    plt.figure(figsize=(14, 10))

    plt.subplot(1, 1, 1)
    plt.plot(epochs, Results["train"]["rec_loss_history"], label='Train')
    plt.plot(epochs, Results["val"]["rec_loss_history"], label='Val')
    plt.plot(epochs, Results["test"]["rec_loss_history"], label='Test')
    plt.title("Reconstruction Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Results/FCNN_pretraining/pretraining_metrics.png")
    plt.close()

     # Create a summary CSV with reconstruction metrics.
    metrics_columns = ["Phase", "Final_RecLoss"]
    metrics_data = [
        ["Train", Results["train"]["rec_loss_history"][-1]],
        ["Validation", Results["val"]["rec_loss_history"][-1]],
        ["Test", Results["test"]["rec_loss_history"][-1]]
    ]
    
    metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)
    metrics_df.to_csv("Results/FCNN_pretraining/pretraining_metrics_summary.csv", index=False)
    print("Metrics summary saved to 'Results/FCNN_pretraining/pretraining_metrics_summary.csv'.")


if __name__ == "__main__":
    main()
