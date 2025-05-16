import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_utils import get_data
from models import PearsonCorrelationLoss
from pretrain_models import preGAN
from utils import create_stratified_dataloader
from pretrain import train_model
from config import config
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# Select device from config.


def plot_confusion_matrix(conf_matrix, title, save_path, class_names=None):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def build_mask(edge_list, species):
    # generate the mask
        edge_df = pd.read_csv(edge_list)
        
        edge_df['parent'] = edge_df['parent'].astype(str)
        parent_nodes = sorted(set(edge_df['parent'].tolist()))  # Sort to ensure consistent order
        mask = torch.zeros(len(species), len(parent_nodes))
        child_nodes = species

        parent_dict = {k: i for i, k in enumerate(parent_nodes)}
        child_dict = {k: i for i, k in enumerate(child_nodes)}
        
        for i, row in edge_df.iterrows():
            if row['child'] != 'Unnamed: 0': 
                mask[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1

        return mask.T, parent_dict
    
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

    os.environ["PYTHONHASHSEED"] = str(42)

    # 2) Seed Python built-ins and numpy
    random.seed(42)
    np.random.seed(42)

    # 3) Seed PyTorch (both CPU and all GPUs)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 4) Force cuDNN deterministic, disable benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # 5) (Optional) Enforce PyTorch to error on nondeterministic ops
    torch.use_deterministic_algorithms(True)

    os.makedirs("Results/MicroKPNN_encoder_confounder_free_pretraining", exist_ok=True)

    # Extract parameters from config.
    model_cfg = config["model"]
    train_cfg = config["pretrain_training"]
    data_cfg = config["pretrain_data"]
    tmp_cfg = config["data"] # TMP: Used for preprocessing pre-training data [move to preprocessing script].

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(train_cfg.get("device", default_device))

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

    # Define feature columns (exclude metadata columns and SampleID).
    # metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    # feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
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

    # Prepare test data (for overall disease prediction).
    x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test_data_all[disease_col].values, dtype=torch.float32).unsqueeze(1)

    # For the confounder classification branch, select samples with the confounder
    confounder_data = merged_data_all[merged_data_all[disease_col] == 1]
    x_confounder = torch.tensor(confounder_data[feature_columns].values, dtype=torch.float32)
    y_confounder = torch.tensor(confounder_data[confounder_col].values, dtype=torch.float32).unsqueeze(1)
    
    # Split confounder data into train/val
    x_confounder_train, x_confounder_val, y_confounder_train, y_confounder_val = train_test_split(
        x_confounder, y_confounder, test_size=0.2, random_state=42
    )
    
    # Same for test data
    confounder_test_data = merged_test_data_all[~merged_test_data_all[confounder_col].isna()]
    x_confounder_test = torch.tensor(confounder_test_data[feature_columns].values, dtype=torch.float32)
    y_confounder_test = torch.tensor(confounder_test_data[confounder_col].values, dtype=torch.float32).unsqueeze(1)

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
    
    # For confounder classification, we use stratified dataloaders
    data_loader = create_stratified_dataloader(x_confounder_train, y_confounder_train, batch_size)
    data_val_loader = create_stratified_dataloader(x_confounder_val, y_confounder_val, batch_size)
    data_test_loader = create_stratified_dataloader(x_confounder_test, y_confounder_test, batch_size)

    # Compute class weights for confounder classifier.
    num_pos_confounder = y_confounder_train.sum().item()
    num_neg_confounder = len(y_confounder_train) - num_pos_confounder
    pos_weight_value = num_neg_confounder / num_pos_confounder if num_pos_confounder > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    ########################
    # Build the mask for the model.
    edge_list = f"Results/MicroKPNN_encoder_confounder_free_plots/required_data/EdgeList.csv"
    mask, parent_dict = build_mask(edge_list, feature_columns)
    print(mask.shape)
    print(mask)
    ########################

    # Build the model using hyperparameters from config.
    model = preGAN(
            mask=mask, 
            input_size=input_size,
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"],
            activation=model_cfg["activation"], 
            last_activation = model_cfg["last_activation"]
    ).to(device)

    # Define loss functions.
    # For the distillation (Pearson correlation) phase.
    criterion = PearsonCorrelationLoss().to(device)
    # For the confounder classifier branch.
    criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    # For the reconstruction branch.
    criterion_reconstructor = nn.MSELoss().to(device)

    # Define optimizers.
    optimizer = optim.AdamW(model.encoder.parameters(), lr=train_cfg["encoder_lr"], weight_decay=train_cfg["weight_decay"])
    optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=train_cfg["classifier_lr"], weight_decay=train_cfg["weight_decay"])
    optimizer_reconstructor = optim.AdamW(
        list(model.encoder.parameters()) + list(model.reconstructor.parameters()),
        lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )

    # Train the model using the modified three-phase training routine.
    Results = train_model(
        model, criterion, optimizer,
        data_loader, data_all_loader, data_val_loader, data_all_val_loader,
        data_test_loader, data_all_test_loader, train_cfg["num_epochs"],
        criterion_classifier, optimizer_classifier,
        criterion_reconstructor, optimizer_reconstructor, device
    )

    # Save model and features.
    torch.save(model.state_dict(), f"Results/MicroKPNN_encoder_confounder_free_pretraining/pretrained_model.pth")
    pd.Series(feature_columns).to_csv("Results/MicroKPNN_encoder_confounder_free_pretraining/feature_columns.csv", index=False)
    print("Model and feature columns saved")

    # Extract the number of epochs actually trained.
    num_epochs_actual = len(Results["train"]["gloss_history"])
    epochs = range(1, num_epochs_actual + 1)

    # Plot metric histories.
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, Results["train"]["gloss_history"], label='Train')
    plt.title("Correlation g Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("G Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, Results["train"]["dcor_history"], label='Train')
    plt.plot(epochs, Results["val"]["dcor_history"], label='Val')
    plt.plot(epochs, Results["test"]["dcor_history"], label='Test')
    plt.title("Distance Correlation History")
    plt.xlabel("Epoch")
    plt.ylabel("Distance Correlation")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, Results["train"]["rloss_history"], label='Train')
    plt.title("Confounder Prediction Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("R Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, Results["train"]["rec_loss_history"], label='Train')
    plt.plot(epochs, Results["val"]["rec_loss_history"], label='Val')
    plt.plot(epochs, Results["test"]["rec_loss_history"], label='Test')
    plt.title("Reconstruction Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Results/MicroKPNN_encoder_confounder_free_pretraining/pretraining_metrics.png")
    plt.close()

    # Create a summary CSV with reconstruction metrics.
    metrics_columns = ["Phase", "Final_RecLoss", "Final_DCor"]
    metrics_data = [
        ["Train", Results["train"]["rec_loss_history"][-1], Results["train"]["dcor_history"][-1]],
        ["Validation", Results["val"]["rec_loss_history"][-1], Results["val"]["dcor_history"][-1]],
        ["Test", Results["test"]["rec_loss_history"][-1], Results["test"]["dcor_history"][-1]]
    ]
    
    metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)
    metrics_df.to_csv("Results/MicroKPNN_encoder_confounder_free_pretraining/pretraining_metrics_summary.csv", index=False)
    print("Metrics summary saved to 'Results/MicroKPNN_encoder_confounder_free_pretraining/pretraining_metrics_summary.csv'.")

if __name__ == "__main__":
    main()
