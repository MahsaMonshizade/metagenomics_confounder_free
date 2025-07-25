import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay

from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model
from config import config
import random
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# Select device.
device = torch.device(config["training"].get("device", "cpu"))

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

    os.makedirs("Results/MicroKPNN_finetune_plots", exist_ok=True)

    # Extract configurations.
    model_cfg = config["model"]
    train_cfg = config["finetuning_training"]
    data_cfg = config["data"]
    # TMP: For loading the pre-trained model [fix it to somewhere as other paths].
    pretrained_model_path = "Results/MicroKPNN_pretraining/pretrained_model.pth"

    # Get the column names for disease and confounder.
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]

     # Load training and test data using the CLR transform.
    merged_data_all, merged_test_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"], data_cfg["test_abundance_path"], data_cfg["test_metadata_path"])

    # Define feature columns (exclude metadata and SampleID).
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Dynamically set input_size from the data.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")

    # Save feature columns (only once).
    pd.Series(feature_columns).to_csv("Results/MicroKPNN_finetune_plots/feature_columns.csv", index=False)

    # Input features.
    X = merged_data_all[feature_columns].values

    # Use 'disease' as the stratification target for cross-validation.
    y_all = merged_data_all[disease_col].values

    # Prepare test data.
    x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test_data_all[disease_col].values, dtype=torch.float32).unsqueeze(1)

    # Stratified k-fold cross-validation.
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_metrics_per_fold = []
    val_metrics_per_fold = []
    test_metrics_per_fold = []

    # Optionally, recalculate input size from feature columns.
    input_size = len(feature_columns)

    ###############added
    edge_list = f"Results/MicroKPNN_finetune_plots/required_data/EdgeList.csv"

    # Build masks
    mask, parent_dict = build_mask(edge_list, feature_columns)
    print(mask.shape)
    print(mask)
    parent_df = pd.DataFrame(list(parent_dict.items()), columns=['Parent', 'Index'])
    parent_dict_csv_path = "Results/MicroKPNN_finetune_plots/required_data/parent_dict_main.csv"
    parent_df.to_csv(parent_dict_csv_path, index=False)
    ########################

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Fold {fold+1}")
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]

        x_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)

        x_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_val = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)

        # Create stratified DataLoaders.
        train_loader = create_stratified_dataloader(x_train, y_train, train_cfg["batch_size"])
        val_loader = create_stratified_dataloader(x_val, y_val, train_cfg["batch_size"])
        test_loader = create_stratified_dataloader(x_test_all, y_test_all, train_cfg["batch_size"])

        # Compute positive class weight for BCE loss.
        num_pos = y_train.sum().item()
        num_neg = len(y_train) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

        # Build the model.
        model = GAN(
            mask = mask, 
            input_size=input_size,  # Using the actual number of features
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"]
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )

        # Train the model.
        Results = train_model(model, train_loader, val_loader, test_loader, 
                                train_cfg["num_epochs"], criterion, optimizer, device, pretrained_model_path)

        # Save model.
        torch.save(model.state_dict(), f"Results/MicroKPNN_finetune_plots/trained_model_fold{fold+1}.pth")
        print("Model and feature columns saved for fold", fold+1)

        train_metrics_per_fold.append(Results["train"])
        val_metrics_per_fold.append(Results["val"])
        test_metrics_per_fold.append(Results["test"])

        # Plot confusion matrices for final epoch of each fold.
        plot_confusion_matrix(Results["train"]["confusion_matrix"][-1],
                              title=f"Train Confusion Matrix - Fold {fold+1}",
                              save_path=f"Results/MicroKPNN_finetune_plots/fold_{fold+1}_train_conf_matrix.png",
                              class_names=["Class 0", "Class 1"])
        plot_confusion_matrix(Results["val"]["confusion_matrix"][-1],
                              title=f"Validation Confusion Matrix - Fold {fold+1}",
                              save_path=f"Results/MicroKPNN_finetune_plots/fold_{fold+1}_val_conf_matrix.png",
                              class_names=["Class 0", "Class 1"])
        plot_confusion_matrix(Results["test"]["confusion_matrix"][-1],
                              title=f"Test Confusion Matrix - Fold {fold+1}",
                              save_path=f"Results/MicroKPNN_finetune_plots/fold_{fold+1}_test_conf_matrix.png",
                              class_names=["Class 0", "Class 1"])
           # Plot average metrics across folds (2x3 grid).
        num_epochs_actual = len(Results["train"]["loss_history"])
        epochs = range(1, num_epochs_actual + 1)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.plot(epochs, Results["train"]["loss_history"], label=f"Train {fold+1}")
        plt.plot(epochs, Results["val"]["loss_history"], label=f"Validation {fold+1}")
        plt.plot(epochs, Results["test"]["loss_history"], label=f"Test {fold+1}")
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 3, 2)
        plt.plot(epochs, Results["train"]["accuracy"], label=f"Train {fold+1}")
        plt.plot(epochs, Results["val"]["accuracy"], label=f"Validation {fold+1}")
        plt.plot(epochs, Results["test"]["accuracy"], label=f"Test {fold+1}")
        plt.title("Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Accuracy")
        plt.legend()

        plt.subplot(2, 3, 3)
        plt.plot(epochs, Results["train"]["f1_score"], label=f"Train {fold+1}")
        plt.plot(epochs, Results["val"]["f1_score"], label=f"Validation {fold+1}")
        plt.plot(epochs, Results["test"]["f1_score"], label=f"Test {fold+1}")
        plt.title("F1 Score History")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(epochs, Results["train"]["auc_pr"], label=f"Train {fold+1}")
        plt.plot(epochs, Results["val"]["auc_pr"], label=f"Validation {fold+1}")
        plt.plot(epochs, Results["test"]["auc_pr"], label=f"Test {fold+1}")
        plt.title("AUCPR History")
        plt.xlabel("Epoch")
        plt.ylabel("AUCPR")
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(epochs, Results["train"]["precision"], label=f"Train {fold+1}")
        plt.plot(epochs, Results["val"]["precision"], label=f"Validation {fold+1}")
        plt.plot(epochs, Results["test"]["precision"], label=f"Test {fold+1}")
        plt.title("Precision History")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(epochs, Results["train"]["recall"], label=f"Train {fold+1}")
        plt.plot(epochs, Results["val"]["recall"], label=f"Validation {fold+1}")
        plt.plot(epochs, Results["test"]["recall"], label=f"Test {fold+1}")
        plt.title("Recall History")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"Results/MicroKPNN_finetune_plots/fold_{fold+1}_metrics.png")
        plt.close()

    # -----------------------------------------------------------
    # After cross-validation: Aggregate metrics and plot averages.
    num_epochs_actual = len(train_metrics_per_fold[0]["loss_history"])
    epochs = range(1, num_epochs_actual + 1)

    # Initialize dictionaries for averaging scalar metrics.
    train_avg_metrics = {key: np.zeros(num_epochs_actual) for key in train_metrics_per_fold[0].keys() if key != "confusion_matrix"}
    val_avg_metrics = {key: np.zeros(num_epochs_actual) for key in val_metrics_per_fold[0].keys() if key != "confusion_matrix"}
    test_avg_metrics = {key: np.zeros(num_epochs_actual) for key in test_metrics_per_fold[0].keys() if key != "confusion_matrix"}

    # Initialize lists for confusion matrices.
    train_conf_matrix_avg = [np.zeros_like(train_metrics_per_fold[0]["confusion_matrix"][0]) for _ in range(num_epochs_actual)]
    val_conf_matrix_avg = [np.zeros_like(val_metrics_per_fold[0]["confusion_matrix"][0]) for _ in range(num_epochs_actual)]
    test_conf_matrix_avg = [np.zeros_like(test_metrics_per_fold[0]["confusion_matrix"][0]) for _ in range(num_epochs_actual)]

    # Accumulate metrics.
    for i in range(n_splits):
        for key in train_avg_metrics.keys():
            train_avg_metrics[key] += np.array(train_metrics_per_fold[i][key])
        for epoch_idx, cm in enumerate(train_metrics_per_fold[i]["confusion_matrix"]):
            train_conf_matrix_avg[epoch_idx] += cm

        for key in val_avg_metrics.keys():
            val_avg_metrics[key] += np.array(val_metrics_per_fold[i][key])
        for epoch_idx, cm in enumerate(val_metrics_per_fold[i]["confusion_matrix"]):
            val_conf_matrix_avg[epoch_idx] += cm

        for key in test_avg_metrics.keys():
            test_avg_metrics[key] += np.array(test_metrics_per_fold[i][key])
        for epoch_idx, cm in enumerate(test_metrics_per_fold[i]["confusion_matrix"]):
            test_conf_matrix_avg[epoch_idx] += cm

    # Compute averages across folds.
    for key in train_avg_metrics.keys():
        train_avg_metrics[key] /= n_splits
    for key in val_avg_metrics.keys():
        val_avg_metrics[key] /= n_splits
    for key in test_avg_metrics.keys():
        test_avg_metrics[key] /= n_splits
    train_conf_matrix_avg = [cm / n_splits for cm in train_conf_matrix_avg]
    val_conf_matrix_avg = [cm / n_splits for cm in val_conf_matrix_avg]
    test_conf_matrix_avg = [cm / n_splits for cm in test_conf_matrix_avg]

    # Plot average metrics across folds (2x3 grid).
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_avg_metrics["loss_history"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["loss_history"], label="Validation Average")
    plt.plot(epochs, test_avg_metrics["loss_history"], label="Test Average")
    plt.title("Average Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_avg_metrics["accuracy"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["accuracy"], label="Validation Average")
    plt.plot(epochs, test_avg_metrics["accuracy"], label="Test Average")
    plt.title("Average Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_avg_metrics["f1_score"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["f1_score"], label="Validation Average")
    plt.plot(epochs, test_avg_metrics["f1_score"], label="Test Average")
    plt.title("Average F1 Score History")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_avg_metrics["auc_pr"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["auc_pr"], label="Validation Average")
    plt.plot(epochs, test_avg_metrics["auc_pr"], label="Test Average")
    plt.title("Average AUCPR History")
    plt.xlabel("Epoch")
    plt.ylabel("AUCPR")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_avg_metrics["precision"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["precision"], label="Validation Average")
    plt.plot(epochs, test_avg_metrics["precision"], label="Test Average")
    plt.title("Average Precision History")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_avg_metrics["recall"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["recall"], label="Validation Average")
    plt.plot(epochs, test_avg_metrics["recall"], label="Test Average")
    plt.title("Average Recall History")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Results/MicroKPNN_finetune_plots/average_metrics.png")
    plt.close()

    # Plot averaged confusion matrices.
    plot_confusion_matrix(train_conf_matrix_avg[-1],
                          title="Average Train Confusion Matrix",
                          save_path="Results/MicroKPNN_finetune_plots/average_train_conf_matrix.png",
                          class_names=["Class 0", "Class 1"])
    plot_confusion_matrix(val_conf_matrix_avg[-1],
                          title="Average Validation Confusion Matrix",
                          save_path="Results/MicroKPNN_finetune_plots/average_val_conf_matrix.png",
                          class_names=["Class 0", "Class 1"])
    plot_confusion_matrix(test_conf_matrix_avg[-1],
                          title="Average Test Confusion Matrix",
                          save_path="Results/MicroKPNN_finetune_plots/average_test_conf_matrix.png",
                          class_names=["Class 0", "Class 1"])

    # Create a summary DataFrame with metrics from each fold and the average.
    metrics_columns = ["Fold", "Train_Accuracy", "Val_Accuracy", "Test_Accuracy",
                       "Train_F1", "Val_F1", "Test_F1", "Train_AUCPR", "Val_AUCPR", "Test_AUCPR",
                       "Train_Precision", "Val_Precision", "Test_Precision",
                       "Train_Recall", "Val_Recall", "Test_Recall"]
    metrics_data = []
    for i in range(n_splits):
        fold_data = [
            i+1,
            train_metrics_per_fold[i]["accuracy"][-1],
            val_metrics_per_fold[i]["accuracy"][-1],
            test_metrics_per_fold[i]["accuracy"][-1],
            train_metrics_per_fold[i]["f1_score"][-1],
            val_metrics_per_fold[i]["f1_score"][-1],
            test_metrics_per_fold[i]["f1_score"][-1],
            train_metrics_per_fold[i]["auc_pr"][-1],
            val_metrics_per_fold[i]["auc_pr"][-1],
            test_metrics_per_fold[i]["auc_pr"][-1],
            train_metrics_per_fold[i]["precision"][-1],
            val_metrics_per_fold[i]["precision"][-1],
            test_metrics_per_fold[i]["precision"][-1],
            train_metrics_per_fold[i]["recall"][-1],
            val_metrics_per_fold[i]["recall"][-1],
            test_metrics_per_fold[i]["recall"][-1]
        ]
        metrics_data.append(fold_data)
    # Append the average across folds.
    avg_data = [
        "Average",
        train_avg_metrics["accuracy"][-1],
        val_avg_metrics["accuracy"][-1],
        test_avg_metrics["accuracy"][-1],
        train_avg_metrics["f1_score"][-1],
        val_avg_metrics["f1_score"][-1],
        test_avg_metrics["f1_score"][-1],
        train_avg_metrics["auc_pr"][-1],
        val_avg_metrics["auc_pr"][-1],
        test_avg_metrics["auc_pr"][-1],
        train_avg_metrics["precision"][-1],
        val_avg_metrics["precision"][-1],
        test_avg_metrics["precision"][-1],
        train_avg_metrics["recall"][-1],
        val_avg_metrics["recall"][-1],
        test_avg_metrics["recall"][-1]
    ]
    metrics_data.append(avg_data)
    metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)
    metrics_df.to_csv("Results/MicroKPNN_finetune_plots/metrics_summary.csv", index=False)
    print("Average metrics, plots, and metrics summary saved.")

if __name__ == "__main__":
    main()
