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
from models import GAN, PearsonCorrelationLoss, MSEUniformLoss, KLDivergenceLoss
from utils import create_stratified_dataloader
from train import train_model
from config import config

# Select device from config.
device = torch.device(config["training"].get("device", "cpu"))

def plot_confusion_matrix(conf_matrix, title, save_path, class_names=None):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs("Results/FCNN_encoder_confounder_free_plots", exist_ok=True)

    # Extract parameters from config.
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Get the column names for disease and confounder.
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]

    # Load training and test data using the CLR transform.
    merged_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"])
    merged_test_data_all = get_data(data_cfg["test_abundance_path"], data_cfg["test_metadata_path"])

    # Define feature columns (exclude metadata columns and SampleID).
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]

    # Dynamically set input_size from the data.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")

    # Overall disease classification.
    X = merged_data_all[feature_columns].values
    merged_data_all['combined'] = (
        merged_data_all[disease_col].astype(str) +
        merged_data_all[confounder_col].astype(str)
    )
    y_all = merged_data_all["combined"].values

    # Prepare test data (for overall disease prediction).
    x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test_data_all[disease_col].values, dtype=torch.float32).unsqueeze(1)

    # In the confounder-free setup, we further use a “drug” (or confounder) classification branch.
    # Here, we select test patients with disease==1 and use their 'sex' as the confounder label.
    test_data_disease = merged_test_data_all[merged_test_data_all[disease_col] == 1]
    x_test_disease = torch.tensor(test_data_disease[feature_columns].values, dtype=torch.float32)
    y_test_disease = torch.tensor(test_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)

    # Set up 5-fold stratified cross-validation.
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Containers for metrics.
    train_metrics_per_fold = []
    val_metrics_per_fold = []
    test_metrics_per_fold = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Fold {fold+1}")
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]

        # Overall disease classification training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)

        # Confounder (drug) classification training data (only disease==1 samples, label is 'sex').
        train_data_disease = train_data[train_data[disease_col] == 1]
        val_data_disease = val_data[val_data[disease_col] == 1]
        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        x_disease_val = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_val = torch.tensor(val_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)

        # Create stratified DataLoaders.
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, train_cfg["batch_size"])
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, train_cfg["batch_size"])
        data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, train_cfg["batch_size"])
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, train_cfg["batch_size"])
        data_test_loader = create_stratified_dataloader(x_test_disease, y_test_disease, train_cfg["batch_size"])
        data_all_test_loader = create_stratified_dataloader(x_test_all, y_test_all, train_cfg["batch_size"])

        # Compute class weights.
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)
        num_pos_drug = y_disease_train.sum().item()
        num_neg_drug = len(y_disease_train) - num_pos_drug
        pos_weight_value_drug = num_neg_drug / num_pos_drug
        pos_weight_drug = torch.tensor([pos_weight_value_drug], dtype=torch.float32).to(device)

        # Build the model using hyperparameters from config.
        model = GAN(
            input_size=input_size,
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"]
        ).to(device)

        # Define loss functions.
        # For the distillation (Pearson correlation) phase.
        # criterion = PearsonCorrelationLoss().to(device)
        # criterion = KLDivergenceLoss().to(device)
        criterion = MSEUniformLoss().to(device)
        # For the confounder classifier branch.
        criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drug).to(device)
        # For the disease classification branch.
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
            data_loader, data_all_loader, data_val_loader, data_all_val_loader,
            data_test_loader, data_all_test_loader, train_cfg["num_epochs"],
            criterion_classifier, optimizer_classifier,
            criterion_disease_classifier, optimizer_disease_classifier, device
        )

        # Save model and features.
        torch.save(model.state_dict(), f"Results/FCNN_encoder_confounder_free_plots/trained_model_fold{fold+1}.pth")
        pd.Series(feature_columns).to_csv("Results/FCNN_encoder_confounder_free_plots/feature_columns.csv", index=False)
        print("Model and feature columns saved for fold", fold+1)

        train_metrics_per_fold.append(Results["train"])
        val_metrics_per_fold.append(Results["val"])
        test_metrics_per_fold.append(Results["test"])

        # Plot per-fold confusion matrices.
        plot_confusion_matrix(Results["train"]["confusion_matrix"][-1],
                              title=f"Train Confusion Matrix - Fold {fold+1}",
                              save_path=f"Results/FCNN_encoder_confounder_free_plots/fold_{fold+1}_train_conf_matrix.png",
                              class_names=["Class 0", "Class 1"])
        plot_confusion_matrix(Results["val"]["confusion_matrix"][-1],
                              title=f"Validation Confusion Matrix - Fold {fold+1}",
                              save_path=f"Results/FCNN_encoder_confounder_free_plots/fold_{fold+1}_val_conf_matrix.png",
                              class_names=["Class 0", "Class 1"])
        plot_confusion_matrix(Results["test"]["confusion_matrix"][-1],
                              title=f"Test Confusion Matrix - Fold {fold+1}",
                              save_path=f"Results/FCNN_encoder_confounder_free_plots/fold_{fold+1}_test_conf_matrix.png",
                              class_names=["Class 0", "Class 1"])

        num_epochs_actual = len(Results["train"]["gloss_history"])
        epochs = range(1, num_epochs_actual + 1)

        # Plot metric histories for this fold.
        plt.figure(figsize=(20, 15))
        plt.subplot(3, 3, 1)
        plt.plot(epochs, Results["train"]["gloss_history"], label=f'Fold {fold+1}')
        plt.title("Correlation g Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.plot(epochs, Results["train"]["dcor_history"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["dcor_history"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["dcor_history"], label=f'Test {fold+1}')
        plt.title("Distance Correlation History")
        plt.xlabel("Epoch")
        plt.ylabel("Distance Correlation")
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(epochs, Results["train"]["loss_history"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["loss_history"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["loss_history"], label=f'Test {fold+1}')
        plt.title("Disease Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(epochs, Results["train"]["accuracy"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["accuracy"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["accuracy"], label=f'Test {fold+1}')
        plt.title("Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(epochs, Results["train"]["f1_score"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["f1_score"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["f1_score"], label=f'Test {fold+1}')
        plt.title("F1 Score History")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(epochs, Results["train"]["auc_pr"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["auc_pr"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["auc_pr"], label=f'Test {fold+1}')
        plt.title("AUCPR History")
        plt.xlabel("Epoch")
        plt.ylabel("AUCPR")
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(epochs, Results["train"]["precision"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["precision"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["precision"], label=f'Test {fold+1}')
        plt.title("Precision History")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(epochs, Results["train"]["recall"], label=f'Train {fold+1}')
        plt.plot(epochs, Results["val"]["recall"], label=f'Val {fold+1}')
        plt.plot(epochs, Results["test"]["recall"], label=f'Test {fold+1}')
        plt.title("Recall History")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"Results/FCNN_encoder_confounder_free_plots/fold_{fold+1}_metrics.png")
        plt.close()

    # --------------- Aggregate Metrics Across Folds ---------------
    num_epochs_actual = len(train_metrics_per_fold[0]["gloss_history"])
    epochs = range(1, num_epochs_actual + 1)

    train_avg_metrics = {key: np.zeros(num_epochs_actual) for key in train_metrics_per_fold[0].keys() if key != "confusion_matrix"}
    val_avg_metrics = {key: np.zeros(num_epochs_actual) for key in val_metrics_per_fold[0].keys() if key != "confusion_matrix"}
    test_avg_metrics = {key: np.zeros(num_epochs_actual) for key in test_metrics_per_fold[0].keys() if key != "confusion_matrix"}

    train_conf_matrix_avg = [np.zeros_like(train_metrics_per_fold[0]["confusion_matrix"][0]) for _ in range(num_epochs_actual)]
    val_conf_matrix_avg = [np.zeros_like(val_metrics_per_fold[0]["confusion_matrix"][0]) for _ in range(num_epochs_actual)]
    test_conf_matrix_avg = [np.zeros_like(test_metrics_per_fold[0]["confusion_matrix"][0]) for _ in range(num_epochs_actual)]

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

    for key in train_avg_metrics.keys():
        train_avg_metrics[key] /= n_splits
    for key in val_avg_metrics.keys():
        val_avg_metrics[key] /= n_splits
    for key in test_avg_metrics.keys():
        test_avg_metrics[key] /= n_splits
    train_conf_matrix_avg = [cm / n_splits for cm in train_conf_matrix_avg]
    val_conf_matrix_avg = [cm / n_splits for cm in val_conf_matrix_avg]
    test_conf_matrix_avg = [cm / n_splits for cm in test_conf_matrix_avg]

    # Plot aggregated average metrics.
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_avg_metrics["loss_history"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["loss_history"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["loss_history"], label="Test Average")
    plt.title("Average Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_avg_metrics["accuracy"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["accuracy"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["accuracy"], label="Test Average")
    plt.title("Average Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(epochs, train_avg_metrics["f1_score"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["f1_score"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["f1_score"], label="Test Average")
    plt.title("Average F1 Score History")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_avg_metrics["auc_pr"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["auc_pr"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["auc_pr"], label="Test Average")
    plt.title("Average AUCPR History")
    plt.xlabel("Epoch")
    plt.ylabel("AUCPR")
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_avg_metrics["precision"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["precision"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["precision"], label="Test Average")
    plt.title("Average Precision History")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_avg_metrics["recall"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["recall"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["recall"], label="Test Average")
    plt.title("Average Recall History")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()

    # NEW: Plot aggregated Correlation g Loss History.
    plt.subplot(3, 3, 7)
    plt.plot(epochs, train_avg_metrics["gloss_history"], label="Train Average")
    # plt.plot(epochs, val_avg_metrics["gloss_history"], label="Val Average")
    # plt.plot(epochs, test_avg_metrics["gloss_history"], label="Test Average")
    plt.title("Average Correlation g Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # NEW: Plot aggregated Distance Correlation History.
    plt.subplot(3, 3, 8)
    plt.plot(epochs, train_avg_metrics["dcor_history"], label="Train Average")
    plt.plot(epochs, val_avg_metrics["dcor_history"], label="Val Average")
    plt.plot(epochs, test_avg_metrics["dcor_history"], label="Test Average")
    plt.title("Average Distance Correlation History")
    plt.xlabel("Epoch")
    plt.ylabel("Distance Correlation")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Results/FCNN_encoder_confounder_free_plots/average_metrics.png")
    plt.close()

    # Plot aggregated confusion matrices.
    plot_confusion_matrix(train_conf_matrix_avg[-1],
                          title="Average Train Confusion Matrix",
                          save_path="Results/FCNN_encoder_confounder_free_plots/average_train_conf_matrix.png",
                          class_names=["Class 0", "Class 1"])
    plot_confusion_matrix(val_conf_matrix_avg[-1],
                          title="Average Validation Confusion Matrix",
                          save_path="Results/FCNN_encoder_confounder_free_plots/average_val_conf_matrix.png",
                          class_names=["Class 0", "Class 1"])
    plot_confusion_matrix(test_conf_matrix_avg[-1],
                          title="Average Test Confusion Matrix",
                          save_path="Results/FCNN_encoder_confounder_free_plots/average_test_conf_matrix.png",
                          class_names=["Class 0", "Class 1"])

    avg_test_acc = test_avg_metrics["accuracy"][-1]
    print(f"Average Test Accuracy over {n_splits} folds: {avg_test_acc:.4f}")

    # Create a summary CSV.
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
    metrics_df.to_csv("Results/FCNN_encoder_confounder_free_plots/metrics_summary.csv", index=False)
    print("Metrics summary saved to 'Results/FCNN_encoder_confounder_free_plots/metrics_summary.csv'.")

if __name__ == "__main__":
    main()
