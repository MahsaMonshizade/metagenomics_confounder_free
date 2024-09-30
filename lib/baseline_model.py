# baseline_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init
import numpy as np
from data_processing import create_batch, dcor_calculation_data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import json
import dcor

def previous_power_of_two(n):
    """Return the greatest power of two less than or equal to n."""
    return 2 ** (n.bit_length() - 1)

class BaselineModel(nn.Module):
    """
    Baseline model with an encoder and a disease classifier.
    """
    def __init__(self, input_dim, latent_dim=64, activation_fn=nn.ReLU, num_layers=3):
        super(BaselineModel, self).__init__()
        # Store parameters for re-initialization
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.num_layers = num_layers

        self.encoder = self._build_encoder(input_dim, latent_dim, num_layers, activation_fn)
        self.disease_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)
        self.disease_classifier_loss = nn.BCEWithLogitsLoss()

        self.initialize_weights()

    def _build_encoder(self, input_dim, latent_dim, num_layers, activation_fn):
        """Build the encoder network."""
        layers = []
        first_layer = previous_power_of_two(input_dim)
        layers.extend([
            nn.Linear(input_dim, first_layer),
            nn.BatchNorm1d(first_layer),
            activation_fn()
        ])
        current_dim = first_layer
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, max(current_dim // 2, latent_dim)),
                nn.BatchNorm1d(max(current_dim // 2, latent_dim)),
                activation_fn()
            ])
            current_dim = max(current_dim // 2, latent_dim)
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, activation_fn, num_layers):
        """Build the disease classifier."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, max(current_dim // 2, 1)),
                nn.BatchNorm1d(max(current_dim // 2, 1)),
                activation_fn()
            ])
            current_dim = max(current_dim // 2, 1)
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        """Initialize weights using Kaiming initialization for layers with ReLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the encoder and classifier."""
        encoded = self.encoder(x)
        output = self.disease_classifier(encoded).view(-1)
        return output

def train_baseline_model(model, epochs, relative_abundance, metadata, batch_size=64, lr=0.001):
    """Train the baseline model using K-Fold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_accuracies = []
    all_eval_aucs = []
    all_eval_f1s = []

    for fold, (train_index, val_index) in enumerate(skf.split(relative_abundance, metadata['disease'])):
        print(f"\nFold {fold + 1}/5")

        # Re-initialize the model and optimizer
        model = BaselineModel(input_dim=model.input_dim, latent_dim=model.latent_dim,
                              activation_fn=model.activation_fn, num_layers=model.num_layers)
        model.initialize_weights()
        model.to(device)

        # Initialize optimizer and scheduler
        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=lr
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        X_clr_df_train = relative_abundance.iloc[train_index].reset_index(drop=True)
        X_clr_df_val = relative_abundance.iloc[val_index].reset_index(drop=True)
        train_metadata = metadata.iloc[train_index].reset_index(drop=True)
        val_metadata = metadata.iloc[val_index].reset_index(drop=True)

        # best_loss = float('inf')
        best_disease_acc = 0
        early_stop_step = 20
        early_stop_patience = 0

        # Prepare data for dcor and MI calculations
        (training_feature_ctrl, metadata_ctrl_age, metadata_ctrl_bmi,
         training_feature_disease, metadata_disease_age, metadata_disease_bmi) = dcor_calculation_data(
            X_clr_df_train, train_metadata, device
        )

        # Lists to store metrics per epoch
        c_losses = []
        train_accuracies = []
        val_accuracies = []
        train_aucs = []
        val_aucs = []
        train_losses = []
        val_losses = []
        train_f1s = []
        val_f1s = []
        dcs0, dcs1, mis0, mis1 = [], [], [], []

        for epoch in range(epochs):
            # Training Phase
            model.train()

            # Create batch
            _, _, _, training_feature_batch, metadata_batch_disease = create_batch(
                X_clr_df_train, train_metadata, batch_size, False, device
            )

            # Train encoder & classifier
            optimizer.zero_grad()
            prediction_scores = model(training_feature_batch)
            c_loss = model.disease_classifier_loss(prediction_scores, metadata_batch_disease)
            c_loss.backward()
            optimizer.step()
            scheduler.step(c_loss.item())

            # Compute training metrics
            pred_prob = torch.sigmoid(prediction_scores)
            pred_tag = (pred_prob > 0.5).float()
            disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
            if len(torch.unique(metadata_batch_disease)) > 1:
                auc = roc_auc_score(metadata_batch_disease.cpu(), pred_prob.detach().cpu())
                disease_f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())
            else:
                auc = np.nan  # Use NaN if AUC cannot be computed

            # Store training metrics
            train_losses.append(c_loss.item())
            train_accuracies.append(disease_acc)
            train_aucs.append(auc)
            train_f1s.append(disease_f1)


            # Evaluation Phase
            model.eval()
            with torch.no_grad():
                # Validation data
                feature_batch_val, metadata_batch_disease_val = create_batch(
                    X_clr_df_val, val_metadata, batch_size=len(val_metadata), is_test=True, device=device
                )
                prediction_scores_val = model(feature_batch_val)
                c_loss_val = model.disease_classifier_loss(prediction_scores_val, metadata_batch_disease_val)

                # Compute validation metrics
                pred_prob_val = torch.sigmoid(prediction_scores_val)
                pred_tag_val = (pred_prob_val > 0.5).float()
                disease_acc_val = balanced_accuracy_score(metadata_batch_disease_val.cpu(), pred_tag_val.cpu())
                if len(torch.unique(metadata_batch_disease_val)) > 1:
                    auc_val = roc_auc_score(metadata_batch_disease_val.cpu(), pred_prob_val.detach().cpu())
                    f1_val = f1_score(metadata_batch_disease_val.cpu(), pred_tag_val.detach().cpu())
                else:
                    auc_val = np.nan  # Use NaN if AUC cannot be computed

                # Store validation metrics
                val_losses.append(c_loss_val.item())
                val_accuracies.append(disease_acc_val)
                val_aucs.append(auc_val)
                val_f1s.append(f1_val)

            # Analyze distance correlation and mutual information
            with torch.no_grad():
                feature0 = model.encoder(training_feature_ctrl)
                dc0 = dcor.u_distance_correlation_sqr(feature0.cpu().numpy(), metadata_ctrl_bmi)
                mi_ctrl = mutual_info_regression(feature0.cpu().numpy(), metadata_ctrl_bmi)

                feature1 = model.encoder(training_feature_disease)
                dc1 = dcor.u_distance_correlation_sqr(feature1.cpu().numpy(), metadata_disease_bmi)
                mi_disease = mutual_info_regression(feature1.cpu().numpy(), metadata_disease_bmi)

                dcs0.append(dc0)
                dcs1.append(dc1)
                mis0.append(mi_ctrl.mean())
                mis1.append(mi_disease.mean())

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {c_loss.item():.4f}, Train Acc: {disease_acc:.4f}, Train AUC: {auc:.4f}, "
                  f"Val Loss: {c_loss_val.item():.4f}, Val Acc: {disease_acc_val:.4f}, Val AUC: {auc_val:.4f}")


            # Early stopping check (optional)
            if disease_acc > best_disease_acc:
                best_disease_acc = disease_acc
                early_stop_patience = 0
            else:
                early_stop_patience += 1
            # Uncomment the early stopping if needed
            if early_stop_patience == early_stop_step:
                print("Early stopping triggered.")
                break

        # Save models
        torch.save(model.encoder.state_dict(), f'baseline_models/encoder_fold{fold}.pth')
        torch.save(model.disease_classifier.state_dict(), f'baseline_models/disease_classifier_fold{fold}.pth')
        print(f'Encoder saved to baseline_models/encoder_fold{fold}.pth.')
        print(f'Classifier saved to baseline_models/disease_classifier_fold{fold}.pth.')
        all_eval_accuracies.append(val_accuracies[-1])
        all_eval_aucs.append(val_aucs[-1])
        all_eval_f1s.append(val_f1s[-1])

        # Plot metrics
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                     train_aucs, val_aucs, train_f1s, val_f1s, dcs0, dcs1, mis0, mis1, fold)

    avg_eval_accuracy = np.mean(all_eval_accuracies)
    avg_eval_auc = np.nanmean(all_eval_aucs)
    save_eval_results(all_eval_accuracies, all_eval_aucs, all_eval_f1s)
    return avg_eval_accuracy, avg_eval_auc

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                     train_aucs, val_aucs, train_f1s, val_f1s, dcs0, dcs1, mis0, mis1, fold):
    """Plot training and validation metrics and save the figures."""
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'baseline_plots/loss_fold{fold}.png')
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'baseline_plots/accuracy_fold{fold}.png')
    plt.close()

    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_aucs, label='Train AUC', color='purple')
    plt.plot(epochs, val_aucs, label='Validation AUC', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'baseline_plots/auc_fold{fold}.png')
    plt.close()

    # Plot f1
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1s, label='Train F1', color='purple')
    plt.plot(epochs, val_f1s, label='Validation F1', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Training and Validation F1')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'baseline_plots/f1_fold{fold}.png')
    plt.close()

    # Plot DCOR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dcs0, label='DCOR Control', color='cyan')
    plt.plot(epochs, dcs1, label='DCOR Disease', color='magenta')
    plt.xlabel('Epoch')
    plt.ylabel('Distance Correlation')
    plt.title('Distance Correlation Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'baseline_plots/dcor_fold{fold}.png')
    plt.close()

    # Plot Mutual Information
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mis0, label='MI Control', color='olive')
    plt.plot(epochs, mis1, label='MI Disease', color='navy')
    plt.xlabel('Epoch')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'baseline_plots/mi_fold{fold}.png')
    plt.close()

def evaluate_baseline_model(model, relative_abundance, metadata, batch_size, t):
    """Evaluate the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size, is_test=True, device=device)
    with torch.no_grad():
        prediction_scores = model(feature_batch)

        pred_prob = torch.sigmoid(prediction_scores).detach().cpu()
        pred_tag = (pred_prob > 0.5).float()
        disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag)
        c_loss = model.disease_classifier_loss(prediction_scores.detach().cpu(), metadata_batch_disease.cpu())

        auc = calculate_auc(metadata_batch_disease.cpu(), pred_prob)
        f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())

    print(f"{t} result --> Accuracy: {disease_acc:.4f}, Loss: {c_loss.item():.4f}, AUC: {auc}, F1: {f1:.4f}")
    return disease_acc, auc

def calculate_auc(metadata_batch_disease, pred_prob):
    """Calculate AUC."""
    if len(torch.unique(metadata_batch_disease)) > 1:
        auc = roc_auc_score(metadata_batch_disease.cpu(), pred_prob.cpu())
        return auc
    else:
        print("Cannot compute ROC AUC as only one class is present.")
        return np.nan

def save_eval_results(accuracies, aucs, f1s, filename='baseline_evaluation_results.json'):
    """Save evaluation accuracies and AUCs to a JSON file."""
    results = {'accuracies': accuracies, 'aucs': aucs, 'f1s': f1s}
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {filename}.")
