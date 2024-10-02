# models.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init
import numpy as np
from losses import CorrelationCoefficientLoss, InvCorrelationCoefficientLoss
from data_processing import create_batch, dcor_calculation_data
import matplotlib.pyplot as plt
import json
import dcor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedKFold

def previous_power_of_two(n):
    """Return the greatest power of two less than or equal to n."""
    return 2 ** (n.bit_length() - 1)


class GAN(nn.Module):
    """
    Generative Adversarial Network class.
    """
    def __init__(self, input_dim, latent_dim=64, activation_fn=nn.SiLU, num_layers=1):
        """Initialize the GAN with an encoder, age regressor, BMI regressor, and disease classifier."""
        super(GAN, self).__init__()

        # Store parameters for re-initialization
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.num_layers = num_layers

        self.encoder = self._build_encoder(input_dim, latent_dim, num_layers, activation_fn)
        self.disease_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)

        # Confounder regressors
        self.age_regressor = nn.Linear(latent_dim, 1, bias=False)
        self.bmi_regressor = nn.Linear(latent_dim, 1, bias=False)

        self.disease_classifier_loss = nn.BCEWithLogitsLoss()
        self.confounder_regression_loss = nn.MSELoss()

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
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                activation_fn()
            ])
            current_dim = current_dim // 2
        layers.extend([
            nn.Linear(current_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            activation_fn()
        ])
        return nn.Sequential(*layers)

    def _build_regressor(self, latent_dim, activation_fn, num_layers):
        """Build the age or BMI regressor."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                activation_fn()
            ])
            current_dim = current_dim // 2
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, activation_fn, num_layers):
        """Build the disease classifier."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.BatchNorm1d(current_dim // 2),
                activation_fn()
            ])
            current_dim = current_dim // 2
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the encoder and orthogonal projection."""
        encoded = self.encoder(x)
        projected = self.orthogonal_projection(encoded)
        return projected
    
    def orthogonal_projection(self, z, epsilon=1e-5):
        """Project latent representations onto the orthogonal complement of confounder directions."""
        # Get confounder directions (weights from regressors)
        w_age = self.age_regressor.weight  # Shape: [1, latent_dim]
        w_bmi = self.bmi_regressor.weight  # Shape: [1, latent_dim]

        # Stack weights to form confounder matrix W
        W = torch.cat([w_age, w_bmi], dim=0)  # Shape: [2, latent_dim]

        # Compute W W^T and add epsilon for numerical stability
        WWT = W @ W.t() + epsilon * torch.eye(2, device=z.device)

        # Compute orthogonal projection matrix P
        WWT_inv = torch.inverse(WWT)  # Shape: [2, 2]
        P = torch.eye(self.latent_dim, device=z.device) - W.t() @ WWT_inv @ W

        # Project z
        z_projected = z @ P  # Shape: [batch_size, latent_dim]
        return z_projected


def train_model(model, epochs, relative_abundance, metadata, batch_size=64, lr_r=0.001, lr_g=0.001, lr_c=0.005):
    """Train the GAN model with orthogonal projection using K-Fold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_accuracies = []
    all_eval_aucs = []
    all_eval_f1s = []

    for fold, (train_index, val_index) in enumerate(skf.split(relative_abundance, metadata['disease'])):
        print(f"\nFold {fold + 1}/5")

        # Re-initialize the model and optimizers
        model = GAN(input_dim=model.input_dim, latent_dim=model.latent_dim,
                    activation_fn=model.activation_fn, num_layers=model.num_layers)
        model.initialize_weights()
        model.to(device)

        # Initialize optimizers
        optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=lr_g)
        optimizer_classifier = optim.Adam(model.disease_classifier.parameters(), lr=lr_c)
        optimizer_regressors = optim.Adam(
            list(model.age_regressor.parameters()) + list(model.bmi_regressor.parameters()), lr=lr_r
        )

        # Learning rate schedulers (optional)
        scheduler_encoder = lr_scheduler.ReduceLROnPlateau(optimizer_encoder, mode='min', factor=0.5, patience=5)
        scheduler_classifier = lr_scheduler.ReduceLROnPlateau(optimizer_classifier, mode='min', factor=0.5, patience=5)
        scheduler_regressors = lr_scheduler.ReduceLROnPlateau(optimizer_regressors, mode='min', factor=0.5, patience=5)

        # Split data into training and validation sets
        X_clr_df_train = relative_abundance.iloc[train_index].reset_index(drop=True)
        X_clr_df_val = relative_abundance.iloc[val_index].reset_index(drop=True)
        train_metadata = metadata.iloc[train_index].reset_index(drop=True)
        val_metadata = metadata.iloc[val_index].reset_index(drop=True)

        best_disease_acc = 0
        early_stop_step = 20
        early_stop_patience = 0

        # Lists to store losses and metrics
        r_age_losses, r_bmi_losses, c_losses = [], [], []
        train_disease_accs, val_disease_accs = [], []
        train_disease_aucs, val_disease_aucs = [], []
        train_disease_f1s, val_disease_f1s = [], []

        for epoch in range(epochs):
            # Set model to training mode
            model.train()

            # Create batches
            (training_feature_ctrl_batch, metadata_ctrl_batch_age, metadata_ctrl_batch_bmi,
             training_feature_batch, metadata_batch_disease) = create_batch(
                X_clr_df_train, train_metadata, batch_size, is_test=False, device=device
            )

            # ----------------------------
            # Train Confounder Regressors (r_loss)
            # ----------------------------
            optimizer_regressors.zero_grad()
            with torch.no_grad():
                encoded_ctrl = model.encoder(training_feature_ctrl_batch)
            age_pred = model.age_regressor(encoded_ctrl)
            bmi_pred = model.bmi_regressor(encoded_ctrl)

            r_loss_age = model.confounder_regression_loss(age_pred, metadata_ctrl_batch_age.view(-1, 1))
            r_loss_bmi = model.confounder_regression_loss(bmi_pred, metadata_ctrl_batch_bmi.view(-1, 1))
            r_loss = r_loss_age + r_loss_bmi
            r_loss.backward()
            optimizer_regressors.step()
            scheduler_regressors.step(r_loss)

            # ----------------------------
            # Train Encoder and Classifier (c_loss)
            # ----------------------------
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()

            # Forward pass with projection
            encoded = model.encoder(training_feature_batch)
            projected = model.orthogonal_projection(encoded)
            prediction_scores = model.disease_classifier(projected).view(-1)

            c_loss = model.disease_classifier_loss(prediction_scores, metadata_batch_disease)
            c_loss.backward()

            optimizer_encoder.step()
            optimizer_classifier.step()
            scheduler_encoder.step(c_loss)
            scheduler_classifier.step(c_loss)

            # Compute training metrics
            pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float()
            disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
            disease_auc = calculate_auc(metadata_batch_disease.cpu(), prediction_scores.cpu())
            disease_f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())

            # Store losses and metrics
            r_age_losses.append(r_loss_age.item())
            r_bmi_losses.append(r_loss_bmi.item())
            c_losses.append(c_loss.item())

            train_disease_accs.append(disease_acc)
            train_disease_aucs.append(disease_auc)
            train_disease_f1s.append(disease_f1)

            # Early stopping check
            if disease_acc > best_disease_acc:
                best_disease_acc = disease_acc
                early_stop_patience = 0
            else:
                early_stop_patience += 1

            if early_stop_patience == early_stop_step:
                print("Early stopping triggered.")
                break

            # Print training status
            print(f"Epoch {epoch + 1}/{epochs}, r_loss_age: {r_loss_age.item():.4f}, r_loss_bmi: {r_loss_bmi.item():.4f}, "
                  f"c_loss: {c_loss.item():.4f}, disease_acc: {disease_acc:.4f}")

            # ----------------------------
            # Evaluate on Validation Set
            # ----------------------------
            model.eval()
            with torch.no_grad():
                # Prepare validation data
                val_features, val_metadata_disease = create_batch(
                    X_clr_df_val, val_metadata, batch_size=val_metadata.shape[0], is_test=True, device=device
                )
                encoded_val = model.encoder(val_features)
                projected_val = model.orthogonal_projection(encoded_val)
                val_prediction_scores = model.disease_classifier(projected_val).view(-1)

                # Compute validation metrics
                val_pred_tag = (torch.sigmoid(val_prediction_scores) > 0.5).float()
                val_disease_acc = balanced_accuracy_score(val_metadata_disease.cpu(), val_pred_tag.cpu())
                val_disease_auc = calculate_auc(val_metadata_disease.cpu(), val_prediction_scores.cpu())
                val_disease_f1 = f1_score(val_metadata_disease.cpu(), val_pred_tag.cpu())

                val_disease_accs.append(val_disease_acc)
                val_disease_aucs.append(val_disease_auc)
                val_disease_f1s.append(val_disease_f1)

                print(f"Validation --> Accuracy: {val_disease_acc:.4f}, AUC: {val_disease_auc:.4f}, F1: {val_disease_f1:.4f}")

        # Save models
        torch.save(model.encoder.state_dict(), f'models/encoder_fold{fold}.pth')
        torch.save(model.disease_classifier.state_dict(), f'models/disease_classifier_fold{fold}.pth')
        torch.save(model.age_regressor.state_dict(), f'models/age_regressor_fold{fold}.pth')
        torch.save(model.bmi_regressor.state_dict(), f'models/bmi_regressor_fold{fold}.pth')
        print(f'Models saved for fold {fold}.')

        # Store evaluation metrics
        all_eval_accuracies.append(val_disease_accs[-1])
        all_eval_aucs.append(val_disease_aucs[-1])
        all_eval_f1s.append(val_disease_f1s[-1])

        # Optionally, plot losses and metrics
        plot_losses(r_age_losses, r_bmi_losses, c_losses, train_disease_accs, val_disease_accs,
                    train_disease_aucs, val_disease_aucs, train_disease_f1s, val_disease_f1s, fold)

    # Compute average evaluation metrics across folds
    avg_eval_accuracy = np.mean(all_eval_accuracies)
    avg_eval_auc = np.mean(all_eval_aucs)
    avg_eval_f1 = np.mean(all_eval_f1s)
    save_eval_results(all_eval_accuracies, all_eval_aucs, all_eval_f1s)
    print(f"Average Evaluation Accuracy: {avg_eval_accuracy:.4f}, AUC: {avg_eval_auc:.4f}, F1: {avg_eval_f1:.4f}")
    return avg_eval_accuracy, avg_eval_auc


def evaluate(model, relative_abundance, metadata, batch_size, t, device):
    """Evaluate the trained GAN model."""
    model.to(device)

    feature_batch, metadata_batch_disease = create_batch(
        relative_abundance, metadata, batch_size, is_test=True, device=device
    )
    with torch.no_grad():
        encoded_feature_batch = model.encoder(feature_batch)
        prediction_scores = model.disease_classifier(encoded_feature_batch).view(-1)

    pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float().cpu()
    disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
    c_loss = model.disease_classifier_loss(prediction_scores.cpu(), metadata_batch_disease.cpu())

    auc = calculate_auc(metadata_batch_disease.cpu(), prediction_scores.cpu())
    f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())

    print(f"{t} result --> Accuracy: {disease_acc:.4f}, Loss: {c_loss.item():.4f}, AUC: {auc}, F1: {f1:.4f}")
    return disease_acc, auc, f1

def calculate_auc(metadata_batch_disease, prediction_scores):
    """Calculate AUC."""
    if len(torch.unique(metadata_batch_disease)) > 1:
        auc = roc_auc_score(metadata_batch_disease.cpu(), torch.sigmoid(prediction_scores).detach().cpu())
        return auc
    else:
        print("Cannot compute ROC AUC as only one class is present.")
        return None

def plot_losses(r_age_losses, r_bmi_losses, c_losses, train_disease_accs, val_disease_accs,
                train_disease_aucs, val_disease_aucs, train_disease_f1s, val_disease_f1s, fold):
    """Plot training losses and save the figures."""
    plot_single_loss(c_losses, 'c_loss', 'blue', f'confounder_free_closs_fold{fold}.png')
    plot_single_loss(r_age_losses, 'r_loss', 'red', f'confounder_free_age_rloss_fold{fold}.png')
    plot_single_loss(r_bmi_losses, 'r_loss', 'red', f'confounder_free_bmi_rloss_fold{fold}.png')

    plot_single_loss(train_disease_accs, 'train_disease_acc', 'red', f'confounder_free_train_disease_acc_fold{fold}.png')
    plot_single_loss(train_disease_aucs, 'train_disease_auc', 'red', f'confounder_free_train_disease_auc_fold{fold}.png')
    plot_single_loss(train_disease_f1s, 'train_disease_f1', 'red', f'confounder_free_train_disease_f1_fold{fold}.png')
    plot_single_loss(val_disease_accs, 'val_disease_acc', 'red', f'confounder_free_val_disease_acc_fold{fold}.png')
    plot_single_loss(val_disease_aucs, 'val_disease_auc', 'red', f'confounder_free_val_disease_auc_fold{fold}.png')
    plot_single_loss(val_disease_f1s, 'val_disease_f1', 'red', f'confounder_free_val_disease_f1_fold{fold}.png')
    

def plot_single_loss(values, label, color, filename):
    """Helper function to plot a single loss."""
    plt.figure(figsize=(12, 6))
    plt.plot(values, label=label, color=color)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(f'{label} Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{filename}')
    plt.close()

def save_eval_results(accuracies, aucs, f1s, filename='evaluation_results.json'):
    """Save evaluation accuracies and AUCs to a JSON file."""
    results = {'accuracies': accuracies, 'aucs': aucs, 'f1s': f1s}
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {filename}.")
