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
        self.age_regressor = self._build_regressor(latent_dim, activation_fn, num_layers)
        # self.bmi_regressor = self._build_regressor(latent_dim, activation_fn, num_layers)
        self.disease_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)

        self.age_regression_loss = InvCorrelationCoefficientLoss()
        # self.bmi_regression_loss = InvCorrelationCoefficientLoss()
        self.distiller_loss = CorrelationCoefficientLoss()
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
        """Forward pass through the encoder."""
        encoded = self.encoder(x)
        return encoded

def train_model(model, epochs, relative_abundance, metadata, batch_size=64, lr_r=0.001, lr_g=0.001, lr_c=0.005):
    """Train the GAN model using K-Fold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_accuracies = []
    all_eval_aucs = []

    for fold, (train_index, val_index) in enumerate(skf.split(relative_abundance, metadata['disease'])):
        print(f"\nFold {fold + 1}/5")

        # Re-initialize the model and optimizer
        model = GAN(input_dim=model.input_dim, latent_dim=model.latent_dim,
                    activation_fn=model.activation_fn, num_layers=model.num_layers)
        model.initialize_weights()
        model.to(device)

        # Initialize optimizers and schedulers
        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=lr_c
        )
        optimizer_distiller = optim.Adam(model.encoder.parameters(), lr=lr_g)
        optimizer_regression_age = optim.Adam(model.age_regressor.parameters(), lr=lr_r)
        # optimizer_regression_bmi = optim.Adam(model.bmi_regressor.parameters(), lr=lr_r)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        scheduler_distiller = lr_scheduler.ReduceLROnPlateau(optimizer_distiller, mode='min', factor=0.5, patience=5)
        scheduler_regression_age = lr_scheduler.ReduceLROnPlateau(optimizer_regression_age, mode='min', factor=0.5, patience=5)
        # scheduler_regression_bmi = lr_scheduler.ReduceLROnPlateau(optimizer_regression_bmi, mode='min', factor=0.5, patience=5)

        X_clr_df_train = relative_abundance.iloc[train_index].reset_index(drop=True)
        X_clr_df_val = relative_abundance.iloc[val_index].reset_index(drop=True)
        train_metadata = metadata.iloc[train_index].reset_index(drop=True)
        val_metadata = metadata.iloc[val_index].reset_index(drop=True)

        best_loss = float('inf')
        early_stop_step = 10
        early_stop_patience = 0

        (training_feature_ctrl, metadata_ctrl_age, metadata_ctrl_bmi,
         training_feature_disease, metadata_disease_age, metadata_disease_bmi) = dcor_calculation_data(
            X_clr_df_train, train_metadata, device
        )

        # Lists to store losses
        r_losses, g_losses, c_losses = [], [], []
        dcs0, dcs1, mis0, mis1 = [], [], [], []

        for epoch in range(epochs):
            # Create batches
            (training_feature_ctrl_batch, metadata_ctrl_batch_age, metadata_ctrl_batch_bmi,
             training_feature_batch, metadata_batch_disease) = create_batch(
                X_clr_df_train, train_metadata, batch_size, False, device
            )


            # ----------------------------
            # Train age regressor (r_loss)
            # ----------------------------
            optimizer_regression_age.zero_grad()
            for param in model.encoder.parameters():
                param.requires_grad = False

            encoded_features = model.encoder(training_feature_ctrl_batch)
            age_prediction = model.age_regressor(encoded_features)
            r_loss = model.age_regression_loss(metadata_ctrl_batch_age.view(-1, 1), age_prediction)
            r_loss.backward()
            optimizer_regression_age.step()
            scheduler_regression_age.step(r_loss)

            for param in model.encoder.parameters():
                param.requires_grad = True


            # ----------------------------
            # Train BMI regressor (r_loss)
            # ----------------------------
            # optimizer_regression_bmi.zero_grad()
            # for param in model.encoder.parameters():
            #     param.requires_grad = False

            # encoded_features = model.encoder(training_feature_ctrl_batch)
            # bmi_prediction = model.bmi_regressor(encoded_features)
            # r_loss = model.bmi_regression_loss(metadata_ctrl_batch_bmi.view(-1, 1), bmi_prediction)
            # r_loss.backward()
            # optimizer_regression_bmi.step()
            # scheduler_regression_bmi.step(r_loss)

            # for param in model.encoder.parameters():
            #     param.requires_grad = True


            # ----------------------------
            # Train distiller (g_loss)
            # ----------------------------
            optimizer_distiller.zero_grad()
            for param in model.age_regressor.parameters():
                param.requires_grad = False

            encoder_features = model.encoder(training_feature_ctrl_batch)
            predicted_age = model.age_regressor(encoder_features)
            g_loss = model.distiller_loss(metadata_ctrl_batch_age.view(-1, 1), predicted_age)
            g_loss.backward()
            optimizer_distiller.step()
            scheduler_distiller.step(g_loss)

            for param in model.age_regressor.parameters():
                param.requires_grad = True

            # ----------------------------
            # Train encoder & classifier (c_loss)
            # ----------------------------
            optimizer.zero_grad()
            encoded_feature_batch = model.encoder(training_feature_batch)
            prediction_scores = model.disease_classifier(encoded_feature_batch).view(-1)
            c_loss = model.disease_classifier_loss(prediction_scores, metadata_batch_disease)
            c_loss.backward()
            pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float()
            disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
            optimizer.step()
            scheduler.step(disease_acc)

            # Store the losses
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())

            # Early stopping check (optional)
            if c_loss < best_loss:
                best_loss = c_loss
                early_stop_patience = 0
            else:
                early_stop_patience += 1
            # Uncomment the early stopping if needed
            if early_stop_patience == early_stop_step:
                print("Early stopping triggered.")
                break

            # Analyze distance correlation and learned features
            with torch.no_grad():
                feature0 = model.encoder(training_feature_ctrl)
                dc0 = dcor.u_distance_correlation_sqr(feature0.cpu().numpy(), metadata_ctrl_age)
                mi_ctrl = mutual_info_regression(feature0.cpu().numpy(), metadata_ctrl_age)
                # dc0 = dcor.u_distance_correlation_sqr(feature0.cpu().numpy(), metadata_ctrl_bmi)
                # mi_ctrl = mutual_info_regression(feature0.cpu().numpy(), metadata_ctrl_bmi)


                feature1 = model.encoder(training_feature_disease)
                dc1 = dcor.u_distance_correlation_sqr(feature1.cpu().numpy(), metadata_disease_age)
                mi_disease = mutual_info_regression(feature1.cpu().numpy(), metadata_disease_age)
                # dc1 = dcor.u_distance_correlation_sqr(feature1.cpu().numpy(), metadata_disease_bmi)
                # mi_disease = mutual_info_regression(feature1.cpu().numpy(), metadata_disease_bmi)

            dcs0.append(dc0)
            dcs1.append(dc1)
            mis0.append(mi_ctrl.mean())
            mis1.append(mi_disease.mean())

            print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item():.4f}, "
                  f"g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}, "
                  f"disease_acc: {disease_acc:.4f}, dc0: {dc0:.4f}, dc1: {dc1:.4f}")

            # Evaluate
            eval_accuracy, eval_auc = evaluate(
                model, X_clr_df_val, val_metadata, val_metadata.shape[0], 'eval', device
            )
            # Optionally, evaluate on training data
            # _, _ = evaluate(model, X_clr_df_train, train_metadata, train_metadata.shape[0], 'train', device)

        # Save models
        torch.save(model.encoder.state_dict(), f'models/encoder_fold{fold}.pth')
        torch.save(model.disease_classifier.state_dict(), f'models/disease_classifier_fold{fold}.pth')
        print(f'Encoder saved to models/encoder_fold{fold}.pth.')
        print(f'Classifier saved to models/disease_classifier_fold{fold}.pth.')

        all_eval_accuracies.append(eval_accuracy)
        all_eval_aucs.append(eval_auc)
        plot_losses(r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, fold)

    avg_eval_accuracy = np.mean(all_eval_accuracies)
    avg_eval_auc = np.mean(all_eval_aucs)
    save_eval_results(all_eval_accuracies, all_eval_aucs)
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
    return disease_acc, auc

def calculate_auc(metadata_batch_disease, prediction_scores):
    """Calculate AUC."""
    if len(torch.unique(metadata_batch_disease)) > 1:
        auc = roc_auc_score(metadata_batch_disease.cpu(), torch.sigmoid(prediction_scores).detach().cpu())
        return auc
    else:
        print("Cannot compute ROC AUC as only one class is present.")
        return None

def plot_losses(r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, fold):
    """Plot training losses and save the figures."""
    plot_single_loss(g_losses, 'g_loss', 'green', f'confounder_free_age_gloss_fold{fold}.png')
    plot_single_loss(r_losses, 'r_loss', 'red', f'confounder_free_age_rloss_fold{fold}.png')
    plot_single_loss(c_losses, 'c_loss', 'blue', f'confounder_free_age_closs_fold{fold}.png')
    plot_single_loss(dcs0, 'dc0', 'orange', f'confounder_free_age_dc0_fold{fold}.png')
    plot_single_loss(dcs1, 'dc1', 'orange', f'confounder_free_age_dc1_fold{fold}.png')
    plot_single_loss(mis0, 'mi0', 'purple', f'confounder_free_age_mi0_fold{fold}.png')
    plot_single_loss(mis1, 'mi1', 'purple', f'confounder_free_age_mi1_fold{fold}.png')

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

def save_eval_results(accuracies, aucs, filename='evaluation_results.json'):
    """Save evaluation accuracies and AUCs to a JSON file."""
    results = {'accuracies': accuracies, 'aucs': aucs}
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {filename}.")
