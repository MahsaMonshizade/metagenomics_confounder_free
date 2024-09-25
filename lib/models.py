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
from sklearn.model_selection import KFold

class GAN(nn.Module):
    """
    Generative Adversarial Network class.
    """
    def __init__(self, input_dim, latent_dim=64, lr_r=0.001, lr_g=0.001, lr_c=0.005,
                 activation_fn=nn.SiLU, num_layers=3):
        """Initialize the GAN with an encoder, age regressor, and disease classifier."""
        super(GAN, self).__init__()

        self.encoder = self._build_encoder(input_dim, latent_dim, num_layers, activation_fn)
        self.age_regressor = self._build_regressor(latent_dim, activation_fn, num_layers)
        self.disease_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)

        self.age_regression_loss = InvCorrelationCoefficientLoss()
        self.distiller_loss = CorrelationCoefficientLoss()
        self.disease_classifier_loss = nn.BCEWithLogitsLoss()

        # Optimizers
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), lr=lr_c
        )
        self.optimizer_distiller = optim.Adam(self.encoder.parameters(), lr=lr_g)
        self.optimizer_regression_age = optim.Adam(self.age_regressor.parameters(), lr=lr_r)

        # Schedulers
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.scheduler_distiller = lr_scheduler.ReduceLROnPlateau(self.optimizer_distiller, mode='min', factor=0.5, patience=5)
        self.scheduler_regression_age = lr_scheduler.ReduceLROnPlateau(self.optimizer_regression_age, mode='min', factor=0.5, patience=5)

    def _build_encoder(self, input_dim, latent_dim, num_layers, activation_fn):
        """Build the encoder network."""
        layers = []
        current_dim = input_dim
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
        """Build the age regressor."""
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

    def train_model(self, epochs, relative_abundance, metadata, batch_size=64):
        """Train the GAN model using K-Fold cross-validation."""
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_eval_accuracies = []
        all_eval_aucs = []

        for fold, (train_index, val_index) in enumerate(kf.split(relative_abundance)):
            print(f"Fold {fold + 1}/5")
            X_clr_df_train = relative_abundance.iloc[train_index].reset_index(drop=True)
            X_clr_df_val = relative_abundance.iloc[val_index].reset_index(drop=True)
            train_metadata = metadata.iloc[train_index].reset_index(drop=True)
            val_metadata = metadata.iloc[val_index].reset_index(drop=True)

            best_loss = float('inf')
            early_stop_step = 10
            early_stop_patience = 0

            training_feature_ctrl, metadata_ctrl_age, training_feature_disease, metadata_disease_age = \
                dcor_calculation_data(X_clr_df_train, train_metadata)

            # Lists to store losses
            r_losses, g_losses, c_losses = [], [], []
            dcs0, dcs1, mis0, mis1 = [], [], [], []

            for epoch in range(epochs):
                # Create batches
                (training_feature_ctrl_batch, metadata_ctrl_batch_age,
                 training_feature_batch, metadata_batch_disease) = \
                    create_batch(X_clr_df_train, train_metadata, batch_size)

                # Train age regressor
                r_loss = self._train_age_regressor(training_feature_ctrl_batch, metadata_ctrl_batch_age)

                # Train distiller
                g_loss = self._train_distiller(training_feature_ctrl_batch, metadata_ctrl_batch_age)

                # Train encoder & classifier
                disease_acc, c_loss = self._train_classifier(training_feature_batch, metadata_batch_disease)

                # Store the losses
                r_losses.append(r_loss.item())
                g_losses.append(g_loss.item())
                c_losses.append(c_loss.item())

                # Early stopping check
                best_loss, early_stop_patience = self._check_early_stopping(
                    c_loss, best_loss, early_stop_patience, early_stop_step
                )
                if early_stop_patience == early_stop_step:
                    print("Early stopping triggered.")
                    break

                # Analyze distance correlation and learned features
                feature0 = self.encoder(training_feature_ctrl)
                dc0 = dcor.u_distance_correlation_sqr(feature0.detach().numpy(), metadata_ctrl_age)
                mi_ctrl = mutual_info_regression(feature0.detach().numpy(), metadata_ctrl_age)

                feature1 = self.encoder(training_feature_disease)
                dc1 = dcor.u_distance_correlation_sqr(feature1.detach().numpy(), metadata_disease_age)
                mi_disease = mutual_info_regression(feature1.detach().numpy(), metadata_disease_age)

                dcs0.append(dc0)
                dcs1.append(dc1)
                mis0.append(mi_ctrl.mean())
                mis1.append(mi_disease.mean())

                print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item():.4f}, "
                      f"g_loss: {g_loss.item():.4f}, c_loss: {c_loss.item():.4f}, "
                      f"disease_acc: {disease_acc:.4f}, dc0: {dc0:.4f}, dc1: {dc1:.4f}")

                # Evaluate
                eval_accuracy, eval_auc = self.evaluate(X_clr_df_val, val_metadata, val_metadata.shape[0], 'eval')
                _, _ = self.evaluate(X_clr_df_train, train_metadata, train_metadata.shape[0], 'train')

            # Save models
            torch.save(self.encoder.state_dict(), f'models/encoder_fold{fold}.pth')
            torch.save(self.disease_classifier.state_dict(), f'models/disease_classifier_fold{fold}.pth')
            print(f'Encoder saved to models/encoder_fold{fold}.pth.')
            print(f'Classifier saved to models/disease_classifier_fold{fold}.pth.')
            all_eval_accuracies.append(eval_accuracy)
            all_eval_aucs.append(eval_auc)
            self.plot_losses(r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, fold)
        avg_eval_accuracy = np.mean(all_eval_accuracies)
        avg_eval_auc = np.mean(all_eval_aucs)
        self.save_eval_results(all_eval_accuracies, all_eval_aucs)
        return avg_eval_accuracy, avg_eval_auc

    def _train_age_regressor(self, training_feature_ctrl_batch, metadata_ctrl_batch_age):
        """Train the age regressor."""
        self.optimizer_regression_age.zero_grad()
        for param in self.encoder.parameters():
            param.requires_grad = False

        encoded_features = self.encoder(training_feature_ctrl_batch)
        age_prediction = self.age_regressor(encoded_features)
        r_loss = self.age_regression_loss(metadata_ctrl_batch_age.view(-1, 1), age_prediction)
        r_loss.backward()
        self.optimizer_regression_age.step()

        for param in self.encoder.parameters():
            param.requires_grad = True

        return r_loss

    def _train_distiller(self, training_feature_ctrl_batch, metadata_ctrl_batch_age):
        """Train the distiller."""
        self.optimizer_distiller.zero_grad()
        for param in self.age_regressor.parameters():
            param.requires_grad = False

        encoder_features = self.encoder(training_feature_ctrl_batch)
        predicted_age = self.age_regressor(encoder_features)
        g_loss = self.distiller_loss(metadata_ctrl_batch_age.view(-1, 1), predicted_age)
        g_loss.backward()

        self.optimizer_distiller.step()

        for param in self.age_regressor.parameters():
            param.requires_grad = True

        return g_loss

    def _train_classifier(self, training_feature_batch, metadata_batch_disease):
        """Train the disease classifier."""
        self.optimizer.zero_grad()
        encoded_feature_batch = self.encoder(training_feature_batch)
        prediction_scores = self.disease_classifier(encoded_feature_batch).view(-1)
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease)
        c_loss.backward()
        pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float()
        disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
        self.optimizer.step()
        self.scheduler.step(disease_acc)

        return disease_acc, c_loss

    def _check_early_stopping(self, c_loss, best_loss, early_stop_patience, early_stop_step):
        """Check for early stopping condition."""
        if c_loss < best_loss:
            best_loss = c_loss
            early_stop_patience = 0
        else:
            early_stop_patience += 1
        return best_loss, early_stop_patience

    def plot_losses(self, r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, fold):
        """Plot training losses and save the figures."""
        self._plot_single_loss(g_losses, 'g_loss', 'green', f'confounder_free_age_gloss_fold{fold}.png')
        self._plot_single_loss(r_losses, 'r_loss', 'red', f'confounder_free_age_rloss_fold{fold}.png')
        self._plot_single_loss(c_losses, 'c_loss', 'blue', f'confounder_free_age_closs_fold{fold}.png')
        self._plot_single_loss(dcs0, 'dc0', 'orange', f'confounder_free_age_dc0_fold{fold}.png')
        self._plot_single_loss(dcs1, 'dc1', 'orange', f'confounder_free_age_dc1_fold{fold}.png')
        self._plot_single_loss(mis0, 'mi0', 'purple', f'confounder_free_age_mi0_fold{fold}.png')
        self._plot_single_loss(mis1, 'mi1', 'purple', f'confounder_free_age_mi1_fold{fold}.png')

    def _plot_single_loss(self, values, label, color, filename):
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

    def evaluate(self, relative_abundance, metadata, batch_size, t):
        """Evaluate the trained GAN model."""
        feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size, is_test=True)
        encoded_feature_batch = self.encoder(feature_batch)
        prediction_scores = self.disease_classifier(encoded_feature_batch).view(-1)

        pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float()
        disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease)

        auc = self._calculate_auc(metadata_batch_disease, prediction_scores, t)
        f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())

        print(f"{t} result --> Accuracy: {disease_acc:.4f}, Loss: {c_loss.item():.4f}, AUC: {auc}, F1: {f1:.4f}")
        return disease_acc, auc

    def _calculate_auc(self, metadata_batch_disease, prediction_scores, t):
        """Calculate AUC."""
        if len(torch.unique(metadata_batch_disease)) > 1:
            auc = roc_auc_score(metadata_batch_disease.cpu(), torch.sigmoid(prediction_scores).detach().cpu())
            return auc
        else:
            print("Cannot compute ROC AUC as only one class is present.")
            return None

    def save_eval_results(self, accuracies, aucs, filename='evaluation_results.json'):
        """Save evaluation accuracies and AUCs to a JSON file."""
        results = {'accuracies': accuracies, 'aucs': aucs}
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation results saved to {filename}.")
