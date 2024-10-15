# models.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init
from torch.utils.data import DataLoader
import numpy as np
from data_processing import dcor_calculation_data, DiseaseDataset, MixedDataset
from losses import CorrelationCoefficientLoss, calculate_pearson_correlation
import matplotlib.pyplot as plt
import json
import dcor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from itertools import cycle


def previous_power_of_two(n):
    """Return the greatest power of two less than or equal to n."""
    return 2 ** (n.bit_length() - 1)


class GAN(nn.Module):
    """
    Generative Adversarial Network class.
    """
    def __init__(self, input_dim, latent_dim=64, activation_fn=nn.SiLU, num_layers=1):
        """Initialize the GAN with an encoder, age regressor, Drug Classifier, and disease classifier."""
        super(GAN, self).__init__()

        # Store parameters for re-initialization
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_fn = activation_fn
        self.num_layers = num_layers

        self.encoder = self._build_encoder(input_dim, latent_dim, num_layers, activation_fn)
        self.drug_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)

        # self.drug_classification_loss = nn.BCEWithLogitsLoss()
        self.distiller_loss = CorrelationCoefficientLoss()

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
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


def train_model(model, epochs, relative_abundance, metadata, batch_size=64, lr_r=0.0002, lr_g=0.0002, lr_c=0.0001):
    """Train the GAN model using K-Fold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_eval_accuracies = []
    all_eval_aucs = []
    all_eval_f1s = []

    for fold, (train_index, val_index) in enumerate(skf.split(relative_abundance, metadata['PATGROUPFINAL_C'])):
        print(f"\nFold {fold + 1}/5")

        # Re-initialize the model and optimizer
        model = GAN(input_dim=model.input_dim, latent_dim=model.latent_dim,
                    activation_fn=model.activation_fn, num_layers=model.num_layers)
        model.initialize_weights()
        model.to(device)

        # Initialize optimizers

        optimizer_distiller = optim.Adam(model.encoder.parameters(), lr=lr_g)
        optimizer_classification_drug = optim.Adam(model.drug_classifier.parameters(), lr=lr_r)

        # Initialize schedulers
        scheduler_distiller = lr_scheduler.ReduceLROnPlateau(optimizer_distiller, mode='min', factor=0.5, patience=5)
        scheduler_classification_drug = lr_scheduler.ReduceLROnPlateau(optimizer_classification_drug, mode='min', factor=0.5, patience=5)

        X_clr_df_train = relative_abundance.iloc[train_index].reset_index(drop=True)
        train_metadata = metadata.iloc[train_index].reset_index(drop=True)
  

        # Create datasets and DataLoaders
        disease_dataset = DiseaseDataset(X_clr_df_train, train_metadata, device)

        disease_loader = DataLoader(disease_dataset, batch_size=batch_size, shuffle=True)

        # Initialize iterators
        disease_iter = iter(disease_loader)

        # Determine the number of batches per epoch
        num_batches = len(disease_loader)


        (training_feature_ctrl, metadata_ctrl_drug,
         training_feature_disease, metadata_disease_drug) = dcor_calculation_data(
            X_clr_df_train, train_metadata, device
        )

        # Lists to store losses and metrics per epoch
        train_disease_accs = []
        train_disease_aucs= []
        train_disease_f1s = []
        g_losses = []
        dcs0, dcs1, mis0, mis1 = [], [], [], []

        for epoch in range(epochs):
            model.train()
            # Reset iterators and accumulators
            disease_iter = iter(disease_loader)
            epoch_g_loss = 0
            epoch_train_preds, epoch_train_labels = [], []

            for i in range(num_batches):
                # Get next batch from disease_loader
                
                training_feature_disease_batch, metadata_disease_batch_drug = next(disease_iter)
               

                # ----------------------------
                # Train distiller (g_loss)
                # ----------------------------
                optimizer_distiller.zero_grad()

                # Manually zero out the drug_classifier's gradients
                for param in model.drug_classifier.parameters():
                    param.grad = None  # or param.grad.detach_().zero_()

                model.drug_classifier.requires_grad_(False)
                
                encoded_features = model.encoder(training_feature_disease_batch)
              
                predicted_drug = model.drug_classifier(encoded_features).view(-1)
                # print("predicted_drug")
                # print(predicted_drug)
                pred_drug = torch.sigmoid(predicted_drug)
                # print("pred_drug")
                # print(pred_drug)
                g_loss = model.distiller_loss(metadata_disease_batch_drug, predicted_drug)
                g_loss.backward()

                print("Gradients for encoder distiller layers:")
                for name, param in model.encoder.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm()}")

                print("Gradients for drug classifier distiller layers:")
                for name, param in model.drug_classifier.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm()}")

                optimizer_distiller.step()


                model.drug_classifier.requires_grad_(True)

                # ----------------------------
                # Train encoder & classifier (c_loss)
                # ----------------------------
               

                epoch_g_loss += g_loss.item()

                # Collect predictions and labels for epoch metrics
                pred_tag = (pred_drug > 0.5).float()
                epoch_train_preds.append(pred_tag.cpu())
                epoch_train_labels.append(metadata_disease_batch_drug.cpu())

            # Compute average losses over the epoch
          
            epoch_g_loss /= num_batches

             # Store the losses
            
            g_losses.append(epoch_g_loss)

            # Compute training metrics for the epoch
            epoch_train_preds = torch.cat(epoch_train_preds)
            epoch_train_labels = torch.cat(epoch_train_labels)

            train_disease_acc = balanced_accuracy_score(epoch_train_labels, epoch_train_preds)
            train_disease_auc = calculate_auc(epoch_train_labels.view(-1, 1), epoch_train_preds)
            train_disease_f1 = f1_score(epoch_train_labels, epoch_train_preds)

            train_disease_accs.append(train_disease_acc)
            train_disease_aucs.append(train_disease_auc)
            train_disease_f1s.append(train_disease_f1)

            # Step schedulers per epoch
            scheduler_distiller.step(epoch_g_loss)
            

            # Analyze distance correlation and learned features
            with torch.no_grad():
                feature0 = model.encoder(training_feature_ctrl)
                dc0 = dcor.u_distance_correlation_sqr(feature0.cpu().numpy(), metadata_ctrl_drug)
                mi_ctrl = mutual_info_classif(feature0.cpu().numpy(), metadata_ctrl_drug, discrete_features=False, random_state=42)
            
                feature1 = model.encoder(training_feature_disease)
                dc1 = dcor.u_distance_correlation_sqr(feature1.cpu().numpy(), metadata_disease_drug)
                mi_disease = mutual_info_classif(feature1.cpu().numpy(), metadata_disease_drug, discrete_features=False, random_state=42)
 
            dcs0.append(dc0)
            dcs1.append(dc1)
            mis0.append(mi_ctrl)
            mis1.append(mi_disease)


            # Print epoch statistics
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"g_loss: {epoch_g_loss:.4f}, "
                  f"train_acc: {train_disease_acc:.4f}, "
                  f"dc0: {dc0:.4f}, dc1: {dc1:.4f}")

        # Save models
        torch.save(model.encoder.state_dict(), f'models/encoder_fold{fold}.pth')
        print(f'Encoder saved to models/encoder_fold{fold}.pth.')

        # Plotting losses and metrics
        plot_losses(
            g_losses, dcs0, dcs1, mis0, mis1,
            train_disease_accs, train_disease_aucs, train_disease_f1s,
            fold
        )


def calculate_auc(metadata_batch_disease, prediction_scores):
    """Calculate AUC."""
    if len(torch.unique(metadata_batch_disease)) > 1:
        auc = roc_auc_score(metadata_batch_disease.cpu(), torch.sigmoid(prediction_scores).detach().cpu())
        return auc
    else:
        print("Cannot compute ROC AUC as only one class is present.")
        return None

def plot_losses(g_losses, dcs0, dcs1, mis0, mis1, train_disease_accs, train_disease_aucs, train_disease_f1s, fold):
    """Plot training losses and save the figures."""
    plot_single_loss(g_losses, 'g_loss', 'green', f'confounder_free_drug_gloss_fold{fold}.png')
    plot_single_loss(dcs0, 'dc0', 'orange', f'confounder_free_drug_dc0_fold{fold}.png')
    plot_single_loss(dcs1, 'dc1', 'orange', f'confounder_free_drug_dc1_fold{fold}.png')
    plot_single_loss(mis0, 'mi0', 'purple', f'confounder_free_drug_mi0_fold{fold}.png')
    plot_single_loss(mis1, 'mi1', 'purple', f'confounder_free_drug_mi1_fold{fold}.png')
    plot_single_loss(train_disease_accs, 'train_disease_acc', 'red', f'confounder_free_drug_train_disease_acc_fold{fold}.png')
    plot_single_loss(train_disease_aucs, 'train_disease_auc', 'red', f'confounder_free_drug_train_disease_auc_fold{fold}.png')
    plot_single_loss(train_disease_f1s, 'train_disease_f1', 'red', f'confounder_free_drug_train_disease_f1_fold{fold}.png')
    
    

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