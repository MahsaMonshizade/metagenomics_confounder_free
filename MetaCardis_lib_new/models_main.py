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
        self.disease_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)

        # self.age_regression_loss = InvCorrelationCoefficientLoss()
        self.drug_classification_loss = nn.BCEWithLogitsLoss()
        self.distiller_loss = nn.BCEWithLogitsLoss()
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
        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=lr_c
        )
        optimizer_distiller = optim.Adam(model.encoder.parameters(), lr=lr_g)
        optimizer_classification_drug = optim.Adam(model.drug_classifier.parameters(), lr=lr_r)

        # Initialize schedulers
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        scheduler_distiller = lr_scheduler.ReduceLROnPlateau(optimizer_distiller, mode='min', factor=0.5, patience=5)
        scheduler_classification_drug = lr_scheduler.ReduceLROnPlateau(optimizer_classification_drug, mode='min', factor=0.5, patience=5)

        X_clr_df_train = relative_abundance.iloc[train_index].reset_index(drop=True)
        X_clr_df_val = relative_abundance.iloc[val_index].reset_index(drop=True)
        train_metadata = metadata.iloc[train_index].reset_index(drop=True)
        val_metadata = metadata.iloc[val_index].reset_index(drop=True)

        # Create datasets and DataLoaders
        disease_dataset = DiseaseDataset(X_clr_df_train, train_metadata, device)
        mixed_dataset = MixedDataset(X_clr_df_train, train_metadata, device)

        disease_loader = DataLoader(disease_dataset, batch_size=batch_size, shuffle=True)
        mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True)

        # Initialize iterators
        disease_iter = iter(disease_loader)

        # Determine the number of batches per epoch
        num_batches = len(mixed_loader)

        best_eval_acc = 0  # Changed variable name for clarity
        early_stop_step = 20
        early_stop_patience = 0

        (training_feature_ctrl, metadata_ctrl_drug,
         training_feature_disease, metadata_disease_drug) = dcor_calculation_data(
            X_clr_df_train, train_metadata, device
        )

        # Lists to store losses and metrics per epoch
        train_disease_accs, val_disease_accs = [], []
        train_disease_aucs, val_disease_aucs = [], []
        train_disease_f1s, val_disease_f1s = [], []
        r_losses, g_losses, c_losses = [], [], []
        dcs0, dcs1, mis0, mis1 = [], [], [], []

        for epoch in range(epochs):
            model.train()
            # Reset iterators and accumulators
            mixed_iter = iter(mixed_loader)
            epoch_r_loss, epoch_g_loss, epoch_c_loss = 0, 0, 0
            epoch_train_preds, epoch_train_labels = [], []

            for i in range(num_batches):
                # Get next batch from disease_loader
                try:
                    training_feature_disease_batch, metadata_disease_batch_drug = next(disease_iter)
                except StopIteration:
                    disease_iter = iter(disease_loader)
                    training_feature_disease_batch, metadata_disease_batch_drug = next(disease_iter)

                # Get next batch from mixed_loader
                training_feature_batch, metadata_batch_disease = next(mixed_iter)
                # print(f"batch_{i}")
                # print(metadata_disease_batch_drug)
                # ----------------------------
                # Train drug classification (r_loss)
                # ----------------------------
                optimizer_classification_drug.zero_grad()
                # Manually zero out the encoder's gradients
                for param in model.encoder.parameters():
                    param.grad = None  # or param.grad.detach_().zero_()

                model.encoder.requires_grad_(False)

                with torch.no_grad():
                    encoded_features = model.encoder(training_feature_disease_batch)
                drug_prediction = model.drug_classifier(encoded_features).view(-1)
                r_loss = model.drug_classification_loss(drug_prediction, metadata_disease_batch_drug)
                r_loss.backward()

                # print("Gradients for encoder layers:")
                # for name, param in model.encoder.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")

                # print("Gradients for drug classifier layers:")
                # for name, param in model.drug_classifier.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")
                        
                optimizer_classification_drug.step()

                model.encoder.requires_grad_(True)

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
                # pred_drug = torch.sigmoid(predicted_drug)
                # print("pred_drug")
                # print(pred_drug)
                g_loss = -1 * model.distiller_loss(metadata_disease_batch_drug, predicted_drug)
                g_loss.backward()

                # print("Gradients for encoder distiller layers:")
                # for name, param in model.encoder.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")

                # print("Gradients for drug classifier distiller layers:")
                # for name, param in model.drug_classifier.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")

                optimizer_distiller.step()


                model.drug_classifier.requires_grad_(True)

                # ----------------------------
                # Train encoder & classifier (c_loss)
                # ----------------------------
                optimizer.zero_grad()
                encoded_feature_batch = model.encoder(training_feature_batch)
                prediction_scores = model.disease_classifier(encoded_feature_batch).view(-1)
                c_loss = model.disease_classifier_loss(prediction_scores, metadata_batch_disease)
                c_loss.backward()

                # print("Gradients for encoder classifier layers:")
                # for name, param in model.encoder.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")

                # print("Gradients for disease classifier layers:")
                # for name, param in model.disease_classifier.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm()}")
                optimizer.step()

                # Store the batch losses
                epoch_r_loss += r_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_c_loss += c_loss.item()

                # Collect predictions and labels for epoch metrics
                pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float()
                epoch_train_preds.append(pred_tag.cpu())
                epoch_train_labels.append(metadata_batch_disease.cpu())

            # Compute average losses over the epoch
            epoch_r_loss /= num_batches
            epoch_g_loss /= num_batches
            epoch_c_loss /= num_batches

             # Store the losses
            r_losses.append(epoch_r_loss)
            g_losses.append(epoch_g_loss)
            c_losses.append(epoch_c_loss)

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
            scheduler.step(train_disease_acc)
            scheduler_distiller.step(epoch_g_loss)
            scheduler_classification_drug.step(epoch_r_loss)

            # Analyze distance correlation and learned features
            with torch.no_grad():
                feature0 = model.encoder(training_feature_ctrl)
                # predicted_ctrl_drug = model.drug_classifier(feature0)
                # predicted_ctrl_drug = torch.sigmoid(predicted_ctrl_drug)
                dc0 = dcor.u_distance_correlation_sqr(feature0.cpu().numpy(), metadata_ctrl_drug)
                mi_ctrl = mutual_info_classif(feature0.cpu().numpy(), metadata_ctrl_drug, discrete_features=False, random_state=42)
                # mi_ctrl = pearsonr(metadata_ctrl_drug, predicted_ctrl_drug.cpu().numpy())

                feature1 = model.encoder(training_feature_disease)
                # predicted_disease_drug = model.drug_classifier(feature1)
                # predicted_disease_drug = torch.sigmoid(predicted_disease_drug)
                # print(metadata_disease_drug.numpy().dtype)
                # print(predicted_disease_drug.cpu().numpy().dtype)
                dc1 = dcor.u_distance_correlation_sqr(feature1.cpu().numpy(), metadata_disease_drug)
                mi_disease = mutual_info_classif(feature1.cpu().numpy(), metadata_disease_drug, discrete_features=False, random_state=42)
                # mi_disease = pearsonr(metadata_disease_drug.numpy(), predicted_disease_drug.cpu().numpy())
            dcs0.append(dc0)
            dcs1.append(dc1)
            mis0.append(mi_ctrl)
            mis1.append(mi_disease)

            # Evaluate on validation data
            eval_accuracy, eval_auc, eval_f1 = evaluate(
                model, X_clr_df_val, val_metadata, batch_size, 'eval', device
            )
            val_disease_accs.append(eval_accuracy)
            val_disease_aucs.append(eval_auc)
            val_disease_f1s.append(eval_f1)

            # Early stopping check based on validation accuracy
            if eval_accuracy > best_eval_acc:
                best_eval_acc = eval_accuracy
                early_stop_patience = 0
            else:
                early_stop_patience += 1

            if early_stop_patience == early_stop_step:
                print("Early stopping triggered.")
                break

            # Print epoch statistics
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"r_loss: {epoch_r_loss:.4f}, g_loss: {epoch_g_loss:.4f}, c_loss: {epoch_c_loss:.4f}, "
                  f"train_acc: {train_disease_acc:.4f}, val_acc: {eval_accuracy:.4f}, "
                  f"dc0: {dc0:.4f}, dc1: {dc1:.4f}")

        # Save models
        torch.save(model.encoder.state_dict(), f'models/encoder_fold{fold}.pth')
        torch.save(model.disease_classifier.state_dict(), f'models/disease_classifier_fold{fold}.pth')
        print(f'Encoder saved to models/encoder_fold{fold}.pth.')
        print(f'Classifier saved to models/disease_classifier_fold{fold}.pth.')

        all_eval_accuracies.append(eval_accuracy)
        all_eval_aucs.append(eval_auc)
        all_eval_f1s.append(eval_f1)

        # Plotting losses and metrics
        plot_losses(
            r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1,
            train_disease_accs, train_disease_aucs, train_disease_f1s,
            val_disease_accs, val_disease_aucs, val_disease_f1s, fold
        )

    avg_eval_accuracy = np.mean(all_eval_accuracies)
    avg_eval_auc = np.mean(all_eval_aucs)
    save_eval_results(all_eval_accuracies, all_eval_aucs, all_eval_f1s)
    return avg_eval_accuracy, avg_eval_auc


def evaluate(model, relative_abundance, metadata, batch_size, t, device):
    """Evaluate the trained GAN model using MixedDataset."""
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Create the MixedDataset and DataLoader for evaluation
    eval_dataset = MixedDataset(relative_abundance, metadata, device)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for feature_batch, metadata_batch_disease in eval_loader:
            feature_batch = feature_batch.to(device)
            metadata_batch_disease = metadata_batch_disease.to(device)

            # Forward pass
            encoded_feature_batch = model.encoder(feature_batch)
            prediction_scores = model.disease_classifier(encoded_feature_batch).view(-1)
            pred_tag = (torch.sigmoid(prediction_scores) > 0.5).float()

            # Collect predictions and labels
            all_preds.append(pred_tag)
            all_labels.append(metadata_batch_disease)
            all_scores.append(prediction_scores)

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)

    # Compute loss
    c_loss = model.disease_classifier_loss(all_scores, all_labels)

    # Move data to CPU for metric computations
    all_preds_cpu = all_preds.cpu()
    all_labels_cpu = all_labels.cpu()
    all_scores_cpu = all_scores.cpu()

    # Compute evaluation metrics
    disease_acc = balanced_accuracy_score(all_labels_cpu, all_preds_cpu)
    auc = calculate_auc(all_labels_cpu.view(-1, 1), all_scores_cpu)
    f1 = f1_score(all_labels_cpu, all_preds_cpu)

    print(f"{t} result --> Accuracy: {disease_acc:.4f}, Loss: {c_loss.item():.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    return disease_acc, auc, f1


def calculate_auc(metadata_batch_disease, prediction_scores):
    """Calculate AUC."""
    if len(torch.unique(metadata_batch_disease)) > 1:
        auc = roc_auc_score(metadata_batch_disease.cpu(), torch.sigmoid(prediction_scores).detach().cpu())
        return auc
    else:
        print("Cannot compute ROC AUC as only one class is present.")
        return None

def plot_losses(r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, train_disease_accs, train_disease_aucs, train_disease_f1s, val_disease_accs, val_disease_aucs, val_disease_f1s, fold):
    """Plot training losses and save the figures."""
    plot_single_loss(g_losses, 'g_loss', 'green', f'confounder_free_drug_gloss_fold{fold}.png')
    plot_single_loss(r_losses, 'r_loss', 'red', f'confounder_free_drug_rloss_fold{fold}.png')
    plot_single_loss(c_losses, 'c_loss', 'blue', f'confounder_free_drug_closs_fold{fold}.png')
    plot_single_loss(dcs0, 'dc0', 'orange', f'confounder_free_drug_dc0_fold{fold}.png')
    plot_single_loss(dcs1, 'dc1', 'orange', f'confounder_free_drug_dc1_fold{fold}.png')
    plot_single_loss(mis0, 'mi0', 'purple', f'confounder_free_drug_mi0_fold{fold}.png')
    plot_single_loss(mis1, 'mi1', 'purple', f'confounder_free_drug_mi1_fold{fold}.png')
    plot_single_loss(train_disease_accs, 'train_disease_acc', 'red', f'confounder_free_drug_train_disease_acc_fold{fold}.png')
    plot_single_loss(train_disease_aucs, 'train_disease_auc', 'red', f'confounder_free_drug_train_disease_auc_fold{fold}.png')
    plot_single_loss(train_disease_f1s, 'train_disease_f1', 'red', f'confounder_free_drug_train_disease_f1_fold{fold}.png')
    plot_single_loss(val_disease_accs, 'val_disease_acc', 'red', f'confounder_free_drug_val_disease_acc_fold{fold}.png')
    plot_single_loss(val_disease_aucs, 'val_disease_auc', 'red', f'confounder_free_drug_val_disease_auc_fold{fold}.png')
    plot_single_loss(val_disease_f1s, 'val_disease_f1', 'red', f'confounder_free_drug_val_disease_f1_fold{fold}.png')
    
    

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