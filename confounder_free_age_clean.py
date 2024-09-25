import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init, functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import mutual_info_regression
import optuna
import matplotlib.pyplot as plt
import json

import dcor


def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clr_transformation(X):
    """Perform CLR (Centered Log-Ratio) transformation on the data."""
    geometric_mean = np.exp(np.mean(np.log(X), axis=1))
    return np.log(X / geometric_mean[:, np.newaxis])


def load_and_transform_data(file_path):
    """Load data from CSV, apply CLR transformation, and return transformed DataFrame with 'loaded_uid'."""
    data = pd.read_csv(file_path)
    loaded_uid = data['loaded_uid']
    X = data.drop(columns=['loaded_uid']).values
    X += 1e-6  # Adding a small pseudocount to avoid log(0)
    X_clr = clr_transformation(X)
    X_clr_df = pd.DataFrame(X_clr, columns=data.columns[1:])
    X_clr_df['loaded_uid'] = loaded_uid
    return X_clr_df[['loaded_uid'] + list(X_clr_df.columns[:-1])]


def preprocess_metadata(metadata):
    """Converts categorical metadata into numeric and one-hot encoded features."""
    disease_dict = {'D006262': 0, 'D003093': 1}
    metadata.loc[:, 'disease_numeric'] = metadata['disease'].map(disease_dict)
    # metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    return metadata


def create_batch(relative_abundance, metadata, batch_size, is_test=False):
    """Creates a batch of data by sampling from the metadata and relative abundance data."""
    metadata = preprocess_metadata(metadata)
    proportions = metadata['disease'].value_counts(normalize=True)
    num_samples_per_group = (proportions * batch_size).round().astype(int)

    # Sample metadata
    metadata_feature_batch = metadata.groupby('disease').apply(
        lambda x: x.sample(n=num_samples_per_group[x.name])
    ).reset_index(drop=True)

    # Sample relative abundance
    training_feature_batch = relative_abundance[relative_abundance['loaded_uid'].isin(metadata_feature_batch['uid'])]
    training_feature_batch = training_feature_batch.set_index('loaded_uid').reindex(metadata_feature_batch['uid']).reset_index()
    training_feature_batch.rename(columns={'loaded_uid': 'uid'}, inplace=True)
    training_feature_batch = training_feature_batch.drop(columns=['uid'])

    # Convert to tensors
    training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
    metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32)

    if is_test:
        return training_feature_batch, metadata_batch_disease

    # Control batch
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    run_ids = ctrl_metadata['uid']
    ctrl_relative_abundance = relative_abundance[relative_abundance['loaded_uid'].isin(run_ids)]

    ctrl_idx = np.random.permutation(ctrl_metadata.index)[:batch_size]
    training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'loaded_uid': 'uid'}).drop(columns=['uid'])
    metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]

    training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
    metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch['host_age_zscore'].values, dtype=torch.float32)

    return (training_feature_ctrl_batch, metadata_ctrl_batch_age, 
            training_feature_batch, metadata_batch_disease)


def dcor_calculation_data(relative_abundance, metadata):
    """C"""
    metadata = preprocess_metadata(metadata)
    # Control
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    ctrl_run_ids = ctrl_metadata['uid']
    ctrl_relative_abundance = relative_abundance[relative_abundance['loaded_uid'].isin(ctrl_run_ids)]
    ctrl_idx = ctrl_metadata.index.to_list()
    training_feature_ctrl = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'loaded_uid': 'uid'}).drop(columns=['uid'])
    metadata_ctrl = ctrl_metadata.loc[ctrl_idx]
    

    training_feature_ctrl = torch.tensor(training_feature_ctrl.values, dtype=torch.float32)
    metadata_ctrl_age = torch.tensor(metadata_ctrl['host_age_zscore'].values, dtype=torch.float32)
    disease_metadata = metadata[metadata['disease'] == 'D003093']
    disease_run_ids = disease_metadata['uid']
    disease_relative_abundance = relative_abundance[relative_abundance['loaded_uid'].isin(disease_run_ids)]
    disease_idx = disease_metadata.index.to_list()
    training_feature_disease = disease_relative_abundance.loc[disease_idx].rename(columns={'loaded_uid': 'uid'}).drop(columns=['uid'])
    metadata_disease = disease_metadata.loc[disease_idx]

    training_feature_disease = torch.tensor(training_feature_disease.values, dtype=torch.float32)
    metadata_disease_age = torch.tensor(metadata_disease['host_age_zscore'].values, dtype=torch.float32)
    return (training_feature_ctrl, metadata_ctrl_age, 
            training_feature_disease, metadata_disease_age)


class CorrelationCoefficientLoss(nn.Module):
    def __init__(self):
        super(CorrelationCoefficientLoss, self).__init__()

    def forward(self, y_true, y_pred):
        mean_x = torch.mean(y_true)
        mean_y = torch.mean(y_pred)
        covariance = torch.mean((y_true - mean_x) * (y_pred - mean_y))
        std_x = torch.std(y_true)
        std_y = torch.std(y_pred)
        eps = 1e-5
        return (covariance / (std_x * std_y) + eps) ** 2


class InvCorrelationCoefficientLoss(nn.Module):
    def __init__(self):
        super(InvCorrelationCoefficientLoss, self).__init__()

    def forward(self, y_true, y_pred):
        mean_x = torch.mean(y_true)
        mean_y = torch.mean(y_pred)
        covariance = torch.mean((y_true - mean_x) * (y_pred - mean_y))
        std_x = torch.std(y_true)
        std_y = torch.std(y_pred)
        eps = 1e-5
        return 1 - (covariance / (std_x * std_y) + eps) ** 2


class GAN(nn.Module):
    def __init__(self, input_dim, latent_dim=256, lr_r=0.001, lr_g=0.001, lr_c=0.005, activation_fn=nn.SiLU, num_layers=3):
        """Initializes the GAN with an encoder, age regressor, and disease classifier."""
        super(GAN, self).__init__()

        self.encoder = self._build_encoder(input_dim, latent_dim, num_layers, activation_fn)
        self.age_regressor = self._build_regressor(latent_dim, activation_fn, num_layers)
        self.disease_classifier = self._build_classifier(latent_dim, activation_fn, num_layers)

        self.age_regression_loss = InvCorrelationCoefficientLoss()
        # self.age_regression_loss = nn.MSELoss()
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
        """Builds the encoder network."""
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, current_dim // 2))
            layers.append(nn.BatchNorm1d(current_dim // 2))
            layers.append(activation_fn())
            current_dim //= 2
        layers.append(nn.Linear(current_dim, latent_dim))
        layers.append(nn.BatchNorm1d(latent_dim))
        layers.append(activation_fn())
        return nn.Sequential(*layers)

    def _build_regressor(self, latent_dim, activation_fn, num_layers):
        """Builds the age regressor."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, current_dim // 2))
            layers.append(nn.BatchNorm1d(current_dim // 2))
            layers.append(activation_fn())
            current_dim //= 2
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, activation_fn, num_layers):
        """Builds the disease classifier."""
        layers = []
        current_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, current_dim // 2))
            layers.append(nn.BatchNorm1d(current_dim // 2))
            layers.append(activation_fn())
            current_dim //= 2
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        """Initialize weights using Kaiming initialization for layers with ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def train(self, epochs, relative_abundance, metadata, batch_size=64):

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_eval_accuracies = []
        all_eval_aucs = []

        for fold, (train_index, val_index) in enumerate(kf.split(relative_abundance)):

            print(f"Fold {fold + 1}/{5}")
            X_clr_df_train, X_clr_df_val = relative_abundance.iloc[train_index], relative_abundance.iloc[val_index]
            train_metadata, val_metadata = metadata.iloc[train_index], metadata.iloc[val_index]

            best_loss = float('inf')
            early_stop_step = 10
            early_stop_patience = 0
        
            training_feature_ctrl, metadata_ctrl_age, training_feature_disease, metadata_disease_age=dcor_calculation_data(X_clr_df_train, train_metadata)
            
            # Lists to store losses
            r_losses, g_losses, c_losses = [], [], []
            features0, features1, dcs0, dcs1, mis0, mis1 = [], [], [], [], [], []

            for epoch in range(epochs):
                # Create batches
                training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease = create_batch(
                    X_clr_df_train, train_metadata, batch_size
                )

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
                best_loss, early_stop_patience = self._check_early_stopping(c_loss, best_loss, early_stop_patience, early_stop_step)
                if early_stop_patience == early_stop_step:
                    break
                
                # analyze distance correlation and learned features for CTRL group
                feature0 = self.encoder(training_feature_ctrl)  
                dc0 = dcor.u_distance_correlation_sqr(feature0.detach().numpy(), metadata_ctrl_age)
                # Calculate mutual information for control group
                mi_ctrl = mutual_info_regression(feature0.detach().numpy(), metadata_ctrl_age)
                # analyze distance correlation and learned features for Disease group
                feature1 = self.encoder(training_feature_disease)  
                dc1 = dcor.u_distance_correlation_sqr(feature1.detach().numpy(), metadata_disease_age)
                # Calculate mutual information for disease group
                mi_disease = mutual_info_regression(feature1.detach().numpy(), metadata_disease_age)
                
                features0.append(feature0)
                features1.append(feature1)
                dcs0.append(dc0)
                dcs1.append(dc1)
                mis0.append(mi_ctrl)
                mis1.append(mi_disease)

                print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item()}, g_loss: {g_loss.item()}, c_loss: {c_loss.item()}, disease_acc: {disease_acc}, dc0: {dc0}, dc1: {dc1}")

                # Evaluate
                eval_accuracy, eval_auc = self.evaluate(X_clr_df_val, val_metadata, val_metadata.shape[0], 'eval')
                _, _ = self.evaluate(X_clr_df_train, train_metadata, train_metadata.shape[0], 'train')
            torch.save(self.encoder.state_dict(), f'models/encoder_fold{fold}.pth')
            torch.save(self.disease_classifier.state_dict(), f'models/disease_classifier_fold{fold}.pth')
            print(f'Encoder saved to .')
            print(f'Classifier saved to .')
            all_eval_accuracies.append(eval_accuracy)    
            all_eval_aucs.append(eval_auc)
            self.plot_losses(r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, fold)
        avg_eval_accuracy = np.mean(all_eval_accuracies)
        avg_eval_auc = np.mean(all_eval_aucs)
        self.save_eval_results(all_eval_accuracies, all_eval_accuracies)
        return avg_eval_accuracy, avg_eval_auc
    
    def save_eval_results(self, accuracies, aucs, filename='evaluation_results.json'):
        """Save evaluation accuracies and AUCs to a JSON file."""
        results = {
            'accuracies': accuracies,
            'aucs': aucs
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation results saved to {filename}.")

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
        prediction_scores = self.disease_classifier(encoded_feature_batch)
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
        c_loss.backward()
        pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
        disease_acc = balanced_accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
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

        if early_stop_patience == early_stop_step:
            print("Early stopping triggered.")
            return best_loss, early_stop_patience
        return best_loss, early_stop_patience

    def plot_losses(self, r_losses, g_losses, c_losses, dcs0, dcs1, mis0, mis1, fold):
        """Plots r_loss, g_loss, and c_loss over epochs."""
        self._plot_single_loss(g_losses, f'g_loss', 'green', 'confounder_free_age_gloss_fold{fold}.png')
        self._plot_single_loss(r_losses, f'r_loss', 'red', 'confounder_free_age_rloss_fold{fold}.png')
        self._plot_single_loss(c_losses, 'c_loss', 'blue', 'confounder_free_age_closs_fold{fold}.png')
        self._plot_single_loss(dcs0, f'dc0', 'orange', 'confounder_free_age_dc0_fold{fold}png')
        self._plot_single_loss(dcs1, f'dc1', 'orange', 'confounder_free_age_dc1_fold{fold}.png')
        self._plot_single_loss(mis0, f'mi0', 'purple', 'confounder_free_age_mi0_fold{fold}.png')
        self._plot_single_loss(mis1, f'mi1', 'purple', 'confounder_free_age_mi1_fold{fold}.png')

    def _plot_single_loss(self, losses, label, color, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label=label, color=color)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Losses Over Epochs ({label})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{filename}')
        plt.close()

    def evaluate(self, relative_abundance, metadata, batch_size, t):
        """Evaluates the trained GAN model on test data."""
        feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size, True)
        encoded_feature_batch = self.encoder(feature_batch)
        prediction_scores = self.disease_classifier(encoded_feature_batch)

        # Convert probabilities to predicted classes
        pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
        disease_acc = balanced_accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))

        auc = self._calculate_auc(metadata_batch_disease, prediction_scores, t)
        f1 = f1_score(metadata_batch_disease.view(-1, 1), pred_tag)

        print(f"{t} result --> Accuracy: {disease_acc}, Loss: {c_loss.item()}, AUC: {auc}, F1: {f1}")
        return disease_acc, auc

    def _calculate_auc(self, metadata_batch_disease, prediction_scores, t):
        """Calculate AUC."""
        if len(np.unique(metadata_batch_disease)) > 1:
            auc = roc_auc_score(metadata_batch_disease.view(-1, 1), prediction_scores.detach().numpy())
            print(f"{t} AUC: {auc}")
            return auc
        else:
            print("Cannot compute ROC AUC as only one class is present.")
            return None

if __name__ == "__main__":
    set_seed(42)

    # Load and transform training data
    file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_train.csv'
    metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_train.csv'
    X_clr_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # Initialize and train GAN
    gan = GAN(input_dim=X_clr_df.shape[1] - 1)
    gan.initialize_weights()
    gan.train(epochs=1500, relative_abundance=X_clr_df, metadata=metadata, batch_size=64)

    # # Load and transform test data
    # test_file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_test.csv'
    # test_metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_test.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)

    # # Evaluate GAN on test data
    # gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test')


# def objective(trial):
#     # Define hyperparameters to be optimized
#     latent_dim = trial.suggest_categorical('latent_dim', [64, 128, 256])
    
#     lr_r = trial.suggest_categorical('lr_r', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
#     lr_g = trial.suggest_categorical('lr_g', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
#     lr_c = trial.suggest_categorical('lr_c', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    
#     activation_fn = trial.suggest_categorical('activation_fn', [nn.ReLU, nn.Tanh, nn.SiLU, nn.SELU, nn.LeakyReLU])  # Added SiLU
#     num_layers = trial.suggest_int('num_layers', 1, 5)  # Assuming you want to tune between 1 and 5 layers

#     # Initialize the model with the suggested parameters
#     gan = GAN(input_dim=X_clr_df.shape[1] - 1, latent_dim=latent_dim, lr_r=lr_r, lr_g=lr_g, lr_c=lr_c, activation_fn=activation_fn, num_layers=num_layers)
#     gan.initialize_weights()
    
#     # Train the model
#     eval_accuracy, eval_auc = gan.train(epochs=1500, relative_abundance=X_clr_df, metadata=metadata, batch_size=64)
    
#     # Objective to minimize (negative AUC for maximization)
#     return eval_auc

# if __name__ == "__main__":
#     set_seed(42)

#     # Load and transform training data
#     file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_train.csv'
#     metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_train.csv'
#     X_clr_df = load_and_transform_data(file_path)
#     metadata = pd.read_csv(metadata_file_path)

#     # Create an Optuna study and optimize it
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=50)

#     # Save the best trial to a file
#     best_trial = study.best_trial
#     # Convert activation function type to string for JSON serialization
#     best_trial_params = best_trial.params.copy()
#     best_trial_params['activation_fn'] = str(best_trial_params['activation_fn'].__name__)

#     with open("best_trial.json", "w") as f:
#         json.dump(best_trial_params, f, indent=4)

#     print(f"Best trial saved: {best_trial.params}")



   