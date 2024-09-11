import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import json

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
    disease_dict = {'D006262': 0, 'D043183': 1}
    gender_dict = {'Male': 0, 'Female': 1}
    metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    metadata['gender_numeric'] = metadata['sex'].map(gender_dict)
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
    metadata_ctrl_batch_gender = torch.tensor(metadata_ctrl_batch['gender_numeric'].values, dtype=torch.float32)

    return (training_feature_ctrl_batch, metadata_ctrl_batch_gender, 
            training_feature_batch, metadata_batch_disease)

def correlation_coefficient_loss(y_true, y_pred):
    """Computes a custom loss function based on the correlation coefficient."""
    y_true, y_pred = np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32)
    mx, my = np.mean(y_true), np.mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)) + 1e-5
    r = np.clip(r_num / r_den, -1.0, 1.0)
    return torch.tensor(r ** 2, requires_grad=True)

class GAN:
    def __init__(self, input_dim, trial, latent_dim=128, dropout_rate=0.3):
        """Initializes the GAN with encoder and classifiers, and uses Optuna to tune layers and learning rates."""
        # Define hyperparameters to tune
        n_layers_encoder = trial.suggest_int("n_layers_encoder", 2, 4)
        n_units_encoder = [trial.suggest_int(f"n_units_encoder_l{i}", 128, 1024) for i in range(n_layers_encoder)]
        n_layers_classifier = trial.suggest_int("n_layers_classifier", 2, 4)
        n_units_classifier = [trial.suggest_int(f"n_units_classifier_l{i}", 16, 128) for i in range(n_layers_classifier)]

        # Learning rates
        lr_encoder = trial.suggest_loguniform("lr_encoder", 1e-5, 1e-1)
        lr_classifier = trial.suggest_loguniform("lr_classifier", 1e-5, 1e-1)

        # Encoder
        encoder_layers = []
        for i in range(n_layers_encoder):
            encoder_layers.append(nn.Linear(input_dim if i == 0 else n_units_encoder[i-1], n_units_encoder[i]))
            encoder_layers.append(nn.BatchNorm1d(n_units_encoder[i]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
        encoder_layers.append(nn.Linear(n_units_encoder[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Gender Classifier
        gender_layers = []
        for i in range(n_layers_classifier):
            gender_layers.append(nn.Linear(latent_dim if i == 0 else n_units_classifier[i-1], n_units_classifier[i]))
            gender_layers.append(nn.BatchNorm1d(n_units_classifier[i]))
            gender_layers.append(nn.ReLU())
            gender_layers.append(nn.Dropout(dropout_rate))
        gender_layers.append(nn.Linear(n_units_classifier[-1], 1))
        self.gender_classification = nn.Sequential(*gender_layers)

        # Disease Classifier
        disease_layers = []
        for i in range(n_layers_classifier):
            disease_layers.append(nn.Linear(latent_dim if i == 0 else n_units_classifier[i-1], n_units_classifier[i]))
            disease_layers.append(nn.BatchNorm1d(n_units_classifier[i]))
            disease_layers.append(nn.ReLU())
            disease_layers.append(nn.Dropout(dropout_rate))
        disease_layers.append(nn.Linear(n_units_classifier[-1], 1))
        self.disease_classifier = nn.Sequential(*disease_layers)

        self.gender_classification_loss = nn.BCEWithLogitsLoss()
        self.disease_classifier_loss = nn.BCEWithLogitsLoss()

        # Optimizers
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), lr=lr_encoder)
        self.optimizer_distiller = optim.Adam(self.encoder.parameters(), lr=lr_encoder)
        self.optimizer_classification_gender = optim.Adam(self.gender_classification.parameters(), lr=lr_classifier)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.scheduler_distiller = lr_scheduler.ReduceLROnPlateau(self.optimizer_distiller, mode='min', factor=0.5, patience=5)
        self.scheduler_classification_gender = lr_scheduler.ReduceLROnPlateau(self.optimizer_classification_gender, mode='min', factor=0.5, patience=5)

    def train(self, epochs, relative_abundance, metadata, batch_size=64):
        """Trains the GAN model for a specified number of epochs."""
        best_acc = 0
        early_stop_step = 30
        early_stop_patience = 0
        
        X_clr_df_train, X_clr_df_val, train_metadata, val_metadata = train_test_split(relative_abundance, metadata, test_size=0.2, random_state=42)

        for epoch in range(epochs):
            # Create batches
            training_feature_ctrl_batch, metadata_ctrl_batch_gender, training_feature_batch, metadata_batch_disease = create_batch(
                X_clr_df_train, train_metadata, batch_size
            )

            # Train gender classifier
            self.optimizer_classification_gender.zero_grad()
            for param in self.encoder.parameters():
                param.requires_grad = False
            encoded_feature_ctrl_batch = self.encoder(training_feature_ctrl_batch)
            gender_prediction = self.gender_classification(encoded_feature_ctrl_batch)
            r_loss = self.gender_classification_loss(gender_prediction, metadata_ctrl_batch_gender.view(-1, 1))
            r_loss.backward()
            self.optimizer_classification_gender.step()
            for param in self.encoder.parameters():
                param.requires_grad = True

            # Train distiller
            self.optimizer_distiller.zero_grad()
            for param in self.gender_classification.parameters():
                param.requires_grad = False
            encoder_features = self.encoder(training_feature_ctrl_batch)
            predicted_gender = self.gender_classification(encoder_features)
            g_loss = correlation_coefficient_loss(metadata_ctrl_batch_gender, predicted_gender.detach())
            g_loss.backward()
            self.optimizer_distiller.step()
            for param in self.gender_classification.parameters():
                param.requires_grad = True

            # Train encoder & classifier
            self.optimizer.zero_grad()
            encoded_feature_batch = self.encoder(training_feature_batch)
            prediction_scores = self.disease_classifier(encoded_feature_batch)
            c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
            c_loss.backward()
            pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
            disease_acc = accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
            self.optimizer.step()
            self.scheduler.step(disease_acc)

            if disease_acc > best_acc:
                best_acc = disease_acc
                early_stop_patience = 0
            else:
                early_stop_patience += 1
            if early_stop_patience == early_stop_step:
                break

            print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item()}, g_loss: {g_loss.item()}, c_loss: {c_loss.item()}, disease_acc: {disease_acc}")

            self.evaluate(relative_abundance=X_clr_df_val, metadata=val_metadata, batch_size=val_metadata.shape[0], t='eval')
            self.evaluate(relative_abundance=X_clr_df_train, metadata=train_metadata, batch_size=train_metadata.shape[0], t='train')

    def evaluate(self, relative_abundance, metadata, batch_size, t):
        """Evaluates the trained GAN model on test data."""
        # Create batches
        feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size, is_test=True)

        # Get encoded features
        encoded_feature_batch = self.encoder(feature_batch)

        # Get prediction scores (probabilities)
        prediction_scores = self.disease_classifier(encoded_feature_batch)

        # Convert probabilities to predicted classes
        pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]

        # Calculate accuracy
        disease_acc = accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)

        # Calculate classifier loss
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))

        # Calculate AUC
        auc = roc_auc_score(metadata_batch_disease.view(-1, 1), prediction_scores.detach().numpy())

        # Print results
        print(f"{t} result --> Accuracy: {disease_acc}, Loss: {c_loss.item()}, AUC: {auc}")

        return disease_acc, auc

def objective(trial):
    # Load and transform data
    file_path = 'GMrepo_data/train_relative_abundance_IBD_balanced.csv'
    metadata_file_path = 'GMrepo_data/train_metadata_IBD_balanced.csv'
    X_clr_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # Split into train and validation sets
    X_clr_df_train, X_clr_df_val, train_metadata, val_metadata = train_test_split(X_clr_df, metadata, test_size=0.2, random_state=42)

    # Initialize and train GAN
    gan = GAN(input_dim=X_clr_df.shape[1] - 1, trial=trial)
    gan.train(epochs=50, relative_abundance=X_clr_df_train, metadata=train_metadata, batch_size=64)  # Train on 50 epochs for tuning speed

    # Validation performance (accuracy/AUC)
    disease_acc, auc = gan.evaluate(relative_abundance=X_clr_df_val, metadata=val_metadata, batch_size=val_metadata.shape[0], t='eval')

    # Optuna tries to minimize the objective, so we return the negative of the accuracy or AUC
    return -disease_acc  # or use auc if that's your primary metric

if __name__ == "__main__":
    set_seed(42)

    # Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Adjust number of trials as needed

    # Display and Save the Best Trial
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    # Save best trial's parameters to a JSON file
    best_params = {
        'value': trial.value,
        'params': trial.params
    }

    # Save to a file
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best hyperparameters saved to 'best_hyperparameters.json'.")
