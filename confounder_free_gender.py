import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import optuna

def one_hot(idx, num_classes):
    """
    Creates a one-hot encoded vector for a given index.
    
    Args:
        idx (int or float): The index to be one-hot encoded. If NaN, returns a vector with NaN values.
        num_classes (int): The length of the one-hot encoded vector.
        
    Returns:
        np.ndarray: One-hot encoded vector of length `num_classes`.
    """
    if np.isnan(idx):
        return np.array([np.nan] * num_classes)
    one_hot_enc = np.zeros(num_classes)
    one_hot_enc[int(idx)] = 1
    return one_hot_enc

def preprocess_metadata(metadata):
    """
    Converts categorical metadata into numeric and one-hot encoded features.
    
    Args:
        metadata (pd.DataFrame): DataFrame containing metadata with 'gender' and 'disease' columns.
        
    Returns:
        pd.DataFrame: Original metadata DataFrame with additional numeric and one-hot encoded columns.
    """
   
    disease_dict = {'D006262': 0, 'D043183': 1}
    gender_dict = {'Male': 0, 'Female': 1}
    
    metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    metadata['gender_numeric'] = metadata['sex'].map(gender_dict)
    
    
    return metadata

def create_batch(relative_abundance, metadata, batch_size, is_test=False):
    """
    Creates a batch of data by sampling from the metadata and relative abundance data.
    
    Args:
        relative_abundance (pd.DataFrame): DataFrame with relative abundance values.
        metadata (pd.DataFrame): DataFrame with metadata.
        batch_size (int): Number of samples per batch.
        is_test (bool): If True, returns only test batch data without control data.
        
    Returns:
        tuple: (training_feature_batch, metadata_batch_disease) for testing, or
               (training_feature_ctrl_batch, metadata_ctrl_batch_gender, training_feature_batch, metadata_batch_disease) for training.
    """
    metadata = preprocess_metadata(metadata)
    proportions = metadata['disease'].value_counts(normalize=True)
    num_samples_per_group = (proportions * batch_size).round().astype(int)
    
    metadata_feature_batch = metadata.groupby('disease').apply(lambda x: x.sample(n=num_samples_per_group[x.name])).reset_index(drop=True)
    training_feature_batch = relative_abundance[relative_abundance['loaded_uid'].isin(metadata_feature_batch['uid'])]
    
    training_feature_batch = training_feature_batch.set_index('loaded_uid').reindex(metadata_feature_batch['uid']).reset_index()
    training_feature_batch.rename(columns={'loaded_uid': 'uid'}, inplace=True)
    training_feature_batch = training_feature_batch.drop(columns=['uid'])
    
    training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
    metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32)
    
    if is_test:
        return training_feature_batch, metadata_batch_disease
    
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    run_ids = ctrl_metadata['uid']
    ctrl_relative_abundance = relative_abundance[relative_abundance['loaded_uid'].isin(run_ids)]
    
    ctrl_idx = np.random.permutation(ctrl_metadata.index)[:batch_size]
    training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'loaded_uid': 'uid'}).drop(columns=['uid'])
    metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]
    
    training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
    metadata_ctrl_batch_gender = torch.tensor(metadata_ctrl_batch['gender_numeric'].values, dtype=torch.float32)
    
    return training_feature_ctrl_batch, metadata_ctrl_batch_gender, training_feature_batch, metadata_batch_disease

def correlation_coefficient_loss(y_true, y_pred):
    """
    Computes a custom loss function based on the correlation coefficient.
    
    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        
    Returns:
        torch.Tensor: Loss value computed as the square of the correlation coefficient.
    """
    y_true, y_pred = np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32)
    
    mx, my = np.mean(y_true), np.mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)) + 1e-5
    
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    
    return torch.tensor(r ** 2, requires_grad=True)

# adversial loss for mse
def inv_mse(y_true, y_pred):
    # Ensure y_true and y_pred are NumPy arrays
    y_true, y_pred = np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32)
    
    # Compute Mean Squared Error
    mse_value = np.sum(np.square(y_true - y_pred))
    
    # Return the negative of the MSE
    return torch.tensor(-mse_value, requires_grad=True)

class GAN:
    def __init__(self, input_dim, latent_dim=128, lr=0.0001, dropout_rate=0.3, pos_weight=2):
        """
        Initializes the GAN with an encoder, gender classifier, and disease classifier.
        
        Args:
            input_dim (int): Dimension of the input features.
        """

        # Build a naive feature encoder

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.gender_classification = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )
        self.gender_classification_loss = nn.BCEWithLogitsLoss()

        self.disease_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
        )

        # self.disease_classifier_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.disease_classifier_loss = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), lr)
        self.optimizer_distiller = optim.Adam(self.encoder.parameters(), lr)
        self.optimizer_classification_gender = optim.Adam(self.gender_classification.parameters(), lr)

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.scheduler_distiller = lr_scheduler.ReduceLROnPlateau(self.optimizer_distiller, mode='min', factor=0.5, patience=5)
        self.scheduler_classification_gender = lr_scheduler.ReduceLROnPlateau(self.optimizer_classification_gender, mode='min', factor=0.5, patience=5)

       
    def train(self, epochs, relative_abundance, metadata, batch_size=256):
        """
        Trains the GAN model for a specified number of epochs.
        
        Args:
            epochs (int): Number of training epochs.
            relative_abundance (pd.DataFrame): DataFrame with relative abundance values.
            metadata (pd.DataFrame): DataFrame with metadata.
            batch_size (int): Number of samples per batch.
        """
        best_acc = 0
        early_stop_step = 30
        early_stop_patience = 0
        for epoch in range(epochs):
            training_feature_ctrl_batch, metadata_ctrl_batch_gender, training_feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size)

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
            self.scheduler.step(disease_acc) # ReduceLROnPlateau

            if disease_acc>best_acc:
                best_acc = disease_acc
                early_stop_patience = 0
            else:
                early_stop_patience += 1
            if early_stop_patience == early_stop_step:
                break

            print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item()}, g_loss: {g_loss.item()}, c_loss: {c_loss.item()}, disease_acc: {disease_acc}")
            if epoch % 1 == 0:

                test_relative_abundance = pd.read_csv('GMrepo_data/test_relative_abundance_IBD.csv')
                test_metadata = pd.read_csv('GMrepo_data/test_metadata_IBD.csv')
                loaded_uid_test = test_relative_abundance['loaded_uid']  # Save the 'loaded_uid' column
                X_test = test_relative_abundance.drop(columns=['loaded_uid']).values  # Drop 'loaded_uid' for transformation

                # Adding a small pseudocount to avoid log(0)
                pseudocount = 1e-6
                X_test += pseudocount


                # Perform CLR transformation
                X_clr_test = clr_transformation(X_test)

                # Convert the CLR transformed data back to a DataFrame
                X_clr_df_test = pd.DataFrame(X_clr_test, columns=test_relative_abundance.columns[1:])  # Use the original column names except 'loaded_uid'

                # Add the 'loaded_uid' column back to the CLR transformed data
                X_clr_df_test['loaded_uid'] = loaded_uid_test

                # Optionally, reorder columns if you want 'loaded_uid' at the start
                X_clr_df_test = X_clr_df_test[['loaded_uid'] + list(X_clr_df_test.columns[:-1])]

                self.evaluate(relative_abundance=test_relative_abundance, metadata=test_metadata, batch_size=test_metadata.shape[0])

    def evaluate(self, relative_abundance, metadata, batch_size):
        """
        Evaluates the trained GAN model on test data.
        
        Args:
            relative_abundance (pd.DataFrame): DataFrame with relative abundance values.
            metadata (pd.DataFrame): DataFrame with metadata.
            batch_size (int): Number of samples for evaluation.
        """
        training_feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size, True)
        encoded_feature_batch = self.encoder(training_feature_batch)
        prediction_scores = self.disease_classifier(encoded_feature_batch)
        pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
        disease_acc = accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
        print(f"test result --> accuracy: {disease_acc}, c_loss: {c_loss.item()}")

if __name__ == "__main__":
    relative_abundance = pd.read_csv('GMrepo_data/train_relative_abundance_IBD_v5.csv')
    # Separate 'loaded_uid' and other columns
    loaded_uid = relative_abundance['loaded_uid']  # Save the 'loaded_uid' column
    X = relative_abundance.drop(columns=['loaded_uid']).values  # Drop 'loaded_uid' for transformation

    # Adding a small pseudocount to avoid log(0)
    pseudocount = 1e-6
    X += pseudocount

    # CLR Transformation function
    def clr_transformation(X):
        geometric_mean = np.exp(np.mean(np.log(X), axis=1))
        clr_data = np.log(X / geometric_mean[:, np.newaxis])
        return clr_data

    # Perform CLR transformation
    X_clr = clr_transformation(X)

    # Convert the CLR transformed data back to a DataFrame
    X_clr_df = pd.DataFrame(X_clr, columns=relative_abundance.columns[1:])  # Use the original column names except 'loaded_uid'

    # Add the 'loaded_uid' column back to the CLR transformed data
    X_clr_df['loaded_uid'] = loaded_uid

    # Optionally, reorder columns if you want 'loaded_uid' at the start
    X_clr_df = X_clr_df[['loaded_uid'] + list(X_clr_df.columns[:-1])]
   
    metadata = pd.read_csv('GMrepo_data/train_metadata_IBD_v5.csv')
    gan_cf = GAN(input_dim=relative_abundance.shape[1] - 1)
    gan_cf.train(epochs=1500, relative_abundance=X_clr_df, metadata=metadata, batch_size=64)
    
    test_relative_abundance = pd.read_csv('GMrepo_data/test_relative_abundance_IBD.csv')
    test_metadata = pd.read_csv('GMrepo_data/test_metadata_IBD.csv')
    loaded_uid_test = test_relative_abundance['loaded_uid']  # Save the 'loaded_uid' column
    X_test = test_relative_abundance.drop(columns=['loaded_uid']).values  # Drop 'loaded_uid' for transformation

    # Adding a small pseudocount to avoid log(0)
    pseudocount = 1e-6
    X_test += pseudocount


    # Perform CLR transformation
    X_clr_test = clr_transformation(X_test)

    # Convert the CLR transformed data back to a DataFrame
    X_clr_df_test = pd.DataFrame(X_clr_test, columns=test_relative_abundance.columns[1:])  # Use the original column names except 'loaded_uid'

    # Add the 'loaded_uid' column back to the CLR transformed data
    X_clr_df_test['loaded_uid'] = loaded_uid_test

    # Optionally, reorder columns if you want 'loaded_uid' at the start
    X_clr_df_test = X_clr_df_test[['loaded_uid'] + list(X_clr_df_test.columns[:-1])]
    gan_cf.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0])

# Objective function for Optuna
# def objective(trial):
#     relative_abundance = pd.read_csv('Data/new_train_relative_abundance.csv')
#     metadata = pd.read_csv('Data/new_train_metadata.csv')
#     latent_dim = trial.suggest_int('latent_dim', 64, 256)
#     lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
#     dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
#     pos_weight = trial.suggest_int('pos_weight', 1, 5)
    
#     gan_model = GAN(input_dim=relative_abundance.shape[1] - 1, latent_dim=latent_dim, lr=lr, dropout_rate=dropout_rate, pos_weight=pos_weight)
    
#     epochs = 500
    
#     gan_model.train(epochs, relative_abundance, metadata, batch_size=64)
    
#     test_relative_abundance = pd.read_csv('Data/new_test_relative_abundance.csv')
#     test_metadata = pd.read_csv('Data/new_test_metadata.csv')
    
#     accuracy = gan_model.evaluate(relative_abundance=test_relative_abundance, metadata=test_metadata, batch_size=test_metadata.shape[0])
    
#     return accuracy

# # Hyperparameter optimization with Optuna
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# # Print the best hyperparameters
# print("Best hyperparameters found: ", study.best_params)