import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


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
        metadata (pd.DataFrame): DataFrame containing metadata with 'host_age' and 'disease' columns.
        
    Returns:
        pd.DataFrame: Original metadata DataFrame with additional numeric and one-hot encoded columns.
    """
    age_dict = {'Children Adolescents': 0, 'Young Adult': 1, 'Middle Aged': 2, np.nan: np.nan}
    disease_dict = {'D006262': 0, 'D047928': 1}
    
    metadata['age_numeric'] = metadata['host_age'].map(age_dict)
    metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    
    age_encoded = np.array([one_hot(idx, len(age_dict)) for idx in metadata['age_numeric']])
    age_encoded_df = pd.DataFrame(age_encoded, columns=[f'age_{i}' for i in range(len(age_dict))])
    
    return pd.concat([metadata, age_encoded_df], axis=1)

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
               (training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease) for training.
    """
    metadata = preprocess_metadata(metadata)
    proportions = metadata['disease'].value_counts(normalize=True)
    num_samples_per_group = (proportions * batch_size).round().astype(int)
    
    metadata_feature_batch = metadata.groupby('disease').apply(lambda x: x.sample(n=num_samples_per_group[x.name])).reset_index(drop=True)
    training_feature_batch = relative_abundance[relative_abundance['Unnamed: 0'].isin(metadata_feature_batch['run_id'])]
    
    training_feature_batch = training_feature_batch.set_index('Unnamed: 0').reindex(metadata_feature_batch['run_id']).reset_index()
    training_feature_batch.rename(columns={'Unnamed: 0': 'run_id'}, inplace=True)
    training_feature_batch = training_feature_batch.drop(columns=['run_id'])
    
    training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
    metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32)
    
    if is_test:
        return training_feature_batch, metadata_batch_disease
    
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    run_ids = ctrl_metadata['run_id']
    ctrl_relative_abundance = relative_abundance[relative_abundance['Unnamed: 0'].isin(run_ids)]
    
    ctrl_idx = np.random.permutation(ctrl_metadata.index)[:batch_size]
    training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'Unnamed: 0': 'run_id'}).drop(columns=['run_id'])
    metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]
    
    training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
    metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch[['age_0', 'age_1', 'age_2']].values, dtype=torch.float32)
    
    return training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease

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

class GAN:
    def __init__(self, input_dim):
        """
        Initializes the GAN with an encoder, age classifier, and disease classifier.
        
        Args:
            input_dim (int): Dimension of the input features.
        """
        latent_dim = 128
        
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
            # nn.Sigmoid()
        )

        self.disease_classifier_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
        # self.disease_classifier_loss = nn.BCELoss(pos_weight=3)

        self.lr = 0.001
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), self.lr)

         # Initialize the scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

       
    def train(self, epochs, relative_abundance, metadata, batch_size=64):
        """
        Trains the GAN model for a specified number of epochs.
        
        Args:
            epochs (int): Number of training epochs.
            relative_abundance (pd.DataFrame): DataFrame with relative abundance values.
            metadata (pd.DataFrame): DataFrame with metadata.
            batch_size (int): Number of samples per batch.
        """
        best_acc = 0
        early_stop_step = 20
        for epoch in range(epochs):
            training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size)


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
            # else:
            #     early_stop_patience += 1
            # if early_stop_patience == early_stop_step:
            #     break

            print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item()}, g_loss: {g_loss.item()}, c_loss: {c_loss.item()}")
            if epoch % 100 == 0:

                test_relative_abundance = pd.read_csv('Data/new_test_relative_abundance.csv')
                test_metadata = pd.read_csv('Data/new_test_metadata.csv')
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
    relative_abundance = pd.read_csv('Data/new_train_relative_abundance.csv')
    metadata = pd.read_csv('Data/new_train_metadata.csv')
    gan_cf = GAN(input_dim=relative_abundance.shape[1] - 1)
    gan_cf.train(epochs=1500, relative_abundance=relative_abundance, metadata=metadata, batch_size=64)
    
    test_relative_abundance = pd.read_csv('Data/new_test_relative_abundance.csv')
    test_metadata = pd.read_csv('Data/new_test_metadata.csv')
    gan_cf.evaluate(relative_abundance=test_relative_abundance, metadata=test_metadata, batch_size=test_metadata.shape[0])
