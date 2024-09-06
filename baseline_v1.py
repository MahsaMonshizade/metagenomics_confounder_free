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
        metadata (pd.DataFrame): DataFrame containing metadata with 'host_age' and 'disease' columns.
        
    Returns:
        pd.DataFrame: Original metadata DataFrame with additional numeric and one-hot encoded columns.
    """

    disease_dict = {'D006262': 0, 'D043183': 1}
    metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    
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
               (training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease) for training.
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
    # print(ctrl_idx)
    training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'loaded_uid': 'uid'}).drop(columns=['uid'])
    metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]
    
    training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
    metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch['host_age'].values, dtype=torch.float32)
    
    return training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease


class GAN:
    def __init__(self, input_dim, latent_dim=128, lr=0.001, dropout_rate=0.3, pos_weight=2):
        """
        Initializes the GAN with an encoder, age classifier, and disease classifier.
        
        Args:
            input_dim (int): Dimension of the input features.
        """

        # Build a naive feature encoder

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            # nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        self.disease_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # # Calculate class weights
        # total_samples = 19 + 45
        # weight_for_0 = total_samples / (2 * 45)
        # weight_for_1 = total_samples / (2 * 19)

        # # Convert the weight for the positive class (1) into a tensor
        # pos_weight = torch.tensor([weight_for_1 / weight_for_0])

        # self.disease_classifier_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.disease_classifier_loss = nn.BCELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), lr)

       
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
        early_stop_patience = 0
        for epoch in range(epochs):
            training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size)

            self.optimizer.zero_grad()
            encoded_feature_batch = self.encoder(training_feature_batch)
            prediction_scores = self.disease_classifier(encoded_feature_batch)

            c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
            c_loss.backward()
            pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
            disease_acc = accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
            self.optimizer.step()

            if disease_acc>best_acc:
                best_acc = disease_acc
            

            print(f"Epoch {epoch + 1}/{epochs}, c_loss: {c_loss.item()}, disease_acc: {disease_acc}")
            if epoch % 100 == 0:

                test_relative_abundance = pd.read_csv('GMrepo_data/test_relative_abundance_IBD.csv')
                test_metadata = pd.read_csv('GMrepo_data/test_metadata_IBD.csv')
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
    relative_abundance = pd.read_csv('GMrepo_data/train_relative_abundance_IBD.csv')
    metadata = pd.read_csv('GMrepo_data/train_metadata_IBD.csv')
    gan_cf = GAN(input_dim=relative_abundance.shape[1] - 1)
    gan_cf.train(epochs=1500, relative_abundance=relative_abundance, metadata=metadata, batch_size=64)
    
    test_relative_abundance = pd.read_csv('GMrepo_data/test_relative_abundance_IBD.csv')
    test_metadata = pd.read_csv('GMrepo_data/test_metadata_IBD.csv')
    gan_cf.evaluate(relative_abundance=test_relative_abundance, metadata=test_metadata, batch_size=test_metadata.shape[0])

