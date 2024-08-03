# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from keras.models import Sequential, Model
# from keras.layers import Activation, Dense, Dropout, Flatten, UpSampling3D, Input, ZeroPadding3D, Lambda, Reshape
# from keras.layers.normalization import BatchNormalization
# from keras.layers import Conv3D, MaxPooling3D
# from keras.losses import mse, binary_crossentropy
# from keras.utils import plot_model
# from keras.constraints import unit_norm, max_norm
# from keras import regularizers
# from keras import backend as K
# from keras.optimizers import Adam

# import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
# import nibabel as nib
import scipy as sp
# import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

import sys
import argparse
import os
import glob 

# import dcor

import torch
import torch.nn as nn
import torch.optim as optim


def dataset(relative_abundance, metadata, batch_size): 
	"""
    Initialize MicroDataset with data and metadata paths, species dictionary, and output directory.
    If embedding data file exists, load it; otherwise, process the data and create the embedding.
    """
	age_dict = {'Children Adolescents':0, 'Young Adult':1, 'Middle Aged':2, np.nan: np.nan}
	disease_dict = {'D006262':0, 'D047928':1}
	metadata['age_numeric'] = metadata['host_age'].map(age_dict)
	metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
	age_encoded = np.array([one_hot(idx, len(age_dict)) for idx in metadata['age_numeric']])
	age_encoded_df = pd.DataFrame(age_encoded, columns=[f'age_{i}' for i in range(len(age_dict))])
	metadata_encoded = pd.concat([metadata, age_encoded_df], axis=1)
	metadata= metadata_encoded

	ctrl_metadata = metadata[metadata['disease'] == 'D006262']
	run_ids = ctrl_metadata['run_id']
	ctrl_relative_abundance = relative_abundance[relative_abundance['Unnamed: 0'].isin(run_ids)]
		
	filtered_indexes = ctrl_metadata.index
	ctrl_idx_perm = np.random.permutation(filtered_indexes)
	ctrl_idx = ctrl_idx_perm[:int(batch_size)]
	training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx]
	training_feature_ctrl_batch.rename(columns={'Unnamed: 0': 'run_id'}, inplace=True)
	metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]
		
	# print(training_feature_ctrl_batch)
	# print(metadata_ctrl_batch)
		
	proportions = metadata['disease'].value_counts(normalize=True)
	num_samples_per_group = (proportions * batch_size).round().astype(int)
	metadata_feature_batch = metadata.groupby('disease').apply(lambda x: x.sample(n=num_samples_per_group[x.name])).reset_index(drop=True)
	training_feature_batch = relative_abundance[relative_abundance['Unnamed: 0'].isin(metadata_feature_batch['run_id'])]
	training_feature_batch = training_feature_batch.set_index('Unnamed: 0').reindex(metadata_feature_batch['run_id']).reset_index()
	training_feature_batch.rename(columns={'Unnamed: 0': 'run_id'}, inplace=True)

	training_feature_ctrl_batch = training_feature_ctrl_batch.drop(columns=['run_id'])
	training_feature_batch = training_feature_batch.drop(columns=['run_id'])

	training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
	metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch[['age_0', 'age_1', 'age_1']].values, dtype=torch.float32)
	training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
	metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32)

	return training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease

		

def test_dataset(relative_abundance, metadata, batch_size): 
	"""
    Initialize MicroDataset with data and metadata paths, species dictionary, and output directory.
    If embedding data file exists, load it; otherwise, process the data and create the embedding.
    """
	age_dict = {'Children Adolescents':0, 'Young Adult':1, 'Middle Aged':2, np.nan: np.nan}
	disease_dict = {'D006262':0, 'D047928':1}
	metadata['age_numeric'] = metadata['host_age'].map(age_dict)
	metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
	age_encoded = np.array([one_hot(idx, len(age_dict)) for idx in metadata['age_numeric']])
	age_encoded_df = pd.DataFrame(age_encoded, columns=[f'age_{i}' for i in range(len(age_dict))])
	metadata_encoded = pd.concat([metadata, age_encoded_df], axis=1)
	metadata= metadata_encoded

	ctrl_metadata = metadata[metadata['disease'] == 'D006262']
	run_ids = ctrl_metadata['run_id']
	ctrl_relative_abundance = relative_abundance[relative_abundance['Unnamed: 0'].isin(run_ids)]
		
	filtered_indexes = ctrl_metadata.index
	ctrl_idx_perm = np.random.permutation(filtered_indexes)
	ctrl_idx = ctrl_idx_perm[:int(batch_size)]
	training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx]
	training_feature_ctrl_batch.rename(columns={'Unnamed: 0': 'run_id'}, inplace=True)
	metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]
		
	# print(training_feature_ctrl_batch)
	# print(metadata_ctrl_batch)
		
	proportions = metadata['disease'].value_counts(normalize=True)
	num_samples_per_group = (proportions * batch_size).round().astype(int)
	metadata_feature_batch = metadata.groupby('disease').apply(lambda x: x.sample(n=num_samples_per_group[x.name])).reset_index(drop=True)
	training_feature_batch = relative_abundance[relative_abundance['Unnamed: 0'].isin(metadata_feature_batch['run_id'])]
	training_feature_batch = training_feature_batch.set_index('Unnamed: 0').reindex(metadata_feature_batch['run_id']).reset_index()
	training_feature_batch.rename(columns={'Unnamed: 0': 'run_id'}, inplace=True)

	training_feature_ctrl_batch = training_feature_ctrl_batch.drop(columns=['run_id'])
	training_feature_batch = training_feature_batch.drop(columns=['run_id'])

	training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
	metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch[['age_0', 'age_1', 'age_1']].values, dtype=torch.float32)
	training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
	metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32)

	return training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease

		

def one_hot(idx, num_classes):
	if np.isnan(idx):
		one_hot_enc = np.array([idx]*num_classes)
	else: 
		one_hot_enc = np.zeros(num_classes)
		one_hot_enc[int(idx)] = 1
	return one_hot_enc


def correlation_coefficient_loss(y_true, y_pred):

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true, dtype=np.float32)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred, dtype=np.float32)
    
    mx = np.mean(y_true)
    # print(mx)
    my = np.mean(y_pred)
    # print(my)
    xm = y_true - mx
    # print(y_true)
    ym = y_pred - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)) + 1e-5
    r = r_num / r_den
    print("r is:")
    print(r)
    r = np.clip(r, -1.0, 1.0)
    return torch.tensor(r ** 2, requires_grad=True)



class GAN():
    
    def __init__(self, input_dim):
        
        
        latent_dim = 128

        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

        self.disease_classifier =  nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.disease_classifier_loss = nn.BCELoss()

        self.lr = 0.0002
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), self.lr)

    def train(self, epochs, relative_abundance, metadata, batch_size=64, fold=0):

        for epoch in range(epochs):
            training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease = dataset(relative_abundance, metadata, batch_size)

            # Train encoder & classifier (actual classification task)
            self.optimizer.zero_grad()
            encoded_feature_batch = self.encoder(training_feature_batch)
            prediction_scores = self.disease_classifier(encoded_feature_batch)
            c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
            c_loss.backward()
            self.optimizer.step()
            

            print(f"Epoch {epoch + 1}/{epochs}, c_loss: {c_loss.item()}")
    def evaluate(self, relative_abundance, metadata, batch_size):
        training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease = test_dataset(relative_abundance, metadata, batch_size)
        encoded_feature_batch = self.encoder(training_feature_batch)
        prediction_scores = self.disease_classifier(encoded_feature_batch)
        print(prediction_scores)
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
        print("tessssst")
        print(f"c_loss: {c_loss.item()}")


if __name__ == "__main__":
      relative_abundance = pd.read_csv('Data/new_train_relative_abundance.csv')
      metadata = pd.read_csv('Data/new_train_metadata.csv')
      gan_cf = GAN(relative_abundance.shape[1]-1)
      gan_cf.train(epochs = 500, relative_abundance=relative_abundance, metadata=metadata, batch_size=64, fold=0)
      
      test_relative_abundance = pd.read_csv('Data/new_test_relative_abundance.csv')
      test_metadata = pd.read_csv('Data/new_test_metadata.csv')
      gan_cf.evaluate(relative_abundance=test_relative_abundance, metadata=test_metadata, batch_size=test_metadata.shape[0])