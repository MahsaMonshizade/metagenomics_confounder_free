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

import dcor

import torch
import torch.nn as nn
import torch.optim as optim


def dataset(relative_abundance, metadata, batch_size): 
	"""
    Initialize MicroDataset with data and metadata paths, species dictionary, and output directory.
    If embedding data file exists, load it; otherwise, process the data and create the embedding.
    """
	gender_dict = {'Female': 0, 'Male': 1, np.nan: np.nan}
	disease_dict = {'D003550':0, 'D016585':1, 'D006262':2, 'D029424':3}
	metadata['sex_numeric'] = metadata['sex'].map(gender_dict)
	print(metadata['sex_numeric'])
	metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
	disease_encoded = np.array([one_hot(idx, len(disease_dict)) for idx in metadata['disease_numeric']])
	disease_encoded_df = pd.DataFrame(disease_encoded, columns=[f'disease_{i}' for i in range(len(disease_dict))])
	metadata_encoded = pd.concat([metadata, disease_encoded_df], axis=1)
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
	# print(training_feature_batch)
	# print(metadata_feature_batch)
	
	return training_feature_ctrl_batch, metadata_ctrl_batch, training_feature_batch, metadata_feature_batch

		
def one_hot(idx, num_classes):
		
		one_hot_enc = np.zeros(num_classes)
		one_hot_enc[idx] = 1
		return one_hot_enc


def correlation_coefficient_loss(y_true, y_pred):

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true, dtype=np.float32)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred, dtype=np.float32)
    
    mx = np.mean(y_true)
    my = np.mean(y_pred)
    xm = y_true - mx
    ym = y_pred - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)) + 1e-5
    r = r_num / r_den
    r = np.clip(r, -1.0, 1.0)
    return r ** 2



class GAN():
    
    def __init__(self, input_dim):
        self.lr = 0.0002
        optimizer = optim.Adam(self.lr)
        optimizer_distiller = optim.Adam(self.lr)
        optimizer_regressor = optim.Adam(self.lr)

        L2_reg = 0.1
        ft_bank_baseline = 16
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

        # Build and compile the cf predictorinv_inv
        self.gender_classifier =  nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.gender_classifier_loss = nn.MSELoss()

        self.disease_classifier =  nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 20),
        )

        self.disease_classifier_loss =nn.CrossEntropyLoss()

    def train(self, epochs, relative_abundance, metadata, batch_size=64, fold=0):

        dc_age = np.zeros((int(epochs/10)+1,))
        min_dc = 0
        for epoch in range(epochs):

            ## Turn on to LR decay manually
            # if epoch % 200 == 0:
            #    self.lr = self.lr * 0.75
            #    optimizer = Adam(self.lr)
            #    self.workflow.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
            #    self.distiller.compile(loss=correlation_coefficient_loss, optimizer=optimizer)
            #    self.regressor.compile(loss='mse', optimizer=optimizer)

            # Select a random batch of images
            
           
            training_feature_ctrl_batch, metadata_ctrl_batch, training_feature_batch, metadata_feature_batch = dataset(relative_abundance, metadata, 64)

            
            # ---------------------
            #  Train regressor (cf predictor)
            # ---------------------

            encoded_feature_ctrl_batch = self.encoder.predict(training_feature_ctrl_batch[:,:32,:,:])
            r_loss = self.regressor.train_on_batch(encoded_feature_ctrl_batch, age_ctrl_batch)

            # ---------------------
            #  Train Disstiller
            # ---------------------
            
            g_loss = self.distiller.train_on_batch(training_feature_ctrl_batch[:,:32,:,:], age_ctrl_batch)
            
            # ---------------------
            #  Train Encoder & Classifier
            # ---------------------
            
            c_loss = self.workflow.train_on_batch(training_feature_batch[:,:32,:,:], dx_batch)

            # ---------------------
            #  flip & re-do everything
            # ---------------------

            training_feature_batch = np.flip(training_feature_batch,1)
            training_feature_ctrl_batch = np.flip(training_feature_ctrl_batch,1)


            encoded_feature_ctrl_batch = self.encoder.predict(training_feature_ctrl_batch[:,:32:,:])
            r_loss = self.regressor.train_on_batch(encoded_feature_ctrl_batch, age_ctrl_batch)
            g_loss = self.distiller.train_on_batch(training_feature_ctrl_batch[:,:32,:,:], age_ctrl_batch)
            c_loss = self.workflow.train_on_batch(training_feature_batch[:,:32,:,:], dx_batch)

            # Plot the progress
            if epoch % 50 == 0:
                c_loss_test_1 = self.workflow.evaluate(test_data_aug[:,:32,:,:],      test_dx_aug, verbose = 0, batch_size = batch_size)    
                c_loss_test_2 = self.workflow.evaluate(test_data_aug_flip[:,:32,:,:], test_dx_aug, verbose = 0, batch_size = batch_size)    

                # feature dist corr
                features_dense = self.encoder.predict(train_data_aug[train_dx_aug == 0,:32,:,:],  batch_size = batch_size)
                dc_age[int(epoch/10)] = dcor.u_distance_correlation_sqr(features_dense, train_age_aug[train_dx_aug == 0])
                print ("%d [Acc: %f,  Test Acc: %f %f,  dc: %f]" % (epoch, c_loss[1], c_loss_test_1[1], c_loss_test_2[1], dc_age[int(epoch/10)]))
                sys.stdout.flush()

                self.classifier.save_weights("res_cf_5cv/classifier.h5")
                self.encoder.save_weights("res_cf_5cv/encoder.h5")
                self.workflow.save_weights("res_cf_5cv/workflow.h5")


                ## Turn on to save all intermediate features for posthoc MI computation
                #features_dense = self.encoder.predict(test_data[:,:32,:,:],  batch_size = 64)
                #filename = 'res_cf/features_'+str(fold)+'.txt'
                #np.savetxt(filename,features_dense)
                #score = self.classifier.predict(features_dense,  batch_size = 64)
                #filename = 'res_cf/scores_'+str(fold)+'_'+str(epoch)+'.txt'
                #np.savetxt(filename,score)

                #features_dense = self.encoder.predict(test_data_flip[:,:32,:,:],  batch_size = 64)
                #filename = 'res_cf/features_flip_'+str(fold)+'.txt'
                #np.savetxt(filename,features_dense)
                #score = self.classifier.predict(features_dense,  batch_size = 64)
                #filename = 'res_cf/scores_flip_'+str(fold)+'_'+str(epoch)+'.txt'
                #np.savetxt(filename,score)

                # save intermediate predictions
                prediction = self.workflow.predict(test_data[:,:32,:,:],  batch_size = 64)
                filename = 'res_cf_5cv/prediction_'+str(fold)+'_'+str(epoch)+'.txt'
                np.savetxt(filename,prediction)
                prediction = self.workflow.predict(test_data_flip[:,:32,:,:],  batch_size = 64)
                filename = 'res_cf_5cv/prediction_flip_'+str(fold)+'_'+str(epoch)+'.txt'
                np.savetxt(filename,prediction)

                # save ground-truth
                filename = 'res_cf_5cv/dx_'+str(fold)+'.txt'
                np.savetxt(filename,test_dx)    
                filename = 'res_cf_5cv/cf_'+str(fold)+'.txt'
                np.savetxt(filename,test_age)       