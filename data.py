from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os
import pickle

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
	

if __name__ == "__main__":
	relative_abundace = pd.read_csv('Data/train_relative_abundance.csv')
	metadata = pd.read_csv('Data/train_metadata.csv')
	training_feature_ctrl_batch, metadata_ctrl_batch, training_feature_batch, metadata_feature_batch = dataset(relative_abundace, metadata, 64)
	
	


		
