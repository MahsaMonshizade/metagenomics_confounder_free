import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os
import pickle

from random import sample 



class MicroDataset(Dataset): 
	def __init__(self, data_path, metadata_path, species_dict, output): 
		"""
        Initialize MicroDataset with data and metadata paths, species dictionary, and output directory.
        If embedding data file exists, load it; otherwise, process the data and create the embedding.
        """

		embedding_file_path = output + "/embedding_data.pkl"
	
		print(f"The file {embedding_file_path} does not exist.")
		self.process_data(data_path, metadata_path, species_dict, output)
		

	def process_data(self, data_path, metadata_path, species_dict, output):
		"""
        Process the data by merging species information, handling missing values, and creating encoding dictionaries.
        """
		df = pd.read_csv(data_path)
		meta_df = pd.read_csv(metadata_path)
		print('Load {} data from {}'.format(len(df), data_path))
		print('Load {} metadata from {}\n'.format(len(meta_df), metadata_path))

		# add lacked species columns to df
		new_columns = [k for k in species_dict.keys() if k not in list(df.columns)]
		df[new_columns] = 0. 
		
		# merge df & meta_df
		df = df.merge(meta_df, how='left', left_on='SampleID', right_on='SampleID')


		self.sample_ids, self.species, self.genders, self.diseases= self.embedding_data(df, species_dict)
		
		embedding_data = {
    	'sample_ids': self.sample_ids,
    	'species': self.species,
    	'genders': self.genders,
    	'diseases': self.diseases,
		}
		print('xxxxxx')
		print(len(embedding_data['sample_ids']))



	def create_encoding_dict(self, output, df, column, file_name):
		"""
        Create an encoding dictionary from a DataFrame column, write it to a CSV file, and return the dictionary.
        """
		column_list = list(set(df[column].dropna().tolist()))
		encoding_dict = {k: i for i, k in enumerate(column_list)}
		encoding_dict[np.nan] = np.nan
		
		with open(os.path.join(output, file_name), "w") as file:
			writer = csv.writer(file)
			for key, val in encoding_dict.items():
				writer.writerow([key, val])
		return encoding_dict, column_list
		
	
	def embedding_data(self, df, species_dict):
		"""
        Embed the data by converting categorical information into numerical representations.
        Return lists containing sample ids, species, ages, genders, bmis, bodysites, diseases, and diseases indices.
        """
		sample_ids, species, genders, diseases= [], [], [], []
		
		print('Embedding the data...')
		for i, row in tqdm(df.iterrows(), total=df.shape[0]): 
			spec = row[species_dict.keys()].astype(float).tolist()
			
			sample_ids.append(row['SampleID'])

			species.append(np.array(spec))
			
			genders.append(np.array(row['sex']))

			diseases.append(np.array(row['disease']))
				
			

		return sample_ids, species, genders, diseases
		

	def one_hot(self, idx, num_classes):
		one_hot_enc = np.zeros(num_classes)
		one_hot_enc[idx] = 1
		return one_hot_enc


	def __getitem__(self, idx): 
		return self.sample_ids[idx], self.species[idx], self.genders[idx], self.diseases[idx]


	def __len__(self): 
		return len(self.sample_ids)