import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import scipy.ndimage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
import argparse
import glob
import dcor
import pandas as pd

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from dataset import MicroDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 0. Settings
	seed = 42
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	# model settings: 
	metadatas = ['BMI', 'gender', 'age', 'bodysite', 'phenotype']
	df_relative_abundance = pd.read_csv('Data/relative_abundance.csv', index_col = 0)
	species = df_relative_abundance.columns.values.tolist()
	print('species num:', len(species))

	metadata_df = pd.read_csv('Data/metadata.csv')
	print('meta-data:', list(metadata_df.columns))
	in_dim = len(species) 
	
	bodysite_num = len(list(set(metadata_df['BodySite'].values.tolist())))
	print('bodysite num:', bodysite_num)
	disease_num = len(list(set(metadata_df['disease'].values.tolist())))
	print('disease num:', disease_num)

	# training settings: 
	lr = 0.0001
	batch_size = 64
	epoch_num = 100
	early_stop_step = 20
	k_fold = 5
	
	# check the directory

	# checkpoint_dir = "/".join(args.checkpoint_path.split('/')[:-1])
	# records_dir = "/".join(args.records_path.split('/')[:-1])

	# --------------- K-Fold Validation --------------- # 
	print('Loading the dataset...')
	
	dataset = MicroDataset(data=df_relative_abundance, metadata=metadata_df, output='./NetworkInput/')
	assert len(dataset.get_disease_dict()) - 1 == disease_num, "Setting and metadata are not match, disease_num={}, \
															but there are {} diseases in metadata".format(disease_num, len(dataset.get_disease_dict)-1)# split the indices into k-fold
    # split the indices into k-fold
	indices = list(range(len(dataset)))
	skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
	for fold_i, (train_indices, valid_indices) in enumerate(skf.split(np.expand_dims(np.array(indices, dtype=int), axis=1), np.array(dataset.diseases_idx, dtype=int))): 
		print('\n# --------------- Fold-{} --------------- #'.format(fold_i))
		# modify the checkpoint_path and resume_path by k-fold
		# resume_path_foldi = args.resume_path.replace('.pt', '_fold{}.pt'.format(str(fold_i)))
		# checkpoint_path_foldi = args.checkpoint_path.replace('.pt', '_fold{}.pt'.format(str(fold_i)))
		# refresh the values used to control early stop
		early_stop_patience = 0
		best_disease_acc = 0

		# 1. Model 
		print('Establishing the model...')
		device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
		print(f'Device: {device}')
		# model = MicroKPNN_MTL(species, args.edge_list, in_dim, hidden_dim, bodysite_num, disease_num, mask.to(device))
		# num_params = sum(p.numel() for p in model.parameters())
		# print(f'{str(model)} #Params: {num_params}')
		# model.to(device)

		# 2. Data
		print('# Train: {}, # Val: {}'.format(len(train_indices), len(valid_indices)))

		train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
		valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)
		train_loader = torch.utils.data.DataLoader(dataset,
													batch_size=batch_size,
													shuffle=False,
													num_workers=0,
													drop_last=True,
													sampler=train_sampler)
		val_loader = torch.utils.data.DataLoader(dataset,
													batch_size=batch_size,
													shuffle=False,
													num_workers=0,
													drop_last=True,
													sampler=valid_sampler)

