import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.init as init
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
	age_dict = {'Children Adolescents': 0, 'Young Adult': 1, 'Middle Aged': 2}
	disease_dict = {'D006262': 0, 'D047928': 1}
	
	metadata['age_numeric'] = metadata['host_age'].map(age_dict)
	metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
	
	age_encoded = np.array([one_hot(idx, len(age_dict)) for idx in metadata['age_numeric']])
	age_encoded_df = pd.DataFrame(age_encoded, columns=[f'age_{i}' for i in range(len(age_dict))])
	
	return pd.concat([metadata, age_encoded_df], axis=1) 

def create_batch(relative_abundance, metadata, batch_size, is_test=False):
	"""
	Creates a balanced batch of data by sampling from the metadata and relative abundance data.
	
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

	# YH: modify the sampling ================================================
	# Get unique diseases
	unique_diseases = metadata['disease'].unique()
	
	# Ensure the batch size is divisible by the number of classes
	if batch_size % len(unique_diseases) != 0:
		raise ValueError("Batch size must be divisible by the number of classes.")

	samples_per_class = batch_size // len(unique_diseases)
	
	# Sample the data to ensure each class is represented equally
	sampled_metadata = pd.concat([
		metadata[metadata['disease'] == disease].sample(n=samples_per_class, replace=True)
		for disease in unique_diseases
	]).reset_index(drop=True)
	# ========================================================================

	# Create training feature batch based on sampled metadata
	training_feature_batch = relative_abundance[
		relative_abundance['Unnamed: 0'].isin(sampled_metadata['run_id'])
	]
	training_feature_batch = training_feature_batch.set_index('Unnamed: 0').reindex(
		sampled_metadata['run_id']
	).reset_index()
	training_feature_batch.rename(columns={'Unnamed: 0': 'run_id'}, inplace=True)
	training_feature_batch = training_feature_batch.drop(columns=['run_id'])

	training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
	metadata_batch_disease = torch.tensor(sampled_metadata['disease_numeric'].values, dtype=torch.float32)

	if is_test:
		return training_feature_batch, metadata_batch_disease
	
	# Sample control data for training
	ctrl_metadata = metadata[metadata['disease'] == 'D006262']
	run_ids = ctrl_metadata['run_id']
	ctrl_relative_abundance = relative_abundance[relative_abundance['Unnamed: 0'].isin(run_ids)]
	
	ctrl_idx = np.random.permutation(ctrl_metadata.index)[:batch_size]
	training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'Unnamed: 0': 'run_id'}).drop(columns=['run_id'])
	metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]
	
	training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
	metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch[['age_0', 'age_1', 'age_2']].values, dtype=torch.float32)
	
	return training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease



class NonGAN(nn.Module): 
	def __init__(self, input_dim):
		super(NonGAN, self).__init__()
		"""
		Initializes the NonGAN with an encoder and a disease classifier.
		
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
			nn.Linear(16, 1)
		)

		self.disease_classifier_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
		self.lr = 0.0001
		self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), self.lr)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

		self._initialize_weights()

	def _initialize_weights(self):
		"""
		Initializes weights of the model using Xavier initialization.
		"""
		for m in self.modules():
			if isinstance(m, nn.Linear):
				init.xavier_uniform_(m.weight)
				if m.bias is not None:
					init.zeros_(m.bias)
					
	def train(self, epochs, relative_abundance, metadata, batch_size=64): 
		"""
		Trains the NonGAN model for a specified number of epochs.
		
		Args:
			epochs (int): Number of training epochs.
			relative_abundance (pd.DataFrame): DataFrame with relative abundance values.
			metadata (pd.DataFrame): DataFrame with metadata.
			batch_size (int): Number of samples per batch.
		"""
		best_acc = 0
		early_stop_patience = 0
		early_stop_step = 30
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
			self.scheduler.step(disease_acc)

			if disease_acc > best_acc:
				best_acc = disease_acc
				early_stop_patience = 0
			else:
				early_stop_patience += 1
				# YH: for debugging
				# print(f"Early stopping patience: {early_stop_patience} / {early_stop_step}, learning rate: {self.optimizer.param_groups[0]['lr']}")

			if early_stop_patience == early_stop_step:
				break

			print(f"Epoch {epoch + 1}/{epochs}, c_loss: {c_loss.item()}, accuracy: {disease_acc}")
			if epoch % 100 == 99: 
				self.evaluate(pd.read_csv('Data/new_test_relative_abundance.csv'), pd.read_csv('Data/new_test_metadata.csv'), batch_size=batch_size)
	
	def evaluate(self, relative_abundance, metadata, batch_size):
		"""
		Evaluates the trained NonGAN model on test data.
		
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
		print(f"Test result --> accuracy: {disease_acc}, c_loss: {c_loss.item()}")

if __name__ == "__main__":
	relative_abundance = pd.read_csv('Data/new_train_relative_abundance.csv')
	metadata = pd.read_csv('Data/new_train_metadata.csv')
	NonGAN_cf = NonGAN(input_dim=relative_abundance.shape[1] - 1)
	NonGAN_cf.train(epochs=1500, relative_abundance=relative_abundance, metadata=metadata, batch_size=256)
	
	test_relative_abundance = pd.read_csv('Data/new_test_relative_abundance.csv')
	test_metadata = pd.read_csv('Data/new_test_metadata.csv')
	NonGAN_cf.evaluate(relative_abundance=test_relative_abundance, metadata=test_metadata, batch_size=test_metadata.shape[0])
