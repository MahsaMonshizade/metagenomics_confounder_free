import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import pandas as pd



class MaskedLinear(nn.Module):
	def __init__(self, in_dim, out_dim, indices_mask):
		"""
		in_features: number of input features
		out_features: number of output features
		indices_mask: tensor used to create the mask
		"""
		super(MaskedLinear, self).__init__()
 
		def backward_hook(grad):
			# Clone due to not being allowed to modify in-place gradients
			out = grad.clone()
			out[~indices_mask] = 0
			return out
 
		self.linear = nn.Linear(in_dim, out_dim)
		self.linear.weight.data[indices_mask] = 0 # zero out bad weights
		self.linear.weight.register_hook(backward_hook) # hook to zero out bad gradients
 
	def forward(self, x): 
		return self.linear(x)

class MicroKPNN_MTL(nn.Module): 
	def __init__(self, species, edge_list, in_dim, hidden_dim, mask): 
		super(MicroKPNN_MTL, self).__init__() 
		
		# generate the mask
		metadatas = ['gender', 'phenotype']
		edge_df = pd.read_csv(edge_list)

		edge_df['parent'] = edge_df['parent'].astype(str)
		parent_nodes = list(set(edge_df['parent'].tolist()))
		parent_nodes = [node for node in parent_nodes if node not in metadatas] # remove metadata from parent nodes

		child_nodes = species 
		
		parent_nodes.sort()
		parent_dict = {k: i for i, k in enumerate(parent_nodes)}
		child_nodes.sort()
		child_dict = {k: i for i, k in enumerate(child_nodes)}
		self.species_dict = child_dict # used outside the class
				
		for i, row in edge_df.iterrows():
			if row["parent"] not in metadatas and row['child'] != 'Unnamed: 0': 
				mask[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1
		mask = mask > 0 
		
		# establish the first customized linear
		assert in_dim == len(child_nodes), "Setting and edge list are not match, in_dim={}, but len(child_nodes)={}".format(in_dim, len(child_nodes))
		assert hidden_dim == len(parent_nodes), "Setting and edge list are not match, hidden_dim={}, but len(parent_nodes)={}".format(hidden_dim, len(parent_nodes))

		self.customized_linear = MaskedLinear(in_dim, hidden_dim, mask.permute(1, 0))
	
									
		self.decoder_gender = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
										nn.ReLU(), 
										nn.Dropout(0.2), # add dropout layer in case overfitting
										nn.Linear(hidden_dim, 1), 
										nn.Sigmoid(), 
										)

		self.decoder_disease = nn.Sequential(nn.Linear(hidden_dim+1, hidden_dim), 
										nn.ReLU(), 
										nn.Dropout(0.2), # add dropout layer in case overfitting
										nn.Linear(hidden_dim, 1), 
										nn.Sigmoid(), # add dropout layer in case overfitting
										)

		for m in self.modules(): 
			if isinstance(m, (nn.Linear)): 
				nn.init.xavier_normal_(m.weight)
	
	def get_species_dict(self): 
		return self.species_dict

	def forward(self, x, real_gender=None): 
		x = self.customized_linear(x)

    # ensure real_gender is 2-D
		if real_gender is not None and real_gender.dim() == 1:
			real_gender = real_gender.unsqueeze(1)

		gender = self.decoder_gender(x)
		disease = self.decoder_disease(torch.cat((x, real_gender), dim=1))
		return gender, disease

		# print("hihihihihihi")
		# print(disease.shape)

		# mix_age, mix_gender, and mix_bmi are only used to predict disease
		# we still need the predicted meta-data for training meta-data prediction
		return gender, disease