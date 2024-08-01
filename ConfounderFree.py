import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import tqdm

from sklearn.model_selection import StratifiedKFold

from dataset import MicroDataset


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def train_step(model, loader, optimizer, device): 
	criterion_age = nn.CrossEntropyLoss()
	criterion_gender = nn.BCELoss()
	criterion_bmi = nn.CrossEntropyLoss()
	criterion_bodysite = nn.CrossEntropyLoss()
	criterion_disease = nn.CrossEntropyLoss()

	with tqdm(total=len(loader)) as bar:
		for step, batch in enumerate(loader): 
			_, spec, age, gender, bmi, bodysite, disease = batch

			batch_size = len(age)
			age_size = max(1,torch.sum(~torch.isnan(age)).item())
			gender_size = max(1, torch.sum(~torch.isnan(gender)).item())
			bmi_size = max(1, torch.sum(~torch.isnan(bmi)).item())
			bodysite_size = max(1, torch.sum(~torch.isnan(bodysite)).item())

			spec = spec.type(torch.cuda.FloatTensor).to(device)

			age = age.type(torch.cuda.FloatTensor).to(device)
			invalid_age = torch.isnan(age)
			age_dim = age.size(1)

			gender = gender.type(torch.cuda.FloatTensor).to(device)
			invalid_gender = torch.isnan(gender)

			bmi = bmi.type(torch.cuda.FloatTensor).to(device)
			invalid_bmi = torch.isnan(bmi)
			bmi_dim = bmi.size(1)

			bodysite = bodysite.type(torch.cuda.FloatTensor).to(device)
			invalid_bodysite = torch.isnan(bodysite)
			bodysite_dim = bodysite.size(1)

			disease = disease.type(torch.cuda.FloatTensor).to(device)
			invalid_disease = torch.isnan(disease)
			disease_dim = disease.size(1)
			
			optimizer.zero_grad()
			model.train()
			
			pred_age, pred_gender, pred_bmi, pred_bodysite, pred_disease = model(spec, age, gender, bmi, bodysite)
			
			loss = criterion_age(pred_age[~invalid_age].view(-1, age_dim), age[~invalid_age].view(-1, age_dim)) * (batch_size/age_size) + \
					criterion_gender(pred_gender[~invalid_gender], gender[~invalid_gender]) * (batch_size/gender_size) + \
					criterion_bmi(pred_bmi[~invalid_bmi].view(-1, bmi_dim), bmi[~invalid_bmi].view(-1, bmi_dim)) * (batch_size/bmi_size) + \
					criterion_bodysite(pred_bodysite[~invalid_bodysite].view(-1, bodysite_dim), bodysite[~invalid_bodysite].view(-1, bodysite_dim)) * (batch_size/bodysite_size) + \
					criterion_disease(pred_disease[~invalid_disease].view(-1, disease_dim), disease[~invalid_disease].view(-1, disease_dim)) 
					
			
			loss.backward()

			bar.set_description('Train')
			bar.set_postfix(lr=get_lr(optimizer), loss=loss.item())
			bar.update(1)

			optimizer.step()
	return 


def eval_step(model, loader, device): 
	model.eval()

	ages, pred_ages = [], []
	genders, pred_genders = [], []
	bmis, pred_bmis = [], []
	bodysites, pred_bodysites = [], []
	diseases, pred_diseases = [], []
	
	sample_ids = []
	
	with tqdm(total=len(loader)) as bar:
		for _, batch in enumerate(loader): 

			sample_id, spec, age, gender, bmi, bodysite, disease = batch

			spec = spec.type(torch.cuda.FloatTensor).to(device)

			age = age.type(torch.cuda.FloatTensor).to(device)
			invalid_age = torch.isnan(age)

			gender = gender.type(torch.cuda.FloatTensor).to(device)
			invalid_gender = torch.isnan(gender)

			bmi = bmi.type(torch.cuda.FloatTensor).to(device)
			invalid_bmi = torch.isnan(bmi)

			bodysite = bodysite.type(torch.cuda.FloatTensor).to(device)
			invalid_bodysite = torch.isnan(bodysite)

			disease = disease.type(torch.cuda.FloatTensor).to(device)
			invalid_disease = torch.isnan(disease)

			with torch.no_grad():
				pred_age, pred_gender, pred_bmi, pred_bodysite, pred_disease = model(spec, age, gender, bmi, bodysite)

			bar.set_description('Eval')
			bar.update(1)

			age_dim = age.size(1)
			ages.append(age[~invalid_age].view(-1, age_dim).detach().cpu())
			pred_ages.append(pred_age[~invalid_age].view(-1, age_dim).detach().cpu())

			genders.append(gender[~invalid_gender].detach().cpu())
			pred_genders.append(pred_gender[~invalid_gender].detach().cpu())

			bmi_dim = bmi.size(1)
			bmis.append(bmi[~invalid_bmi].view(-1, bmi_dim).detach().cpu())
			pred_bmis.append(pred_bmi[~invalid_bmi].view(-1, bmi_dim).detach().cpu())

			bodysite_dim = bodysite.size(1)
			bodysites.append(bodysite[~invalid_bodysite].view(-1, bodysite_dim).detach().cpu())
			pred_bodysites.append(pred_bodysite[~invalid_bodysite].view(-1, bodysite_dim).detach().cpu())

			disease_dim = disease.size(1)
			diseases.append(disease[~invalid_disease].view(-1, disease_dim).detach().cpu())
			pred_diseases.append(pred_disease[~invalid_disease].view(-1, disease_dim).detach().cpu())

			sample_ids = sample_ids + list(sample_id)

	ages = torch.cat(ages, dim = 0)
	pred_ages = torch.cat(pred_ages, dim = 0)

	genders = torch.cat(genders, dim = 0)
	pred_genders = torch.cat(pred_genders, dim = 0)

	bmis = torch.cat(bmis, dim = 0)
	pred_bmis = torch.cat(pred_bmis, dim = 0)

	bodysites = torch.cat(bodysites, dim = 0)
	pred_bodysites = torch.cat(pred_bodysites, dim = 0)

	diseases = torch.cat(diseases, dim = 0)
	pred_diseases = torch.cat(pred_diseases, dim = 0)

	return sample_ids, ages, pred_ages, genders, pred_genders, bmis, pred_bmis, bodysites, pred_bodysites, diseases, pred_diseases



if __name__ == "__main__":
	device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

    # 0. Settings
	# seed = 42
	# torch.manual_seed(seed)
	# np.random.seed(seed)
	# random.seed(seed)

	# model settings: 
	metadatas = ['BMI', 'gender', 'age', 'bodysite', 'phenotype']
	df_relative_abundance = pd.read_csv('Data/relative_abundance.csv', index_col=0)
	species = df_relative_abundance.columns.values.tolist()
	print('species num:', len(species)-1)

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
	# print(species)
	species_dict = {k: i for i, k in enumerate(species)}
	df_relative_abundance = pd.read_csv('Data/relative_abundance.csv')
	dataset = MicroDataset(data=df_relative_abundance, metadata=metadata_df, species_dict=species_dict, output='./NetworkInput/')
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
		device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
		print(f'Device: {device}')
		model = MicroKPNN_MTL(species, in_dim, bodysite_num, disease_num)
		num_params = sum(p.numel() for p in model.parameters())
		print(f'{str(model)} #Params: {num_params}')
		model.to(device)

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

