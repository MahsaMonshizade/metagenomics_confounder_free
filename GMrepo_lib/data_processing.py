# data_processing.py

import numpy as np
import pandas as pd
import torch
import random

def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clr_transformation(X):
    """Perform Centered Log-Ratio (CLR) transformation on the data."""
    geometric_mean = np.exp(np.mean(np.log(X), axis=1))
    return np.log(X / geometric_mean[:, np.newaxis])

def load_and_transform_data(file_path):
    """
    Load data from CSV, apply CLR transformation, and return transformed DataFrame with 'uid'.
    """
    data = pd.read_csv(file_path)
    uid = data['loaded_uid']
    X = data.drop(columns=['loaded_uid']).values
    X += 1e-6  # Adding a small pseudocount to avoid log(0)
    X_clr = clr_transformation(X)
    X_clr_df = pd.DataFrame(X_clr, columns=data.columns[1:])
    X_clr_df['uid'] = uid
    return X_clr_df[['uid'] + list(X_clr_df.columns[:-1])]

def preprocess_metadata(metadata):
    """Convert categorical metadata into numeric features."""
    disease_dict = {'D006262': 0, 'D003093': 1}
    metadata = metadata.copy()
    metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    return metadata

def create_batch(relative_abundance, metadata, batch_size, is_test=False, device='cpu'):
    """
    Create a batch of data by sampling from the metadata and relative abundance data.
    """
    metadata = preprocess_metadata(metadata)
    proportions = metadata['disease'].value_counts(normalize=True)
    num_samples_per_group = (proportions * batch_size).round().astype(int)

    # Sample metadata
    metadata_feature_batch = metadata.groupby('disease').apply(
        lambda x: x.sample(n=num_samples_per_group[x.name], random_state=42)
    ).reset_index(drop=True)

    # Sample relative abundance
    training_feature_batch = relative_abundance[relative_abundance['uid'].isin(metadata_feature_batch['uid'])]
    training_feature_batch = training_feature_batch.set_index('uid').reindex(metadata_feature_batch['uid']).reset_index()
    training_feature_batch = training_feature_batch.drop(columns=['uid'])

    # Convert to tensors
    training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32).to(device)
    metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32).to(device)

    if is_test:
        return training_feature_batch, metadata_batch_disease

    # Control batch
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    run_ids = ctrl_metadata['uid']
    ctrl_relative_abundance = relative_abundance[relative_abundance['uid'].isin(run_ids)]

    ctrl_idx = np.random.permutation(ctrl_metadata.index)[:batch_size]
    training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].drop(columns=['uid'])
    metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]

    training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32).to(device)
    metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch['host_age_zscore'].values, dtype=torch.float32).to(device)
    metadata_ctrl_batch_bmi = torch.tensor(metadata_ctrl_batch['BMI_zscore'].values, dtype=torch.float32).to(device)

    return (training_feature_ctrl_batch, metadata_ctrl_batch_age, metadata_ctrl_batch_bmi, 
            training_feature_batch, metadata_batch_disease)

def dcor_calculation_data(relative_abundance, metadata, device='cpu'):
    """
    Prepare data for distance correlation calculation.
    """
    metadata = preprocess_metadata(metadata)

    # Control group
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    ctrl_run_ids = ctrl_metadata['uid']
    ctrl_relative_abundance = relative_abundance[relative_abundance['uid'].isin(ctrl_run_ids)]
    ctrl_idx = ctrl_metadata.index.tolist()
    training_feature_ctrl = ctrl_relative_abundance.loc[ctrl_idx].drop(columns=['uid'])
    metadata_ctrl = ctrl_metadata.loc[ctrl_idx]

    training_feature_ctrl = torch.tensor(training_feature_ctrl.values, dtype=torch.float32).to(device)
    metadata_ctrl_age = torch.tensor(metadata_ctrl['host_age_zscore'].values, dtype=torch.float32)
    metadata_ctrl_bmi = torch.tensor(metadata_ctrl['BMI_zscore'].values, dtype=torch.float32)
    # Disease group
    disease_metadata = metadata[metadata['disease'] == 'D003093']
    disease_run_ids = disease_metadata['uid']
    disease_relative_abundance = relative_abundance[relative_abundance['uid'].isin(disease_run_ids)]
    disease_idx = disease_metadata.index.tolist()
    training_feature_disease = disease_relative_abundance.loc[disease_idx].drop(columns=['uid'])
    metadata_disease = disease_metadata.loc[disease_idx]

    training_feature_disease = torch.tensor(training_feature_disease.values, dtype=torch.float32).to(device)
    metadata_disease_age = torch.tensor(metadata_disease['host_age_zscore'].values, dtype=torch.float32)
    metadata_disease_bmi = torch.tensor(metadata_disease['BMI_zscore'].values, dtype=torch.float32)

    return (training_feature_ctrl, metadata_ctrl_age,  metadata_ctrl_bmi,
            training_feature_disease, metadata_disease_age, metadata_disease_bmi)