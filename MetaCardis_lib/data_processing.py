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

def load_and_transform_data(file_path):
    """
    Load data from CSV, apply CLR transformation, and return transformed DataFrame with 'uid'.
    """
    data = pd.read_csv(file_path)
    uid = data['SampleID']
    X = data.drop(columns=['SampleID']).values
    X_log = np.log(X + 1)
    X_log_df = pd.DataFrame(X_log, columns=data.columns[1:])
    X_log_df['SampleID'] = uid
    return X_log_df[['SampleID'] + list(X_log_df.columns[:-1])]

def preprocess_metadata(metadata):
    """Convert categorical metadata into numeric features."""
    disease_dict = {8: 0, 3: 1}
    metadata = metadata.copy()
    metadata['disease_numeric'] = metadata['PATGROUPFINAL_C'].map(disease_dict)
    return metadata


def create_batch(relative_abundance, metadata, batch_size, is_test=False, device='cpu'):
    """
    Create a batch of data by sampling from the metadata and relative abundance data.
    """
    metadata = preprocess_metadata(metadata)
    proportions = metadata['disease_numeric'].value_counts(normalize=True)
    num_samples_per_group = (proportions * batch_size).round().astype(int)

    # Sample metadata
    metadata_feature_batch = metadata.groupby('disease_numeric').apply(
        lambda x: x.sample(n=num_samples_per_group[x.name], random_state=42)
    ).reset_index(drop=True)

    # Sample relative abundance
    training_feature_batch = relative_abundance[relative_abundance['SampleID'].isin(metadata_feature_batch['SampleID'])]
    training_feature_batch = training_feature_batch.set_index('SampleID').reindex(metadata_feature_batch['SampleID']).reset_index()
    training_feature_batch = training_feature_batch.drop(columns=['SampleID'])

    # Convert to tensors
    training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32).to(device)
    metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32).to(device)

    if is_test:
        return training_feature_batch, metadata_batch_disease

    # Disease batch
    disease_metadata = metadata[metadata['disease_numeric'] == 1]
    run_ids = disease_metadata['SampleID']
    disease_relative_abundance = relative_abundance[relative_abundance['SampleID'].isin(run_ids)]

    disease_idx = np.random.permutation(disease_metadata.index)[:batch_size]
    training_feature_disease_batch = disease_relative_abundance.loc[disease_idx].drop(columns=['SampleID'])
    metadata_disease_batch = disease_metadata.loc[disease_idx]

    training_feature_disease_batch = torch.tensor(training_feature_disease_batch.values, dtype=torch.float32).to(device)
    metadata_disease_batch_drug = torch.tensor(metadata_disease_batch['METFORMIN_C'].values, dtype=torch.float32).to(device)

    return (training_feature_disease_batch, metadata_disease_batch_drug, 
            training_feature_batch, metadata_batch_disease)

def dcor_calculation_data(relative_abundance, metadata, device='cpu'):
    """
    Prepare data for distance correlation calculation.
    """
    metadata = preprocess_metadata(metadata)

    # Control group
    ctrl_metadata = metadata[metadata['disease_numeric'] == 0]
    ctrl_run_ids = ctrl_metadata['SampleID']
    ctrl_relative_abundance = relative_abundance[relative_abundance['SampleID'].isin(ctrl_run_ids)]
    ctrl_idx = ctrl_metadata.index.tolist()
    training_feature_ctrl = ctrl_relative_abundance.loc[ctrl_idx].drop(columns=['SampleID'])
    metadata_ctrl = ctrl_metadata.loc[ctrl_idx]

    training_feature_ctrl = torch.tensor(training_feature_ctrl.values, dtype=torch.float32).to(device)
    metadata_ctrl_age = torch.tensor(metadata_ctrl['METFORMIN_C'].values, dtype=torch.float32)
    # Disease group
    disease_metadata = metadata[metadata['disease_numeric'] == 1]
    disease_run_ids = disease_metadata['SampleID']
    disease_relative_abundance = relative_abundance[relative_abundance['SampleID'].isin(disease_run_ids)]
    disease_idx = disease_metadata.index.tolist()
    training_feature_disease = disease_relative_abundance.loc[disease_idx].drop(columns=['SampleID'])
    metadata_disease = disease_metadata.loc[disease_idx]

    training_feature_disease = torch.tensor(training_feature_disease.values, dtype=torch.float32).to(device)
    metadata_disease_age = torch.tensor(metadata_disease['METFORMIN_C'].values, dtype=torch.float32)

    return (training_feature_ctrl, metadata_ctrl_age, 
            training_feature_disease, metadata_disease_age)