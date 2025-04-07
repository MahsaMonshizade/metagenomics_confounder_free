import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_transform_data(file_path):
    """
    Load data from CSV, apply log transformation, and return transformed DataFrame with 'SampleID'.
    """
    data = pd.read_csv(file_path)
    uid = data['SampleID']
    
    # Drop the SampleID column for numerical processing
    X = data.drop(columns=['SampleID']).values.astype(np.float32)
    
    # Add a small constant to avoid zeros
    X = X + 1e-6
    
    # Apply log transformation; using log(x + log_offset)
    # X_log = np.log(X + 1)
    
    # Normalize the transformed features
    # scaler = StandardScaler()
    # features_normalized = scaler.fit_transform(X)
    
    # Create a DataFrame for the normalized features
    X_log_df = pd.DataFrame(X, columns=data.columns[1:])
    X_log_df['SampleID'] = uid
    
    # Reorder so that SampleID is the first column
    return X_log_df[['SampleID'] + list(X_log_df.columns[:-1])]

def get_data(file_path, metadata_file_path):
    """
    Load and merge metadata and relative abundance data for training.
    """
    relative_abundance = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # Merge metadata and relative abundance data
    merged_data = pd.merge(metadata, relative_abundance, on='SampleID')

    return merged_data
