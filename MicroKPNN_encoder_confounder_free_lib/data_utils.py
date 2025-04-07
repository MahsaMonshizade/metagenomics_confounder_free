import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_transform_data(file_path):
    """
    Load data from CSV, apply log transformation, and return transformed DataFrame with 'SampleID'.
    """
    data = pd.read_csv(file_path)
    uid = data['SampleID']
    X = data.drop(columns=['SampleID']).values
    X_log = np.log(X + 1)
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(X_log)
    X_log_df = pd.DataFrame(features_normalized, columns=data.columns[1:])
    X_log_df['SampleID'] = uid
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



# def load_and_transform_data(file_path):
#     """
#     Load data from CSV, apply CLR transformation, and return
#     a transformed DataFrame with 'SampleID'.
#     """
#     data = pd.read_csv(file_path)
    
#     # Extract 'SampleID' separately
#     uid = data['SampleID']
    
#     # Extract abundance matrix (drop the 'SampleID' column)
#     X = data.drop(columns=['SampleID']).values
    
#     # 1) Add a small pseudocount to handle zeros
#     pseudocount = 1e-6
#     X_pseudo = X + pseudocount
    
#     # 2) Compute geometric mean per sample (row)
#     # Using exp(mean(log(...))) to avoid overflow from direct multiplication
#     geom_means = np.exp(np.mean(np.log(X_pseudo), axis=1, keepdims=True))
    
#     # 3) Perform the CLR transform:
#     #    CLR(x) = log( x / geometric_mean(x) )
#     X_clr = np.log(X_pseudo / geom_means)
    
#     # 4) (Optional) Standard scale if you'd like each feature to have 0 mean, unit variance
#     # scaler = StandardScaler()
#     # X_clr_scaled = scaler.fit_transform(X_clr)
    
#     # Build a new DataFrame with the same feature column names, plus 'SampleID'
#     feature_names = data.columns[1:]  # all except 'SampleID'
#     X_clr_df = pd.DataFrame(X_clr, columns=feature_names)
#     X_clr_df['SampleID'] = uid
    
#     # Reorder columns so 'SampleID' is first
#     return X_clr_df[['SampleID'] + list(X_clr_df.columns[:-1])]


# def get_data(file_path, metadata_file_path):
#     """
#     Load and merge metadata and CLR-transformed abundance data for training.
#     """
#     # This will now do CLR transform instead of simple log transform
#     relative_abundance = load_and_transform_data(file_path)
    
#     # Read metadata
#     metadata = pd.read_csv(metadata_file_path)

#     # Merge metadata and abundance data on SampleID
#     merged_data = pd.merge(metadata, relative_abundance, on='SampleID')

#     return merged_data
