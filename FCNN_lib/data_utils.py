import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_transform_data(file_path, pseudocount=1e-6):
    """
    Load data from CSV, normalize each row to sum to 1, add a pseudocount to avoid zeros,
    apply a centered log‑ratio (CLR) transformation, and standard-scale the data.
    Returns a DataFrame with 'SampleID' preserved.
    """
    data = pd.read_csv(file_path)
    uid = data['SampleID']
    df_features = data.drop(columns=['SampleID'])
    
    # Normalize each row so that it sums to 1 (converting to proportions)
    row_sums = df_features.sum(axis=1)
    df_normalized = df_features.div(row_sums, axis=0)
    
    # Add a small constant to avoid zeros in the CLR transformation
    df_normalized = df_normalized + pseudocount
    
    # Compute the geometric mean for each sample (row)
    gm = np.exp(np.log(df_normalized).mean(axis=1))
    
    # Apply CLR transformation: log(x) - log(geometric mean)
    clr_data = np.log(df_normalized).subtract(np.log(gm), axis=0)
    
    # Standard scaling: zero mean, unit variance
    scaler = StandardScaler()
    clr_scaled = scaler.fit_transform(clr_data)
    clr_scaled_df = pd.DataFrame(clr_scaled, columns=clr_data.columns)
    
    # Reinsert the SampleID column
    clr_scaled_df.insert(0, 'SampleID', uid)
    
    return clr_scaled_df

def get_data(file_path, metadata_file_path):
    """
    Load and merge the CLR-transformed abundance data with metadata.
    """
    relative_abundance = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)
    merged_data = pd.merge(metadata, relative_abundance, on='SampleID')
    return merged_data
