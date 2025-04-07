import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_transform_data(file_path):
    """
    Load data from CSV, apply log transformation, and return transformed DataFrame with 'SampleID'.
    """
    data = pd.read_csv(file_path)
    uid = data['SampleID']
    df_features = data.drop(columns=['SampleID'])
    # df_features = df_features / 100.0
    df_features = df_features + 1e-6
    gm = np.exp(np.log(df_features).mean(axis=1))
    clr_data = np.log(df_features) - np.log(gm).values.reshape(-1, 1)
    scaler = StandardScaler()
    clr_scaled = scaler.fit_transform(clr_data)
    clr_scaled_df = pd.DataFrame(clr_scaled, columns=clr_data.columns)
    clr_scaled_df.insert(0, 'SampleID', uid)
    
    return clr_scaled_df


def get_data(file_path, metadata_file_path):
    """
    Load and merge metadata and relative abundance data for training.
    """
    relative_abundance = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # Merge metadata and relative abundance data
    merged_data = pd.merge(metadata, relative_abundance, on='SampleID')

    return merged_data







