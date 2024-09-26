# run_baseline.py

import os
import pandas as pd
from data_processing import set_seed, load_and_transform_data
from baseline_model import BaselineModel, train_baseline_model

def main():
    """Main function to run the baseline model training."""
    set_seed(42)
    
    # Create directories if they don't exist
    os.makedirs('baseline_models', exist_ok=True)
    os.makedirs('baseline_plots', exist_ok=True)
    
    # Load and transform training data
    file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_train.csv'
    metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_train.csv'
    X_clr_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)
    
   # Initialize Baseline model
    baseline_model = BaselineModel(input_dim=X_clr_df.shape[1] - 1)
    baseline_model.initialize_weights()

    # Train Baseline model using the separate training function
    train_baseline_model(baseline_model, epochs=150, relative_abundance=X_clr_df, metadata=metadata, batch_size=64)
if __name__ == "__main__":
    main()
