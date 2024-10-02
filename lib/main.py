# main.py

import os
import pandas as pd
from data_processing import set_seed, load_and_transform_data
# from models import GAN, train_model
from benchmark_model import GAN, train_model

def main():
    """Main function to run the GAN training."""
    set_seed(42)

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Load and transform training data
    file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_train.csv'
    metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_train.csv'
    X_clr_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # Initialize GAN model
    gan = GAN(input_dim=X_clr_df.shape[1] - 1)
    gan.initialize_weights()

    # Train GAN model using the separate training function
    train_model(gan, epochs=150, relative_abundance=X_clr_df, metadata=metadata, batch_size=64)

    # Optional: Load and evaluate on test data
    # test_file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_test.csv'
    # test_metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_test.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # evaluate(gan, relative_abundance=X_clr_df_test, metadata=test_metadata,
    #          batch_size=test_metadata.shape[0], t='test', device=device)

if __name__ == "__main__":
    main()
