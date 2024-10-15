# main.py

import os
import pandas as pd
from data_processing import set_seed, load_and_transform_data
from models import GAN, train_model
import torch.nn as nn
import optuna

def main():
    """Main function to run the GAN training with hyperparameter optimization."""
    set_seed(42)

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Load and transform training data
    file_path = 'MetaCardis_data/train_T2D_abundance.csv'
    metadata_file_path = 'MetaCardis_data/train_T2D_metadata.csv'
    X_log_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # Define the objective function for hyperparameter optimization
    def objective(trial):
        # Suggest hyperparameters
        latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128])
        num_layers = trial.suggest_int('num_layers', 0, 3)
        activation_name = trial.suggest_categorical('activation_fn', ['ReLU', 'LeakyReLU', 'SiLU'])
        activation_fn = getattr(nn, activation_name)
        lr_r = trial.suggest_float('lr_r', 1e-5, 1e-3, log=True)
        lr_g = trial.suggest_float('lr_g', 1e-5, 1e-3, log=True)
        lr_c = trial.suggest_float('lr_c', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        # Initialize GAN model with suggested hyperparameters
        gan = GAN(
            input_dim=X_log_df.shape[1] - 1,
            latent_dim=latent_dim,
            activation_fn=activation_fn,
            num_layers=num_layers
        )
        gan.initialize_weights()

        # Train GAN model
        avg_eval_accuracy, avg_eval_auc = train_model(
            gan,
            epochs=50,  # You might want to reduce epochs during hyperparameter optimization
            relative_abundance=X_log_df,
            metadata=metadata,
            batch_size=batch_size,
            lr_r=lr_r,
            lr_g=lr_g,
            lr_c=lr_c
        )

        # Return the metric to optimize (e.g., validation AUC)
        return avg_eval_auc

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

    # Print best hyperparameters
    print('Best trial:')
    trial = study.best_trial
    print(f'  Validation AUC: {trial.value}')
    print('  Best hyperparameters:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == "__main__":
    main()
