import os
import json
import optuna
import pandas as pd
import torch
import torch.nn as nn  # Added import for torch.nn
from data_processing import set_seed, load_and_transform_data
from models import GAN

def objective(trial, X_clr_df, metadata):
    """Objective function for Optuna hyperparameter optimization."""
    # Define hyperparameters to be optimized
    latent_dim = trial.suggest_categorical('latent_dim', [64, 128, 256])
    lr_r = trial.suggest_loguniform('lr_r', 1e-5, 1e-2)
    lr_g = trial.suggest_loguniform('lr_g', 1e-5, 1e-2)
    lr_c = trial.suggest_loguniform('lr_c', 1e-5, 1e-2)
    activation_fn = trial.suggest_categorical(
        'activation_fn', [nn.ReLU, nn.Tanh, nn.SiLU, nn.SELU, nn.LeakyReLU]
    )
    num_layers = trial.suggest_int('num_layers', 1, 5)
    
    # Initialize the model with the suggested parameters
    gan = GAN(
        input_dim=X_clr_df.shape[1] - 1,
        latent_dim=latent_dim,
        lr_r=lr_r,
        lr_g=lr_g,
        lr_c=lr_c,
        activation_fn=activation_fn,
        num_layers=num_layers
    )
    gan.initialize_weights()
    
    try:
        # Train the model
        eval_accuracy, eval_auc = gan.train_model(
            epochs=150,
            relative_abundance=X_clr_df,
            metadata=metadata,
            batch_size=64
        )
        
        # Report the evaluation AUC to Optuna
        return eval_auc
    
    except Exception as e:
        # Handle exceptions and report a high loss to Optuna
        print(f"An exception occurred: {e}")
        return float('inf')  # Or a very low score if direction='maximize'

def main():
    """Main function to run Optuna hyperparameter optimization."""
    set_seed(42)
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load and transform training data
    file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_train.csv'
    metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_train.csv'
    X_clr_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)
    
    # Create an Optuna study and optimize it
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_clr_df, metadata), n_trials=50)
    
    # Save the best trial to a file
    best_trial = study.best_trial
    best_trial_params = best_trial.params.copy()
    best_trial_params['activation_fn'] = best_trial_params['activation_fn'].__name__
    
    with open("best_trial.json", "w") as f:
        json.dump(best_trial_params, f, indent=4)
    
    print(f"Best trial saved: {best_trial_params}")

if __name__ == "__main__":
    main()
