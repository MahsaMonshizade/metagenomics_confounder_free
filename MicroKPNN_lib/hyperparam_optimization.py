#!/usr/bin/env python3
import copy
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from config import config
from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Process
import os

# Use the device specified in the configuration.
device = torch.device(config["training"].get("device", "cpu"))

def build_mask(edge_list, species):
    # generate the mask
        edge_df = pd.read_csv(edge_list)
        
        edge_df['parent'] = edge_df['parent'].astype(str)
        parent_nodes = sorted(set(edge_df['parent'].tolist()))  # Sort to ensure consistent order
        mask = torch.zeros(len(species), len(parent_nodes))
        child_nodes = species

        parent_dict = {k: i for i, k in enumerate(parent_nodes)}
        child_dict = {k: i for i, k in enumerate(child_nodes)}
        
        for i, row in edge_df.iterrows():
            if row['child'] != 'Unnamed: 0': 
                mask[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1

        return mask.T, parent_dict

def run_trial(trial_config, num_epochs=10):
    """
    Run a single training trial using 5-fold cross-validation on the training data and return 
    the average validation accuracy across folds.

    This procedure:
      - Loads training data using the paths in the config.
      - Determines the feature columns dynamically (excluding metadata and SampleID).
      - Computes the input size from the training data.
      - Uses StratifiedKFold to split the data into 5 folds.
      - For each fold:
          • Prepares training and validation subsets.
          • Creates DataLoaders from the training and validation splits.
          • Computes class weights.
          • Builds a model (using hyperparameters from trial_config and with the chosen activation function).
          • Trains the model using your three-phase training routine.
          • Extracts the final validation accuracy for that fold.
      - Returns the average validation accuracy over all 5 folds.
    """
    # Extract configuration sections.
    data_cfg = trial_config["data"]
    train_cfg = trial_config["training"]
    model_cfg = trial_config["model"]

    # Load merged training data.
    merged_data_all, merged_test_data_all = get_data(data_cfg["train_abundance_path"], 
                                                    data_cfg["train_metadata_path"], 
                                                    data_cfg["test_abundance_path"], 
                                                    data_cfg["test_metadata_path"])
    
    # Get column names from config (must exist in the config file).
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]
    
    # Define feature columns (exclude metadata columns and 'SampleID').
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Compute the input size dynamically from the training data.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")
    
    # Overall disease classification.
    X = merged_data_all[feature_columns].values
    y_all = merged_data_all[disease_col].values  # Assumes the disease column is named "disease"
    
    # Set up 5-fold stratified cross-validation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_accuracies = []

    ###############added

    edge_list = f"Results/MicroKPNN_plots/required_data/EdgeList.csv"
    # Build masks
    
    mask, parent_dict = build_mask(edge_list, feature_columns)
    print(mask.shape)
    print(mask)
    parent_df = pd.DataFrame(list(parent_dict.items()), columns=['Parent', 'Index'])
    parent_dict_csv_path = "Results/MicroKPNN_plots/required_data/parent_dict_main.csv"
    parent_df.to_csv(parent_dict_csv_path, index=False)

    ########################
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Running fold {fold+1} of 5")
        
        # Split into training and validation subsets.
        train_data = merged_data_all.iloc[train_index]
        val_data   = merged_data_all.iloc[val_index]
        
        # Overall disease training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val   = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val   = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        
        batch_size = train_cfg["batch_size"]
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        
        # Compute class weights.
        num_pos = y_all_train.sum().item()
        num_neg = len(y_all_train) - num_pos
        pos_weight_value = num_neg / num_pos
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
        
        # Build the model using hyperparameters from trial_config (including activation).
        model = GAN(
            mask = mask,
            input_size=input_size,
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"],
            activation=model_cfg["activation"]
        ).to(device)
        
        # Define the loss function.
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        
        # Define the optimizer for the disease classification branch.
        optimizer = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )
        
        # Train the model using your training routine.
        # Here, we pass the same DataLoader as both 'val' and 'test', since we are using validation performance.
        Results = train_model(
            model, data_all_loader, data_all_val_loader, data_all_val_loader,
            num_epochs, criterion, optimizer, device
        )
        
        fold_val_acc = Results["val"]["accuracy"][-1]
        print(f"Fold {fold+1} validation accuracy: {fold_val_acc:.4f}")
        val_accuracies.append(fold_val_acc)
    
    avg_val_accuracy = np.mean(val_accuracies)
    print(f"Average validation accuracy over 5 folds: {avg_val_accuracy:.4f}")
    return avg_val_accuracy

def objective(trial):
    """
    Objective function with constraints on learning rates.
    """
    trial_config = copy.deepcopy(config)

    # Sample learning rates
    learning_rate = trial.suggest_categorical("learning_rate", config["tuning"]["learning_rate"])
    
    # Assign valid values to config
    trial_config["training"]["learning_rate"] = learning_rate
    
    # Run the trial
    avg_val_accuracy = run_trial(trial_config, num_epochs=100)
    return avg_val_accuracy

def worker_process(worker_id, study_name, storage_url, n_trials):
    """
    Worker function for parallel execution.
    Each worker loads the same study and runs optimization independently.
    """
    print(f"Worker {worker_id} starting with PID {os.getpid()}")
    
    try:
        # Load the existing study from shared storage
        study = optuna.load_study(
            study_name=study_name, 
            storage=storage_url
        )
        
        # Run optimization for specified number of trials
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Worker {worker_id} completed {n_trials} trials successfully!")
        
    except Exception as e:
        print(f"Worker {worker_id} encountered error: {e}")
        import traceback
        traceback.print_exc()

def run_parallel_optimization(n_workers=4, trials_per_worker=4, storage_file="optuna_study.db"):
    """
    Run parallel hyperparameter optimization using multiple processes.
    
    Args:
        n_workers: Number of parallel worker processes
        trials_per_worker: Number of trials each worker should run
        storage_file: SQLite database file to store study results
    """
    storage_url = f"sqlite:///{storage_file}"
    study_name = "microkpnn_cf_learning_rate_optimization"  # Changed study name for MicroKPNN-CF
    
    print(f"Starting parallel optimization with {n_workers} workers")
    print(f"Each worker will run {trials_per_worker} trials")
    print(f"Total trials: {n_workers * trials_per_worker}")
    print(f"Storage: {storage_file}")
    print("-" * 60)
    
    # Create the study once (this will create the database)
    try:
        study = optuna.create_study(
            direction="maximize",
            storage=storage_url,
            study_name=study_name,
            load_if_exists=True  # Don't fail if study already exists
        )
        print(f"Created/loaded study: {study_name}")
    except Exception as e:
        print(f"Error creating study: {e}")
        return
    
    # Launch parallel worker processes
    processes = []
    
    for worker_id in range(n_workers):
        p = Process(
            target=worker_process,
            args=(worker_id, study_name, storage_url, trials_per_worker)
        )
        p.start()
        processes.append(p)
        print(f"Launched worker {worker_id} (PID: {p.pid})")
    
    # Wait for all processes to complete
    print("\nWaiting for all workers to complete...")
    for i, p in enumerate(processes):
        p.join()
        print(f"Worker {i} finished")
    
    # Load final results and display best trial
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    
    try:
        final_study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        print(f"Total trials completed: {len(final_study.trials)}")
        print(f"Best trial:")
        print(f"  Final validation accuracy: {final_study.best_trial.value:.6f}")
        print(f"  Best hyperparameters:")
        for key, value in final_study.best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Show trial history
        print(f"\nTrial History:")
        for i, trial in enumerate(final_study.trials):  # Show all trials
            if trial.value is not None: 
                print(f"  Trial {trial.number}: {trial.value:.6f} - {trial.params}")
            else: 
                print(f"  Trial {trial.number}: PRUNED/FAILED - {trial.params}")
                
    except Exception as e:
        print(f"Error loading final results: {e}")

if __name__ == "__main__":
    # Configuration for parallel execution
    N_WORKERS = 4  # Adjust based on your CPU cores
    TRIALS_PER_WORKER = 4  # Each worker runs 4 trials
    STORAGE_FILE = "microkpnn_hyperparameter_optimization.db"  # Changed filename for MicroKPNN-CF
    if os.path.exists(STORAGE_FILE):
        os.remove(STORAGE_FILE)
        
    # For debugging/testing, you can run single-threaded:
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=20) # 14 trials in total
    # print("Best trial:", study.best_trial)
    
    # Run parallel optimization
    run_parallel_optimization(
        n_workers=N_WORKERS, 
        trials_per_worker=TRIALS_PER_WORKER,
        storage_file=STORAGE_FILE
    )
