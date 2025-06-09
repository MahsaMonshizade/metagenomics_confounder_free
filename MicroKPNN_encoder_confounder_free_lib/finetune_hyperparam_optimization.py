#!/usr/bin/env python3
import copy
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from multiprocessing import Process
import os
import sys
from config import config
from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model
from sklearn.model_selection import StratifiedKFold

# Use the device as specified in the configuration.
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

def run_trial(trial_config, num_epochs=50):
    """
    Run a training trial using 5-fold cross-validation on the training set and return the average validation accuracy.

    This function:
      - Loads the training data from the paths in config.
      - Computes the feature columns and thus the input size dynamically.
      - Uses the column names for disease and confounder as given in config.
      - Splits the training data into 5 folds.
      - For each fold, trains the model (using the three-phase training routine) and evaluates the validation performance.
      - Returns the average validation accuracy (from the 'val' branch) across all folds.
    """
    # Extract configuration sections.
    data_cfg = trial_config["data"]
    train_cfg = trial_config["training"]
    model_cfg = trial_config["model"]
    # TMP: For loading the pre-trained model [fix it to somewhere as other paths].
    pretrained_model_path = "Results/MicroKPNN_encoder_confounder_free_pretraining/pretrained_model.pth"
    
    # Load merged training data. 
    merged_data_all, merged_test_data_all = get_data(data_cfg["train_abundance_path"], data_cfg["train_metadata_path"], data_cfg["test_abundance_path"], data_cfg["test_metadata_path"])
    
    # Get column names from config (must exist in the config file).
    disease_col = data_cfg["disease_column"]
    confounder_col = data_cfg["confounder_column"]
    
    # Determine feature columns (exclude metadata and SampleID).
    metadata_columns = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
    feature_columns = [col for col in merged_data_all.columns if col not in metadata_columns and col != "SampleID"]
    
    # Compute the input size dynamically.
    input_size = len(feature_columns)
    print(f"Determined input size: {input_size}")
    
    # Overall disease labels.
    X = merged_data_all[feature_columns].values
    merged_data_all['combined'] = (
        merged_data_all[disease_col].astype(str) +
        merged_data_all[confounder_col].astype(str)
    )
    y_all = merged_data_all["combined"].values
    
    # Prepare to aggregate validation accuracies.
    fold_val_accuracies = []
    
    # Set up 5-fold stratified cross-validation on the training data.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ###############added - Build masks for MicroKPNN
    edge_list = "./Results/MicroKPNN_encoder_confounder_free_plots/required_data/EdgeList.csv"
    # Build masks
    mask, parent_dict = build_mask(edge_list, feature_columns)
    print(mask.shape)
    print(mask)
    parent_df = pd.DataFrame(list(parent_dict.items()), columns=['Parent', 'Index'])
    # parent_dict_csv_path = "parent_dict_main.csv"
    # parent_df.to_csv(parent_dict_csv_path, index=False)
    ########################
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Running fold {fold+1} of 5")
        # Split into training and validation subsets.
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]
        
        # Overall disease training data.
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val = torch.tensor(val_data[disease_col].values, dtype=torch.float32).unsqueeze(1)
        
        # Confounder (drug/sex) training data: use only samples where disease == 1.
        train_data_disease = train_data[train_data[disease_col] == 1]
        val_data_disease = val_data[val_data[disease_col] == 1]
        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        x_disease_val = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_val = torch.tensor(val_data_disease[confounder_col].values, dtype=torch.float32).unsqueeze(1)
        
        batch_size = train_cfg["batch_size"]
        
        # Create stratified DataLoaders.
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        
        # Compute class weights.
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)
        
        num_pos_drug = y_disease_train.sum().item()
        num_neg_drug = len(y_disease_train) - num_pos_drug
        pos_weight_value_drug = num_neg_drug / num_pos_drug
        pos_weight_drug = torch.tensor([pos_weight_value_drug], dtype=torch.float32).to(device)
        
        # Build the model - INCLUDING MASK for MicroKPNN
        model = GAN(
            mask=mask,  # Added mask parameter for MicroKPNN
            input_size=input_size,
            latent_dim=model_cfg["latent_dim"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_classifier_layers=model_cfg["num_classifier_layers"],
            dropout_rate=model_cfg["dropout_rate"],
            norm=model_cfg["norm"],
            classifier_hidden_dims=model_cfg["classifier_hidden_dims"],
            activation=model_cfg["activation"]
        ).to(device)
        
        # Define loss functions.
        criterion = PearsonCorrelationLoss().to(device)
        criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drug).to(device)
        criterion_disease_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_disease).to(device)
        
        # Define optimizers.
        optimizer = optim.AdamW(model.encoder.parameters(), lr=train_cfg["encoder_lr"], weight_decay=train_cfg["weight_decay"])
        optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=train_cfg["classifier_lr"], weight_decay=train_cfg["weight_decay"])
        optimizer_disease_classifier = optim.AdamW(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()),
            lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
        )
        
        # Train the model using the three-phase training routine.
        # Here, both the confounder branch and disease branch are trained.
        Results = train_model(
            model, criterion, optimizer,
            data_loader, data_all_loader, data_val_loader, data_all_val_loader,
            data_val_loader, data_all_val_loader, num_epochs,
            criterion_classifier, optimizer_classifier,
            criterion_disease_classifier, optimizer_disease_classifier,
            device,
            pretrained_model_path, 
        )
        
        # Extract the final validation accuracy from this fold.
        fold_val_acc = Results["val"]["accuracy"][-1]
        print(f"Fold {fold+1} validation accuracy: {fold_val_acc:.4f}")
        fold_val_accuracies.append(fold_val_acc)
    
    # Compute the average validation accuracy over all folds.
    avg_val_accuracy = np.mean(fold_val_accuracies)
    print(f"Average validation accuracy over 5 folds: {avg_val_accuracy:.4f}")
    return avg_val_accuracy

def objective(trial):
    """
    Objective function with constraints on learning rates.
    """
    trial_config = copy.deepcopy(config)
    
    # Sample learning rates
    learning_rate = trial.suggest_categorical("learning_rate", config["tuning"]["learning_rate"])
    encoder_lr = trial.suggest_categorical("encoder_lr", config["tuning"]["encoder_lr"])
    classifier_lr = trial.suggest_categorical("classifier_lr", config["tuning"]["classifier_lr"])
    
    # Apply constraints - prune invalid combinations
    if learning_rate >= encoder_lr or learning_rate >= classifier_lr:
        raise optuna.exceptions.TrialPruned()
    
    # Assign valid values to config
    trial_config["training"]["learning_rate"] = learning_rate
    trial_config["training"]["encoder_lr"] = encoder_lr
    trial_config["training"]["classifier_lr"] = classifier_lr
    
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
    study_name = "learning_rate_optimization"
    
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
    STORAGE_FILE = "microkpnn_cf_ft_hyperparameter_optimization.db"
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