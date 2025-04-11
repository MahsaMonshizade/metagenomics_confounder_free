#!/usr/bin/env python3
import os
import copy
import itertools
import json
from hyperparam_optimization import run_trial
from config import config

# Define the hyperparameter search space from the tuning section.
search_space = {
    "num_encoder_layers": config["tuning"]["num_encoder_layers"],
    "num_classifier_layers": config["tuning"]["num_classifier_layers"],
    "dropout_rate": config["tuning"]["dropout_rate"],
    "learning_rate": config["tuning"]["learning_rate"],
    "encoder_lr": config["tuning"]["encoder_lr"],
    "classifier_lr": config["tuning"]["classifier_lr"],
    "activation": config["tuning"]["activation"]
}

# Create a directory to save grid search results.
result_root_dir = "grid_search_results"
os.makedirs(result_root_dir, exist_ok=True)

# Generate all combinations of hyperparameters.
hp_keys = list(search_space.keys())
hp_values = [search_space[key] for key in hp_keys]
hp_combinations = list(itertools.product(*hp_values))
print(f"Total hyperparameter combinations to try: {len(hp_combinations)}")

# Dictionary to hold overall results.
overall_results = {}

# Loop over each hyperparameter combination.
for combination in hp_combinations:
    # Build a dictionary for this combination.
    current_hparams = dict(zip(hp_keys, combination))
    
    # Create a unique directory name for this combination.
    # We replace periods with underscores for a safe directory name.
    dir_name = "_".join(f"{key}_{str(current_hparams[key]).replace('.', '_')}" for key in hp_keys)
    result_dir = os.path.join(result_root_dir, dir_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create a trial configuration by deep copying the base config.
    trial_config = copy.deepcopy(config)
    # Update trial_config with the current hyperparameters.
    trial_config["model"]["num_encoder_layers"] = current_hparams["num_encoder_layers"]
    trial_config["model"]["num_classifier_layers"] = current_hparams["num_classifier_layers"]
    trial_config["model"]["dropout_rate"] = current_hparams["dropout_rate"]
    trial_config["training"]["learning_rate"] = current_hparams["learning_rate"]
    trial_config["training"]["encoder_lr"] = current_hparams["encoder_lr"]
    trial_config["training"]["classifier_lr"] = current_hparams["classifier_lr"]
    trial_config["model"]["activation"] = current_hparams["activation"]

    print(f"Running combination: {current_hparams}")

    # Run the trial with a reduced number of epochs to save time.
    try:
        final_val_accuracy = run_trial(trial_config, num_epochs=10)
    except Exception as e:
        print(f"Combination {current_hparams} failed with error: {e}")
        final_val_accuracy = None

    print(f"Combination {current_hparams} achieved final_val_accuracy: {final_val_accuracy}")

    # Prepare result data.
    result_data = {
        "hyperparameters": current_hparams,
        "final_val_accuracy": final_val_accuracy
    }

    # Save result_data to a JSON file in the result directory.
    result_file_path = os.path.join(result_dir, "result.json")
    with open(result_file_path, "w") as f:
        json.dump(result_data, f, indent=4)

    overall_results[dir_name] = result_data

# Save an overall summary file.
summary_file = os.path.join(result_root_dir, "grid_search_summary.json")
with open(summary_file, "w") as f:
    json.dump(overall_results, f, indent=4)

print("Grid search completed.")
print(f"Results are saved in the '{result_root_dir}' directory.")
