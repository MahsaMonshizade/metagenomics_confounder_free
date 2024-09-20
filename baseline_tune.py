import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json


# Load data
relative_abundance = pd.read_csv('GMrepo_data/train_relative_abundance_IBD_balanced.csv')
metadata = pd.read_csv('GMrepo_data/train_metadata_IBD_balanced.csv')

# Map disease labels to numeric values
disease_dict = {'D006262': 0, 'D043183': 1}
metadata['disease_numeric'] = metadata['disease'].map(disease_dict)

# Separate features and labels
X = relative_abundance.drop(columns=['loaded_uid']).values  # Drop sample IDs
y = metadata['disease_numeric'].values

# Adding a small pseudocount to avoid log(0)
pseudocount = 1e-6
X += pseudocount

# CLR Transformation
def clr_transformation(X):
    geometric_mean = np.exp(np.mean(np.log(X), axis=1))
    clr_data = np.log(X / geometric_mean[:, np.newaxis])
    return clr_data

X_clr = clr_transformation(X)

# Stratified split into train, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X_clr, y, test_size=0.3, stratify=y, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create a custom dataset
class AbundanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Instantiate the datasets
train_dataset = AbundanceDataset(X_train_tensor, y_train_tensor)
val_dataset = AbundanceDataset(X_val_tensor, y_val_tensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    # batch_size = trial.suggest_int('batch_size', 32, 128, step=32)
    possible_batch_sizes = [32, 64, 128, 256]
    batch_size = trial.suggest_categorical('batch_size', possible_batch_sizes)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)

    # Tune number of layers and hidden nodes
    num_layers = trial.suggest_int('num_layers', 1, 5)  # Number of hidden layers
    hidden_units = []
    input_size = X_train_tensor.shape[1]

    # Hidden layer sizes should be one of [2048, 1024, 512, 256, 128, 64, 32, 16]
    possible_hidden_units = [2048, 1024, 512, 256, 128, 64, 32, 16]
    
    for i in range(num_layers):
        hidden_units.append(trial.suggest_categorical(f'n_units_l{i}', possible_hidden_units))
    
    # for i in range(num_layers):
    #     hidden_units.append(trial.suggest_int(f'n_units_l{i}', 64, 1024, step=64))  # Hidden layer sizes

    # Create the model dynamically based on the number of layers and units
    class TunedNN(nn.Module):
        def __init__(self, input_size, hidden_units, dropout_rate):
            super(TunedNN, self).__init__()
            layers = []
            in_features = input_size
            for units in hidden_units:
                layers.append(nn.Linear(in_features, units))
                layers.append(nn.BatchNorm1d(units))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_features = units
            layers.append(nn.Linear(in_features, 1))  # Output layer
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Initialize model with dynamic layers and units
    model = TunedNN(input_size, hidden_units, dropout_rate).to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_train_true, y_train_pred = [], []

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            y_train_true.extend(labels.cpu().numpy())
            y_train_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

        # Validation loop
        model.eval()
        val_loss = 0.0
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                y_val_true.extend(labels.cpu().numpy())
                y_val_pred.extend(torch.sigmoid(outputs).cpu().numpy())

        val_auc = roc_auc_score(y_val_true, y_val_pred)

        # Report validation AUC to Optuna
        trial.report(val_auc, epoch)

        # Stop early if the validation AUC does not improve for several epochs
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_auc

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

# Print the best trial
print('Best trial:')
trial = study.best_trial

print(f'AUC: {trial.value}')
print('Best hyperparameters: {}'.format(trial.params))

# Save best hyperparameters to a file
best_params = trial.params
with open('best_hyperparameters_baseline.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print("Best hyperparameters saved to 'best_hyperparameters.json'")
