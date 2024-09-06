import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load data
relative_abundance = pd.read_csv('GMrepo_data/train_relative_abundance_IBD_v3.csv')
metadata = pd.read_csv('GMrepo_data/train_metadata_IBD_v3.csv')

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

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.b1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.b2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.b3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.b1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.b2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.b3(self.fc3(x)))
        x = self.fc4(x)
        # x =  self.sigmoid(self.fc4(x))  # No Sigmoid here for BCEWithLogitsLoss
        return x

# Initialize the model
input_size = X_train_tensor.shape[1]
model = SimpleNN(input_size)

# Loss and optimizer
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization

# Track metrics
train_accuracies, val_accuracies = [], []
train_aucs, val_aucs = [], []
train_losses, val_losses = [], []

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    correct_predictions = 0
    running_loss = 0.0
    y_train_true = []
    y_train_pred = []
    
    for inputs, labels in train_dataloader:
        labels = labels.unsqueeze(1)  # Adjust shape for BCEWithLogitsLoss
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predicted = torch.sigmoid(outputs).round()  # Apply sigmoid and round for binary prediction
        correct_predictions += (predicted == labels).sum().item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        y_train_true.extend(labels.cpu().numpy())
        y_train_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())
    
    # Calculate metrics for training
    epoch_loss = running_loss / len(train_dataset)
    accuracy = correct_predictions / len(train_dataset)
    auc = roc_auc_score(y_train_true, y_train_pred)
    
    train_accuracies.append(accuracy)
    train_aucs.append(auc)
    train_losses.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train AUC: {auc:.4f}')

    if epoch == 49:
        print('hiii')
        print(labels)
        print('pred')
        print(predicted)
        print('bye')

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val_predictions = 0
    y_val_true = []
    y_val_pred = []
    
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            labels = labels.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            predicted = torch.sigmoid(outputs).round()
            correct_val_predictions += (predicted == labels).sum().item()
            
            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(torch.sigmoid(outputs).cpu().numpy())

    val_loss /= len(val_dataset)
    val_accuracy = correct_val_predictions / len(val_dataset)
    val_auc = roc_auc_score(y_val_true, y_val_pred)
    
    val_accuracies.append(val_accuracy)
    val_aucs.append(val_auc)
    val_losses.append(val_loss)
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation AUC: {val_auc:.4f}')

# Plotting training and validation metrics
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, train_aucs, 'b', label='Train AUC')
plt.plot(epochs, val_aucs, 'r', label='Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('AUC over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('performance.png')
