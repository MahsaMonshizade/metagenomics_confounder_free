import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load data
relative_abundance = pd.read_csv('GMrepo_data/test_relative_abundance_IBD.csv')
metadata = pd.read_csv('GMrepo_data/test_metadata_IBD.csv')

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



# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_clr, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.float32)


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
test_dataset = AbundanceDataset(X_test_tensor, y_test_tensor)

# Create DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        # self.fc1 = nn.Linear(input_size, 512)
        # self.b1 = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(512, 16)
        # self.b2 = nn.BatchNorm1d(16)
        # self.fc3 = nn.Linear(16, 1)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.nn = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize Linear layers with He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            
            # Initialize BatchNorm layers
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        # x = self.relu(self.b1(self.fc1(x)))
        # x = self.dropout(x)
        # x = self.relu(self.b2(self.fc2(x)))
        # x = self.dropout(x)
        # x = self.fc3(x)
        x = self.nn(x)
        return x


# Initialize the model
input_size = X_test_tensor.shape[1]
model = SimpleNN(input_size)
model.load_state_dict(torch.load("best_model.pth"))
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=5e-5)

# Track metrics
test_accuracies = []
test_aucs = []
test_losses = []


# Validation loop
model.eval()
test_loss = 0.0
correct_test_predictions = 0
y_test_true = []
y_test_pred = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        labels = labels.unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
            
        predicted = torch.sigmoid(outputs).round()
        correct_test_predictions += (predicted == labels).sum().item()
            
        y_test_true.extend(labels.cpu().numpy())
        y_test_pred.extend(torch.sigmoid(outputs).cpu().numpy())
test_loss /= len(test_dataset)
test_accuracy = correct_test_predictions / len(test_dataset)
test_auc = roc_auc_score(y_test_true, y_test_pred)
    
test_accuracies.append(test_accuracy)
test_aucs.append(test_auc)
test_losses.append(test_loss)
    
print(f'test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.4f}, test AUC: {test_auc:.4f}\n')
