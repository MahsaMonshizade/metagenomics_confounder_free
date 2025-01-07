import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import GAN
from data_utils import get_data
from models import MaskedLinear


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data and model files
train_abundance_path = "MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv"
train_metadata_path = "MetaCardis_data/train_T2D_metadata.csv"
feature_columns_path = "Results/MicroKPNN_encoder_confounder_free_plots/feature_columns.csv"
mapping_file_path = "Default_Database/species_ids.csv"  # Update with the correct path

# Load merged data
merged_data = get_data(train_abundance_path, train_metadata_path)

# Define feature columns
metadata_columns = pd.read_csv(train_metadata_path).columns.tolist()
feature_columns = pd.read_csv(feature_columns_path, header=None).squeeze("columns").astype(str).tolist()

# Load the mapping file
mapping_df = pd.read_csv(mapping_file_path)

# Create a mapping dictionary from the mapping file
taxon_to_species = dict(zip(mapping_df['taxon_id'].astype(str), mapping_df['species']))

# Replace numeric feature names in feature_columns for visualization purposes only
visual_feature_columns = [taxon_to_species.get(col, col) for col in feature_columns]

# Ensure feature columns match merged data
merged_data.columns = merged_data.columns.astype(str)
feature_columns = [col for col in feature_columns if col in merged_data.columns]

# Prepare the feature matrix
X = merged_data[feature_columns].values
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

def build_mask(edge_list, species):
    edge_df = pd.read_csv(edge_list)
    edge_df['parent'] = edge_df['parent'].astype(str)
    parent_nodes = list(set(edge_df['parent'].tolist()))
    mask = torch.zeros(len(species), len(parent_nodes))
    parent_dict = {k: i for i, k in enumerate(parent_nodes)}
    child_dict = {k: i for i, k in enumerate(species)}
    for i, row in edge_df.iterrows():
        if row['child'] != 'Unnamed: 0': 
            mask[child_dict[str(row['child'])]][parent_dict[row['parent']]] = 1
    return mask.T

# Initialize mask
edge_list = "Default_Database/EdgeList.csv"
relative_abundance = pd.read_csv(train_abundance_path, index_col=0)
species = relative_abundance.columns.values.tolist()
mask = build_mask(edge_list, species)

# Model Wrapper Class
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        encoded_features = self.model.encoder(x)
        predictions = self.model.disease_classifier(encoded_features)
        return predictions

# SHAP Explainer Function
def explain_model(model_paths, X):
    """
    Apply SHAP to explain the trained models over 5 folds.

    Args:
        model_paths (list): List of paths to the saved models.
        X (numpy.ndarray): Feature matrix (NumPy array).
    """
    sample_data_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Use a subset for SHAP

    for fold, model_path in enumerate(model_paths):
        print(f"Explaining model for Fold {fold + 1}")

        # Load the trained model
        print(mask.device) 
        base_model = GAN(mask=mask , latent_dim=64, num_layers=1).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        # Wrap the base model
        wrapped_model = ModelWrapper(base_model)

        # Create SHAP DeepExplainer
        explainer = shap.DeepExplainer(wrapped_model, sample_data_tensor)

        # Calculate SHAP values
        shap_values = explainer.shap_values(sample_data_tensor)

        # Convert SHAP values to NumPy array for visualization
        shap_values_np = shap_values[0] if isinstance(shap_values, list) else shap_values

         # Summary Plot
        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
        shap.summary_plot(shap_values_np, X, feature_names=visual_feature_columns, plot_type="bar", show=False)
        plt.xticks(fontsize=10)  # Adjust font size for x-axis
        plt.yticks(fontsize=10)  # Adjust font size for y-axis
        plt.tight_layout()       # Ensure layout fits
        plt.savefig(f"Results/MicroKPNN_encoder_confounder_free_plots/shap_summary_fold{fold + 1}.png")
        plt.close()

        # Bee Swarm Plot
        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
        shap.summary_plot(shap_values_np, X, feature_names=visual_feature_columns, plot_type="dot", show=False)
        plt.xticks(fontsize=10, rotation=45)  # Rotate x-axis labels if needed
        plt.yticks(fontsize=10)               # Adjust font size for y-axis
        plt.tight_layout()                    # Ensure layout fits
        plt.savefig(f"Results/MicroKPNN_encoder_confounder_free_plots/shap_bee_swarm_fold{fold + 1}.png")
        plt.close()

        # Save SHAP values as CSV
        shap_df = pd.DataFrame(shap_values_np, columns=visual_feature_columns)
        shap_df.to_csv(f"Results/MicroKPNN_encoder_confounder_free_plots/shap_values_fold{fold + 1}.csv", index=False)
        print(f"SHAP values and plots saved for Fold {fold + 1}\n")

# List of trained model paths
model_paths = [
    f"Results/MicroKPNN_encoder_confounder_free_plots/trained_model{fold + 1}.pth" for fold in range(5)
]

# Run SHAP explainability
explain_model(model_paths, X)
