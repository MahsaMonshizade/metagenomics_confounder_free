import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import GAN
from data_utils import get_data

if not hasattr(np, 'int'):
    np.int = int

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data and model files
train_abundance_path = "MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv"
train_metadata_path = "MetaCardis_data/train_T2D_metadata.csv"
feature_columns_path = "Results/FCNN_encoder_confounder_free_plots/feature_columns.csv"

# Load merged data
merged_data = get_data(train_abundance_path, train_metadata_path)

# Define feature columns
metadata_columns = pd.read_csv(train_metadata_path).columns.tolist()
feature_columns = pd.read_csv(feature_columns_path, header=None).squeeze("columns").astype(str).tolist()


X = merged_data[feature_columns].values
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# SHAP Explainer Function
def explain_model(model_paths, X):
    """
    Apply SHAP to explain the trained models over 5 folds.
    
    Args:
        model_paths (list): List of paths to the saved models.
        X (numpy.ndarray): Feature matrix.
    """
    # Initialize the explainer with a subset of the data
    sample_data = X  # Take a sample for the explainer to save computation
    
    for fold, model_path in enumerate(model_paths):
        print(f"Explaining model for Fold {fold + 1}")
        
        # Load the trained model
        model = GAN(input_size=X.shape[1], latent_dim=32, num_layers=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Define the prediction function for SHAP
        def predict(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                encoded_features = model.encoder(x_tensor)
                predictions = torch.sigmoid(model.disease_classifier(encoded_features))
            return predictions.cpu().numpy()
        
        # Create a SHAP KernelExplainer
        explainer = shap.KernelExplainer(predict, sample_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)  # Limit to 500 samples for efficiency
        
        # Plot SHAP summary
        shap.summary_plot(shap_values, X, feature_names=feature_columns, show=False)
        plt.savefig(f"Results/FCNN_encoder_confounder_free_plots/shap_summary_fold{fold + 1}.png")
        plt.close()
        
       # Extract the first element if shap_values is a list of arrays (for binary classification)
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_df = pd.DataFrame(shap_values[0], columns=feature_columns)
        else:
            shap_df = pd.DataFrame(shap_values, columns=feature_columns)
        shap_df.to_csv(f"Results/FCNN_encoder_confounder_free_plots/shap_values_fold{fold + 1}.csv", index=False)
        print(f"SHAP values saved for Fold {fold + 1}\n")

# List of trained model paths
model_paths = [
    f"Results/FCNN_encoder_confounder_free_plots/trained_model{fold + 1}.pth" for fold in range(5)
]

# Run SHAP explainability
explain_model(model_paths, X)
