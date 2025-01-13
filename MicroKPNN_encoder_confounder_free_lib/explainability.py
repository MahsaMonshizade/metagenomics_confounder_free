#########################
# explainability.py
#########################

import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from models import GAN, MaskedLinear
from data_utils import get_data

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# -------- PATHS TO DATA & MODEL --------
#########################################
train_abundance_path = "MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv"
train_metadata_path = "MetaCardis_data/train_T2D_metadata.csv"
feature_columns_path = "Results/MicroKPNN_encoder_confounder_free_plots/feature_columns.csv"
mapping_file_path = "Default_Database/species_ids.csv"  # Adjust if needed

#########################################
# --- LOAD & PREPARE DATA + FEATURES ----
#########################################
# 1) Merged data
merged_data = get_data(train_abundance_path, train_metadata_path)

# 2) Feature columns
metadata_columns = pd.read_csv(train_metadata_path).columns.tolist()
feature_columns = pd.read_csv(feature_columns_path, header=None).squeeze("columns").astype(str).tolist()

# 3) (Optional) Map feature names for visualization
mapping_df = pd.read_csv(mapping_file_path)
taxon_to_species = dict(zip(mapping_df['taxon_id'].astype(str), mapping_df['species']))
visual_feature_columns = [taxon_to_species.get(col, col) for col in feature_columns]

# 4) Ensure feature_columns exist in merged_data
merged_data.columns = merged_data.columns.astype(str)  # unify types
feature_columns = [col for col in feature_columns if col in merged_data.columns]

# 5) Prepare the feature matrix
X = merged_data[feature_columns].values
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

#########################################
# ------------- BUILD MASK -------------
#########################################
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
    


edge_list = "Default_Database/EdgeList.csv"
relative_abundance = pd.read_csv(train_abundance_path, index_col=0)
species = relative_abundance.columns.values.tolist()
mask, parent_dict = build_mask(edge_list, species)
df = pd.DataFrame(list(parent_dict.items()), columns=["key", "value"])
df.to_csv("Results/MicroKPNN_encoder_confounder_free_plots/parent_dict.csv", index=False)

#########################################
# -------- MODEL WRAPPER (FEATURE) ------
#########################################
class ModelWrapper(torch.nn.Module):
    """
    Wraps the entire model to get final disease predictions 
    from input features X.
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        encoded_features = self.model.encoder(x)
        predictions = self.model.disease_classifier(encoded_features)
        return predictions

#########################################
# --------- EXPLAIN MODEL (FEATURE) -----
# (Already in your script; SHAP on input)
#########################################
def explain_model(model_paths, X):
    """
    Existing function that applies SHAP to the *input features*.
    """
    sample_data_tensor = torch.tensor(X, dtype=torch.float32).to(device) 

    for fold, model_path in enumerate(model_paths):
        print(f"Explaining model (input features) for Fold {fold + 1}")

        # Load trained model
        base_model = GAN(mask=mask, latent_dim=64, num_layers=1).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        # Wrap model
        wrapped_model = ModelWrapper(base_model)

        # Create SHAP explainer
        explainer = shap.DeepExplainer(wrapped_model, sample_data_tensor)

        # Calculate SHAP values
        shap_values = explainer.shap_values(sample_data_tensor)

        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        # Summary Plot
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=visual_feature_columns, 
            plot_type="bar", 
            show=False
        )
        plt.title(f"SHAP (Features) - Fold {fold + 1}")
        plt.tight_layout()
        plt.savefig(f"Results/MicroKPNN_encoder_confounder_free_plots/shap_summary_fold{fold+1}.png")
        plt.close()

        # Bee Swarm Plot
        plt.figure(figsize=(15, 10))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=visual_feature_columns, 
            plot_type="dot", 
            show=False
        )
        plt.title(f"SHAP Bee Swarm (Features) - Fold {fold+1}")
        plt.tight_layout()
        plt.savefig(f"Results/MicroKPNN_encoder_confounder_free_plots/shap_bee_swarm_fold{fold+1}.png")
        plt.close()

        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=visual_feature_columns)
        shap_df.to_csv(
            f"Results/MicroKPNN_encoder_confounder_free_plots/shap_values_fold{fold+1}.csv", 
            index=False
        )
        print(f"SHAP values (features) saved for Fold {fold + 1}\n")

#########################################
# -- EXPLAIN FIRST HIDDEN LAYER (NEW) ---
#########################################

def get_first_hidden_activations(model, x):
    """
    Forward pass x through the *first hidden layer*:
     - index 0: MaskedLinear
     - index 1: BatchNorm
     - index 2: ReLU
    Returns activations => shape: [batch_size, hidden_dim].
    """
    out = model.encoder[0](x)   # MaskedLinear
    out = model.encoder[1](out) # BatchNorm
    out = model.encoder[2](out) # ReLU
    return out

class SubModel(torch.nn.Module):
    """
    A sub-model that takes first-layer activations as input,
    then runs the *rest* of the encoder + disease_classifier.
    """
    def __init__(self, original_model):
        super(SubModel, self).__init__()
        # everything after the first hidden layer in model.encoder
        self.post_first_layer = torch.nn.Sequential(*list(original_model.encoder[3:]))

        # final disease classifier
        self.disease_classifier = original_model.disease_classifier

    def forward(self, h):
        x = self.post_first_layer(h)
        logits = self.disease_classifier(x)
        return logits

def explain_first_hidden_layer(model_paths, X, 
                               device=device, 
                               background_size=100, 
                               output_dir="Results/MicroKPNN_encoder_confounder_free_plots"):
    """
    Uses SHAP to explain *first hidden-layer node importance* for final disease output.
    1) We run X through the first hidden layer (MaskedLinear + BN + ReLU).
    2) We build a SubModel (the rest of encoder + disease_classifier).
    3) We treat those hidden activations as 'features' and apply SHAP.

    model_paths: list of str
        e.g. [ 'trained_model1.pth', 'trained_model2.pth', ... ]
    X: numpy.ndarray (N x input_features)
        Original input data to feed into the first hidden layer.
    device: str
        "cpu" or "cuda"
    background_size: int
        Number of random samples used as SHAP background.
    output_dir: str
        Directory to store SHAP plots & CSV for each fold.
    """
    sample_data_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    for fold, model_path in enumerate(model_paths):
        print(f"\n=== Explaining FIRST-HIDDEN-LAYER for Fold {fold + 1} ===")

        # 1) Load the trained model
        base_model = GAN(mask=mask, latent_dim=64, num_layers=1).to(device)
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        base_model.eval()

        # 2) Get the first-layer activations for *all* samples
        with torch.no_grad():
            hidden_activations = get_first_hidden_activations(base_model, sample_data_tensor)
        # shape => [N, d_hidden]

        # 3) Build sub-model that maps hidden_activations -> disease logits
        sub_model = SubModel(base_model).to(device)
        sub_model.eval()

        # 4) Random background set for SHAP
        N = hidden_activations.shape[0]
        idx = np.random.choice(N, size=min(background_size, N), replace=False)
        background = hidden_activations[idx]

        # 5) Create SHAP explainer
        explainer = shap.DeepExplainer(sub_model, background)

        # 6) Compute SHAP values for all hidden_activations
        shap_values = explainer.shap_values(hidden_activations)
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]

        shap_values_np = shap_values
        hidden_activations_np = hidden_activations.cpu().numpy()

        # 7) Node names
        d_hidden = shap_values_np.shape[1]
        node_names = [f"Node_{i}" for i in range(d_hidden)]

        # --- Plot (bar) ---
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_np, 
            hidden_activations_np, 
            feature_names=node_names, 
            plot_type="bar", 
            show=False
        )
        plt.title(f"SHAP Bar (First Hidden Layer) - Fold {fold+1}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary_firstlayer_fold{fold+1}.png")
        plt.close()

        # --- Plot (bee swarm) ---
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_np, 
            hidden_activations_np, 
            feature_names=node_names, 
            plot_type="dot", 
            show=False
        )
        plt.title(f"SHAP Bee Swarm (First Hidden Layer) - Fold {fold+1}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_bee_swarm_firstlayer_fold{fold+1}.png")
        plt.close()

        # 8) Save SHAP values to CSV
        shap_df = pd.DataFrame(shap_values_np, columns=node_names)
        csv_path = f"{output_dir}/shap_values_first_hidden_fold{fold+1}.csv"
        shap_df.to_csv(csv_path, index=False)

        print(f"Saved first-layer SHAP CSV to: {csv_path}")

#########################################
# EXAMPLE USAGE (uncomment if needed)
#########################################
if __name__ == "__main__":
    model_paths = [
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model1.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model2.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model3.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model4.pth",
        "Results/MicroKPNN_encoder_confounder_free_plots/trained_model5.pth"
    ]
    
    # 1) SHAP on input features
    explain_model(model_paths, X)
    
    # 2) SHAP on first hidden layer
    explain_first_hidden_layer(model_paths, X, device=device)         