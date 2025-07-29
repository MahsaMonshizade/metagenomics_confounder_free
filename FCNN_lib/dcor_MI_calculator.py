import torch
import numpy as np
import dcor
from sklearn.feature_selection import mutual_info_regression
from models import GAN
from config import config
from data_utils import get_data
import pandas as pd

# Setup
device = torch.device(config["training"]["device"])
model_cfg = config["model"]
data_cfg = config["data"]

# Load test data with CLR and scaling
_, merged_test_data = get_data(
    data_cfg["train_abundance_path"], data_cfg["train_metadata_path"],
    data_cfg["test_abundance_path"], data_cfg["test_metadata_path"]
)

# Extract feature columns
metadata_cols = pd.read_csv(data_cfg["test_metadata_path"]).columns.tolist()
feature_columns = [col for col in merged_test_data.columns if col not in metadata_cols and col != "SampleID"]

# Prepare test tensors
x_test = torch.tensor(merged_test_data[feature_columns].values, dtype=torch.float32).to(device)
y_confounder = merged_test_data[data_cfg["confounder_column"]].values.astype(float)  # METFORMIN_C

input_size = len(feature_columns)
latent_dim = model_cfg["latent_dim"]
n_folds = 5

# Store results
mi_scores = []
dcor_scores = []

for fold in range(1, n_folds + 1):
    print(f"\n🔍 Fold {fold}")

    # Load model
    model = GAN(
        input_size=input_size,
        latent_dim=latent_dim,
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_classifier_layers=model_cfg["num_classifier_layers"],
        dropout_rate=model_cfg["dropout_rate"],
        norm=model_cfg["norm"],
        classifier_hidden_dims=model_cfg["classifier_hidden_dims"],
        activation=model_cfg["activation"]
    ).to(device)

    model_path = f"Results/FCNN_plots/trained_model_fold{fold}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get encoder output
    with torch.no_grad():
        latent = model.encoder(x_test).cpu().numpy()

    # Mutual Information
    mi = mutual_info_regression(latent, y_confounder, random_state=42)
    avg_mi = np.mean(mi)
    mi_scores.append(mi)

    # Distance Correlation
    dcor_val = dcor.distance_correlation_sqr(latent, y_confounder.reshape(-1, 1))
    dcor_scores.append(dcor_val)

    print(f"  Average MI: {avg_mi:.4f}")
    print(f"  dCor²:      {dcor_val:.4f}")

# --- Summary ---
print("\n📊 Summary over all folds:")
for i in range(n_folds):
    print(f"Fold {i+1}: Avg MI = {np.mean(mi_scores[i]):.4f}, dCor² = {dcor_scores[i]:.4f}")

print(f"\n🧾 Overall Average MI:    {np.mean([np.mean(m) for m in mi_scores]):.4f}")
print(f"🧾 Overall Average dCor²: {np.mean(dcor_scores):.4f}")
