import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models import GAN
from config import config
from data_utils import get_data
import os

# Setup
device = torch.device(config["training"]["device"])
model_cfg = config["model"]
data_cfg = config["data"]

# Load test data
_, test_data = get_data(
    data_cfg["train_abundance_path"], data_cfg["train_metadata_path"],
    data_cfg["test_abundance_path"], data_cfg["test_metadata_path"]
)

# Extract columns
metadata_cols = pd.read_csv(data_cfg["train_metadata_path"]).columns.tolist()
feature_cols = [col for col in test_data.columns if col not in metadata_cols and col != "SampleID"]
label_col = data_cfg["disease_column"]
confounder_col = data_cfg["confounder_column"]

# Filter only label=1 samples
subset = test_data[test_data[label_col] == 1].copy()
x_subset = torch.tensor(subset[feature_cols].values, dtype=torch.float32).to(device)
y_confounder = subset[confounder_col].values.astype(int)  # METFORMIN_C

# Load model (you can loop over folds if needed)
model = GAN(
    input_size=len(feature_cols),
    latent_dim=model_cfg["latent_dim"],
    num_encoder_layers=model_cfg["num_encoder_layers"],
    num_classifier_layers=model_cfg["num_classifier_layers"],
    dropout_rate=model_cfg["dropout_rate"],
    norm=model_cfg["norm"],
    classifier_hidden_dims=model_cfg["classifier_hidden_dims"],
    activation=model_cfg["activation"]
).to(device)

model_path = "Results/FCNN_encoder_confounder_free_plots/trained_model_fold2.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Get encoder outputs
with torch.no_grad():
    latent = model.encoder(x_subset).cpu().numpy()

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(latent)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(latent)

# Ensure output directory exists
output_dir = "Results/FCNN_encoder_confounder_free_plots"
os.makedirs(output_dir, exist_ok=True)

# Plotting helper
def plot_and_save(embedding, title, confounder_labels, filename):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=confounder_labels, cmap="coolwarm", alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    legend1 = plt.legend(*scatter.legend_elements(), title="Metformin")
    plt.gca().add_artist(legend1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# Generate and save figures
plot_and_save(pca_result, "PCA (Encoder Output, Label=1)", y_confounder, "pca_label1_metformin.png")
plot_and_save(tsne_result, "t-SNE (Encoder Output, Label=1)", y_confounder, "tsne_label1_metformin.png")
