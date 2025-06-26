# Re-import necessary packages after code state reset
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.preprocessing import LabelEncoder

# Upload paths to be filled after user provides data
microbiome_path = "dataset/MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv"
metadata_path = "dataset/MetaCardis_data/train_T2D_metadata.csv"

# Load data
microbiome_df = pd.read_csv(microbiome_path, index_col=0)
metadata_df = pd.read_csv(metadata_path)

# Set SampleID as index for metadata and align both datasets
metadata_df = metadata_df.set_index("SampleID")
metadata_df = metadata_df.loc[microbiome_df.index]

# Extract relevant columns
label = metadata_df["PATGROUPFINAL_C"]  # e.g., 0 = control, 1 = T2D
metformin = metadata_df["METFORMIN_C"]  # e.g., 0 = no, 1 = yes

# Ensure binary encoding
label = LabelEncoder().fit_transform(label)
metformin = LabelEncoder().fit_transform(metformin)

# === Test 1: Metformin vs. T2D (Label) using Chi-squared Test ===
contingency_table = pd.crosstab(label, metformin)
chi2, p_label, _, _ = chi2_contingency(contingency_table)

# === Test 2: Metformin vs. Microbiome Features ===
stats = []
p_values = []
features = []

for col in microbiome_df.columns:
    group0 = microbiome_df[metformin == 0][col]
    group1 = microbiome_df[metformin == 1][col]

    if group0.nunique() <= 1 and group1.nunique() <= 1:
        continue

    stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
    stats.append(stat)
    p_values.append(p)
    features.append(col)

# Save results
cor_df = pd.DataFrame({
    "Feature": features,
    "MannWhitneyU_Statistic": stats,
    "P-value": p_values
})
cor_df.to_csv("metformin_microbiome_association.csv", index=False)

# Output summary
print("\n--- Metformin Confounder Analysis Summary ---")
print(f"Chi-squared P-value (Metformin vs. T2D): {p_label:.4e}")
print(f"Significant Microbial Features (P < 0.05): {sum(np.array(p_values) < 0.05)}")
print(f"Total Features Tested: {len(features)}")
print("Result File: metformin_microbiome_association.csv")

