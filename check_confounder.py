# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from scipy.stats import mannwhitneyu, spearmanr

# # Load microbiome and metadata from CSV-like input directly
# microbiome_path = "dataset/CRC_data/crc_abundance_PRJEB6070.csv"
# metadata_path = "dataset/CRC_data/crc_metadata_PRJEB6070.csv"

# microbiome_df = pd.read_csv(microbiome_path, index_col=0)
# metadata_df = pd.read_csv(metadata_path)

# # Ensure index alignment
# metadata_df = metadata_df.set_index("SampleID")
# metadata_df = metadata_df.loc[microbiome_df.index]

# # Get age and disease label
# age = metadata_df["BMI"]
# label = metadata_df["disease"]

# # Step 1: Age vs. Label association
# group0 = age[label == 0]
# group1 = age[label == 1]
# age_vs_label_p = mannwhitneyu(group0, group1).pvalue

# # Step 2: Age vs. each microbial feature (Spearman correlation)
# correlations = []
# p_values = []

# for col in microbiome_df.columns:
#     corr, p = spearmanr(age, microbiome_df[col])
#     correlations.append(corr)
#     p_values.append(p)

# # Step 3: Overall variance explained by age (using linear regression)
# X = age.values.reshape(-1, 1)
# Y = microbiome_df.values
# model = LinearRegression().fit(X, Y)
# Y_pred = model.predict(X)
# r2 = r2_score(Y, Y_pred)

# # Step 4: Permutation test
# n_perm = 1000
# perm_r2 = []
# for _ in range(n_perm):
#     perm_age = np.random.permutation(age)
#     perm_model = LinearRegression().fit(perm_age.reshape(-1, 1), Y)
#     perm_pred = perm_model.predict(perm_age.reshape(-1, 1))
#     perm_r2.append(r2_score(Y, perm_pred))
# p_perm = np.mean([r >= r2 for r in perm_r2])

# # Save Spearman correlation results to CSV
# cor_df = pd.DataFrame({
#     "Feature": microbiome_df.columns,
#     "SpearmanR": correlations,
#     "P-value": p_values
# })
# cor_df.to_csv("spearman_correlation_with_age.csv", index=False)
# print("✅ Saved Spearman correlation results to spearman_correlation_with_age.csv")

# # Print summary statistics
# print("\n--- Summary Statistics ---")
# print(f"Mann–Whitney U Test (Age vs. Disease Label): P = {age_vs_label_p}")
# print(f"Linear Regression R² (Age explains microbiome): {r2}")
# print(f"Permutation Test P-value: {p_perm}")




# Re-import necessary packages after code state reset
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.preprocessing import LabelEncoder

# Upload paths to be filled after user provides data
microbiome_path = "dataset/CRC_data/crc_abundance_PRJEB6070.csv"
metadata_path = "dataset/CRC_data/crc_metadata_PRJEB6070.csv"

# Load data
microbiome_df = pd.read_csv(microbiome_path, index_col=0)
metadata_df = pd.read_csv(metadata_path)

# Set SampleID as index for metadata and align both datasets
metadata_df = metadata_df.set_index("SampleID")
metadata_df = metadata_df.loc[microbiome_df.index]

# Extract relevant columns
label = metadata_df["disease"]  # e.g., 0 = control, 1 = T2D
metformin = metadata_df["sex"]  # e.g., 0 = no, 1 = yes

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
cor_df.to_csv("gender_microbiome_association.csv", index=False)

# Output summary
print("\n--- Gender Confounder Analysis Summary ---")
print(f"Chi-squared P-value (Gender vs. CRC): {p_label:.4e}")
print(f"Significant Microbial Features (P < 0.05): {sum(np.array(p_values) < 0.05)}")
print(f"Total Features Tested: {len(features)}")
print("Result File: gender_microbiome_association.csv")

