# Comparing two models on the same dataset using DeLong's test for AUC
# e.g. prob1, label1, sampleid1: Results/FCNN_encoder_confounder_free_plots/test_results.npz
#      prob2, labe2, sampleid2: Results/FCNN_plots/test_results.npz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats
import pandas as pd
from itertools import combinations

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single classifier using DeLong et al. method.
    
    Parameters:
    ground_truth: array-like, true binary labels (0 or 1)
    predictions: array-like, predicted probabilities
    
    Returns:
    float: AUC variance
    """
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    
    # Separate positive and negative samples
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    
    pos_scores = predictions[pos_idx]
    neg_scores = predictions[neg_idx]
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    # Compute pairwise comparisons
    pos_matrix = np.tile(pos_scores, (n_neg, 1)).T
    neg_matrix = np.tile(neg_scores, (n_pos, 1))
    
    # V10: Indicator matrix for positive > negative
    V10 = (pos_matrix > neg_matrix).astype(float)
    
    # V01: Indicator matrix for positive < negative  
    V01 = (pos_matrix < neg_matrix).astype(float)
    
    # Structural components
    S10 = np.sum(V10, axis=1) / n_neg  # For each positive sample
    S01 = np.sum(V01, axis=0) / n_pos  # For each negative sample
    
    # AUC estimate
    auc_est = np.sum(V10) / (n_pos * n_neg)
    
    # Variance components
    var_pos = np.var(S10, ddof=1) if n_pos > 1 else 0
    var_neg = np.var(S01, ddof=1) if n_neg > 1 else 0
    
    # Total variance
    variance = var_pos / n_pos + var_neg / n_neg
    
    return variance

def delong_roc_test(ground_truth, predictions1, predictions2):
    """
    Performs DeLong's test for comparing two ROC curves.
    
    Parameters:
    ground_truth: array-like, true binary labels (0 or 1)
    predictions1: array-like, predicted probabilities from model 1
    predictions2: array-like, predicted probabilities from model 2
    
    Returns:
    dict: Contains AUC values, difference, variance, z-score, and p-value
    """
    ground_truth = np.asarray(ground_truth)
    predictions1 = np.asarray(predictions1)
    predictions2 = np.asarray(predictions2)
    
    # Calculate AUC for both models
    auc1 = roc_auc_score(ground_truth, predictions1)
    auc2 = roc_auc_score(ground_truth, predictions2)
    auc_diff = auc1 - auc2
    
    # Separate positive and negative samples
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    if n_pos == 0 or n_neg == 0:
        return {
            'auc1': auc1,
            'auc2': auc2,
            'auc_diff': auc_diff,
            'variance': 0,
            'z_score': 0,
            'p_value': 1.0,
            'se': 0
        }
    
    # Get scores for positive and negative samples
    pos_scores1 = predictions1[pos_idx]
    neg_scores1 = predictions1[neg_idx]
    pos_scores2 = predictions2[pos_idx]
    neg_scores2 = predictions2[neg_idx]
    
    # Compute structural components for covariance
    pos_matrix1 = np.tile(pos_scores1, (n_neg, 1)).T
    neg_matrix1 = np.tile(neg_scores1, (n_pos, 1))
    pos_matrix2 = np.tile(pos_scores2, (n_neg, 1)).T
    neg_matrix2 = np.tile(neg_scores2, (n_pos, 1))
    
    # V10 matrices
    V10_1 = (pos_matrix1 > neg_matrix1).astype(float)
    V10_2 = (pos_matrix2 > neg_matrix2).astype(float)
    
    # V01 matrices
    V01_1 = (pos_matrix1 < neg_matrix1).astype(float)
    V01_2 = (pos_matrix2 < neg_matrix2).astype(float)
    
    # Structural components
    S10_1 = np.sum(V10_1, axis=1) / n_neg
    S01_1 = np.sum(V01_1, axis=0) / n_pos
    S10_2 = np.sum(V10_2, axis=1) / n_neg
    S01_2 = np.sum(V01_2, axis=0) / n_pos
    
    # Covariance computation
    if n_pos > 1:
        cov_pos = np.cov(S10_1, S10_2, ddof=1)[0, 1]
    else:
        cov_pos = 0
        
    if n_neg > 1:
        cov_neg = np.cov(S01_1, S01_2, ddof=1)[0, 1]
    else:
        cov_neg = 0
    
    # Individual variances
    var1 = delong_roc_variance(ground_truth, predictions1)
    var2 = delong_roc_variance(ground_truth, predictions2)
    
    # Variance of the difference
    var_diff = var1 + var2 - 2 * (cov_pos / n_pos + cov_neg / n_neg)
    
    # Ensure variance is non-negative
    var_diff = max(var_diff, 1e-10)
    
    # Z-score and p-value
    se_diff = np.sqrt(var_diff)
    z_score = auc_diff / se_diff if se_diff > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        'auc1': auc1,
        'auc2': auc2,
        'auc_diff': auc_diff,
        'variance': var_diff,
        'z_score': z_score,
        'p_value': p_value,
        'se': se_diff
    }

def match_samples_by_id(data1, data2, prob_key='pred_probs', label_key='labels', id_key='sample_ids'):
    """
    Match and reorder samples between two datasets based on sample IDs.
    Deduplicates by keeping only the first occurrence of each ID.
    """
    
    # Get sample IDs
    ids1 = data1[id_key]
    ids2 = data2[id_key]
    
    # Convert to strings if needed for consistent comparison
    if isinstance(ids1[0], bytes):
        ids1 = [id_.decode('utf-8') for id_ in ids1]
    if isinstance(ids2[0], bytes):
        ids2 = [id_.decode('utf-8') for id_ in ids2]
    
    ids1 = np.array(ids1)
    ids2 = np.array(ids2)
    
    print(f"Total samples in model 1: {len(ids1)}")
    print(f"Total samples in model 2: {len(ids2)}")
    
    # Deduplicate by keeping first occurrence of each ID
    _, unique_indices1 = np.unique(ids1, return_index=True)
    _, unique_indices2 = np.unique(ids2, return_index=True)
    
    # Sort indices to maintain original order for first occurrences
    unique_indices1 = np.sort(unique_indices1)
    unique_indices2 = np.sort(unique_indices2)
    
    # Extract deduplicated data
    dedup_ids1 = ids1[unique_indices1]
    dedup_ids2 = ids2[unique_indices2]
    dedup_probs1 = data1[prob_key][unique_indices1]
    dedup_probs2 = data2[prob_key][unique_indices2]
    dedup_labels1 = data1[label_key][unique_indices1]
    dedup_labels2 = data2[label_key][unique_indices2]
    
    print(f"After deduplication - Model 1: {len(dedup_ids1)} unique samples")
    print(f"After deduplication - Model 2: {len(dedup_ids2)} unique samples")
    
    # Find common sample IDs after deduplication
    common_ids = np.intersect1d(dedup_ids1, dedup_ids2)
    print(f"Common samples found: {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common sample IDs found between the two datasets!")
    
    # Create masks for common samples in deduplicated data
    mask1 = np.isin(dedup_ids1, common_ids)
    mask2 = np.isin(dedup_ids2, common_ids)
    
    # Extract common samples from deduplicated data
    common_ids1 = dedup_ids1[mask1]
    common_ids2 = dedup_ids2[mask2]
    common_probs1 = dedup_probs1[mask1]
    common_probs2 = dedup_probs2[mask2]
    common_labels1 = dedup_labels1[mask1]
    common_labels2 = dedup_labels2[mask2]
    
    # Create sorting indices to match order
    sort_idx1 = np.argsort(common_ids1)
    sort_idx2 = np.argsort(common_ids2)
    
    # Extract and reorder matched data
    matched_data1 = {
        prob_key: common_probs1[sort_idx1],
        label_key: common_labels1[sort_idx1],
        id_key: common_ids1[sort_idx1]
    }
    
    matched_data2 = {
        prob_key: common_probs2[sort_idx2],
        label_key: common_labels2[sort_idx2],
        id_key: common_ids2[sort_idx2]
    }
    
    # Verify matching worked
    assert np.array_equal(matched_data1[id_key], matched_data2[id_key]), "Sample ID matching failed!"
    
    print(f"Final matched samples: {len(matched_data1[id_key])}")
    
    return matched_data1, matched_data2, common_ids

def compare_auc_delong(file1_path, file2_path, model1_name="Model 1", model2_name="Model 2", 
                      prob_key='pred_probs', label_key='labels', id_key='sample_ids',
                      save_plots=True, plot_path="./tmp.png"):
    """
    Compare AUC between two models using DeLong's test with proper sample ID matching.
    
    Parameters:
    file1_path: str, path to first model results (.npz file)
    file2_path: str, path to second model results (.npz file)
    model1_name: str, name for first model
    model2_name: str, name for second model
    prob_key: str, key for predicted probabilities in npz files
    label_key: str, key for true labels in npz files
    id_key: str, key for sample IDs in npz files
    save_plots: bool, whether to save plots
    plot_path: str, path to save results
    
    Returns:
    dict: Comprehensive AUC comparison results using DeLong's test
    """
    
    print("Loading data...")
    # Load data
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)
    
    print("Matching samples by ID...")
    # Match samples by ID
    matched_data1, matched_data2, common_ids = match_samples_by_id(
        data1, data2, prob_key, label_key, id_key
    )
    
    # Extract matched data
    y_true1 = matched_data1[label_key]
    y_true2 = matched_data2[label_key]
    matched_data1[prob_key] = np.squeeze(matched_data1[prob_key]) # TMP: ensure probabilities are 1D
    matched_data2[prob_key] = np.squeeze(matched_data2[prob_key]) # TMP: ensure probabilities are 1D

    # Verify labels are identical (they should be for same test set)
    if not np.array_equal(y_true1, y_true2):
        print("Warning: True labels differ between datasets. Using labels from first model.")
    y_true = y_true1
    
    # Check if binary classification
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError(f"DeLong's test requires binary classification. Found {len(unique_labels)} classes: {unique_labels}")
    
    # Ensure labels are 0 and 1
    if not set(unique_labels) == {0, 1}:
        print(f"Converting labels from {unique_labels} to [0, 1]")
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        y_true = np.array([label_map[label] for label in y_true])
    
    # Get probabilities
    if matched_data1[prob_key].ndim == 2:
        # Multi-class probabilities - use positive class probability
        if matched_data1[prob_key].shape[1] == 2:
            y_prob1 = matched_data1[prob_key][:, 1]  # Probability of positive class
        else:
            raise ValueError("For multi-class problems with >2 classes, specify which class to use for AUC")
    else:
        # Binary probabilities
        y_prob1 = matched_data1[prob_key]
        
    if matched_data2[prob_key].ndim == 2:
        # Multi-class probabilities - use positive class probability
        if matched_data2[prob_key].shape[1] == 2:
            y_prob2 = matched_data2[prob_key][:, 1]  # Probability of positive class
        else:
            raise ValueError("For multi-class problems with >2 classes, specify which class to use for AUC")
    else:
        # Binary probabilities
        y_prob2 = matched_data2[prob_key]
    
    print("Performing DeLong's test...")
    # Perform DeLong's test
    delong_results = delong_roc_test(y_true, y_prob1, y_prob2)
    
    # Calculate ROC curves
    fpr1, tpr1, _ = roc_curve(y_true, y_prob1)
    fpr2, tpr2, _ = roc_curve(y_true, y_prob2)
    
    # Create visualization with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # ROC Curves
    ax1.plot(fpr1, tpr1, color='blue', lw=2, label=f'{model1_name} (AUC = {delong_results["auc1"]:.3f})')
    ax1.plot(fpr2, tpr2, color='red', lw=2, label=f'{model2_name} (AUC = {delong_results["auc2"]:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=13)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Combined Sample Information and DeLong Test Results
    ax2.axis('off')
    
    # Sample information
    n_samples = len(y_true)
    pos_samples = np.sum(y_true)
    neg_samples = n_samples - pos_samples
    
    # Calculate confidence interval
    ci_lower = delong_results['auc_diff'] - 1.96 * delong_results['se']
    ci_upper = delong_results['auc_diff'] + 1.96 * delong_results['se']
    
    # Determine significance level
    if delong_results['p_value'] < 0.001:
        significance = "*** (p < 0.001)"
    elif delong_results['p_value'] < 0.01:
        significance = "** (p < 0.01)"
    elif delong_results['p_value'] < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "ns (not significant)"
    
    # Determine better model
    if delong_results['auc_diff'] > 0:
        better_model = model1_name
    else:
        better_model = model2_name
    
    # Create combined text
    combined_text = "SAMPLE INFORMATION\n"
    combined_text += "=" * 30 + "\n"
    combined_text += f"Total Samples: {n_samples}\n"
    combined_text += f"Positive Samples: {pos_samples} ({pos_samples/n_samples:.1%})\n"
    combined_text += f"Negative Samples: {neg_samples} ({neg_samples/n_samples:.1%})\n"
    combined_text += f"Common Sample IDs: {len(common_ids)}\n\n"
    
    combined_text += "DELONG'S TEST RESULTS\n"
    combined_text += "=" * 30 + "\n"
    combined_text += f"{model1_name}: {delong_results['auc1']:.4f}\n"
    combined_text += f"{model2_name}: {delong_results['auc2']:.4f}\n"
    combined_text += f"AUC Difference: {delong_results['auc_diff']:.4f}\n"
    combined_text += f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
    combined_text += f"Standard Error: {delong_results['se']:.4f}\n"
    combined_text += f"Z-score: {delong_results['z_score']:.4f}\n"
    combined_text += f"P-value: {delong_results['p_value']:.6f}\n"
    combined_text += f"Significance: {significance}\n\n"
    combined_text += f"Better Model: {better_model}\n"
    combined_text += f"Absolute Difference: {abs(delong_results['auc_diff']):.4f}"
    
    ax2.text(0.05, 1.00, combined_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             # bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5),
             bbox=None)
    
    plt.tight_layout()
    
    if save_plots: 
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"AUC comparison plot saved to: {plot_path}")
    
    plt.show()
    
    # Print detailed results
    print("\n" + "="*80)
    print("DELONG'S TEST FOR AUC COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n  SAMPLE INFORMATION:")
    print(f"  Total matched samples: {n_samples}")
    print(f"  Positive samples: {pos_samples} ({pos_samples/n_samples:.3f})")
    print(f"  Negative samples: {neg_samples} ({neg_samples/n_samples:.3f})")
    print(f"  Common sample IDs: {len(common_ids)}")
    
    print(f"\n  AUC RESULTS:")
    print(f"  {model1_name}: {delong_results['auc1']:.4f}")
    print(f"  {model2_name}: {delong_results['auc2']:.4f}")
    print(f"  Difference: {delong_results['auc_diff']:.4f}")
    print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\n  DELONG'S TEST:")
    print(f"  Z-statistic: {delong_results['z_score']:.4f}")
    print(f"  Standard Error: {delong_results['se']:.4f}")
    print(f"  P-value: {delong_results['p_value']:.6f}")
    
    if delong_results['p_value'] < 0.001:
        significance_level = "p < 0.001"
    elif delong_results['p_value'] < 0.01:
        significance_level = "p < 0.01"
    elif delong_results['p_value'] < 0.05:
        significance_level = "p < 0.05"
    else:
        significance_level = "p ≥ 0.05"
    
    delong_significant = "significant" if delong_results['p_value'] < 0.05 else "not significant"
    print(f"  Significance level: {significance_level}")
    print(f"  Result: AUC difference is {delong_significant}")
    
    print(f"\n  SUMMARY:")
    if delong_results['auc_diff'] > 0:
        better_model = model1_name
        worse_model = model2_name
    else:
        better_model = model2_name
        worse_model = model1_name
    
    print(f"  Better performing model: {better_model}")
    print(f"  Absolute AUC difference: {abs(delong_results['auc_diff']):.4f}")
    
    if delong_results['p_value'] < 0.05:
        print(f"  Statistical significance: YES")
        print(f"  Conclusion: {better_model} has significantly better AUC than {worse_model}")
    else:
        print(f"  Statistical significance: NO")
        print(f"  Conclusion: No significant difference in AUC detected")
    
    return {
        'delong_results': delong_results,
        'roc_curve1': (fpr1, tpr1),
        'roc_curve2': (fpr2, tpr2),
        'matched_samples': len(common_ids),
        'common_ids': common_ids,
        'confidence_interval': (ci_lower, ci_upper)
    }

# Example usage
if __name__ == "__main__":
    # Compare AUC between two models using DeLong's test
    results = compare_auc_delong(
        file1_path="Results/FCNN_encoder_confounder_free_plots/test_results.npz",
        file2_path="Results/FCNN_plots/test_results.npz",
        model1_name="FCNN Encoder Confounder Free",
        model2_name="FCNN",
        save_plots=True,
        plot_path="Results/tmp1.png"
    )

    # Compare AUC between two models using DeLong's test
    results = compare_auc_delong(
        file2_path="Results/FCNN_encoder_confounder_free_plots/test_results.npz",
        file1_path="Results/FCNN_plots/test_results.npz",
        model2_name="FCNN Encoder Confounder Free",
        model1_name="FCNN",
        save_plots=True,
        plot_path="Results/tmp2.png"
    )