import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

def match_samples_by_id(data1, data2, prob_key='pred_probs', label_key='labels', id_key='sample_ids'):
    """
    Match and reorder samples between two datasets based on sample IDs.
    
    Parameters:
    data1: dict-like, first dataset (e.g., loaded .npz file)
    data2: dict-like, second dataset
    prob_key: str, key for predicted probabilities
    label_key: str, key for true labels
    id_key: str, key for sample IDs
    
    Returns:
    tuple: (matched_data1, matched_data2, common_ids) where matched data contains only common samples in same order
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
    
    # Find common sample IDs
    common_ids = np.intersect1d(ids1, ids2)
    print(f"Total samples in model 1: {len(ids1)}")
    print(f"Total samples in model 2: {len(ids2)}")
    print(f"Common samples found: {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common sample IDs found between the two datasets!")
    
    # Create masks for common samples
    mask1 = np.isin(ids1, common_ids)
    mask2 = np.isin(ids2, common_ids)
    
    # Extract common samples
    common_ids1 = ids1[mask1]
    common_ids2 = ids2[mask2]
    
    # Create sorting indices to match order
    # Sort both by the common IDs to ensure same order
    sort_idx1 = np.argsort(common_ids1)
    sort_idx2 = np.argsort(common_ids2)
    
    # Get indices in original arrays
    original_indices1 = np.where(mask1)[0][sort_idx1]
    original_indices2 = np.where(mask2)[0][sort_idx2]
    
    # Extract and reorder data
    matched_data1 = {
        prob_key: data1[prob_key][original_indices1],
        label_key: data1[label_key][original_indices1],
        id_key: ids1[original_indices1]
    }
    
    matched_data2 = {
        prob_key: data2[prob_key][original_indices2],
        label_key: data2[label_key][original_indices2],
        id_key: ids2[original_indices2]
    }
    
    # Verify matching worked
    assert np.array_equal(matched_data1[id_key], matched_data2[id_key]), "Sample ID matching failed!"
    
    return matched_data1, matched_data2, common_ids
    
def calculate_f1_per_class(y_true, y_pred, average='macro'):
    """
    Calculate F1 score with different averaging methods.
    
    Parameters:
    y_true: array-like, true labels
    y_pred: array-like, predicted labels
    average: str, averaging method ('macro', 'micro', 'weighted', 'binary', or None)
    
    Returns:
    float or array: F1 score(s)
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def bootstrap_f1_scores(y_true, y_pred1, y_pred2, n_bootstrap=1000, random_state=42):
    """
    Generate bootstrap samples of F1 scores for two models.
    
    Parameters:
    y_true: array-like, true labels
    y_pred1: array-like, predictions from model 1
    y_pred2: array-like, predictions from model 2
    n_bootstrap: int, number of bootstrap samples
    random_state: int, random seed for reproducibility
    
    Returns:
    tuple: (f1_scores_model1, f1_scores_model2)
    """
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    f1_scores_1 = []
    f1_scores_2 = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Calculate F1 scores for bootstrap sample
        f1_1 = calculate_f1_per_class(y_true[indices], y_pred1[indices])
        f1_2 = calculate_f1_per_class(y_true[indices], y_pred2[indices])
        
        f1_scores_1.append(f1_1)
        f1_scores_2.append(f1_2)
    
    return np.array(f1_scores_1), np.array(f1_scores_2)

def bootstrap_f1_test(y_true, y_pred1, y_pred2, n_bootstrap=1000, random_state=42, 
                     f1_average='macro', alpha=0.05):
    """
    Perform bootstrap test on F1 score differences between two models.
    
    Parameters:
    y_true: array-like, true labels
    y_pred1: array-like, predictions from model 1
    y_pred2: array-like, predictions from model 2
    n_bootstrap: int, number of bootstrap samples
    random_state: int, random seed for reproducibility
    f1_average: str, F1 averaging method ('macro', 'micro', 'weighted', 'binary')
    alpha: float, significance level (default 0.05)
    
    Returns:
    dict: Bootstrap test results
    """
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    f1_scores_1 = []
    f1_scores_2 = []
    differences = []
    
    # Original F1 scores
    original_f1_1 = calculate_f1_per_class(y_true, y_pred1, average=f1_average)
    original_f1_2 = calculate_f1_per_class(y_true, y_pred2, average=f1_average)
    original_diff = original_f1_1 - original_f1_2
    
    print(f"Generating {n_bootstrap} bootstrap samples...")
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Calculate F1 scores for bootstrap sample
        f1_1 = calculate_f1_per_class(y_true[indices], y_pred1[indices], average=f1_average)
        f1_2 = calculate_f1_per_class(y_true[indices], y_pred2[indices], average=f1_average)
        
        f1_scores_1.append(f1_1)
        f1_scores_2.append(f1_2)
        differences.append(f1_1 - f1_2)
    
    f1_scores_1 = np.array(f1_scores_1)
    f1_scores_2 = np.array(f1_scores_2)
    differences = np.array(differences)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(differences, (alpha/2) * 100)
    ci_upper = np.percentile(differences, (1 - alpha/2) * 100)
    
    # Bootstrap p-value (two-tailed test)
    # Null hypothesis: no difference (difference = 0)
    p_value = 2 * min(
        np.mean(differences >= 0),  # P(diff >= 0)
        np.mean(differences <= 0)   # P(diff <= 0)
    )
    
    # Effect size (Cohen's d approximation)
    pooled_std = np.sqrt((np.var(f1_scores_1) + np.var(f1_scores_2)) / 2)
    cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0
    
    return {
        'original_f1_1': original_f1_1,
        'original_f1_2': original_f1_2,
        'original_diff': original_diff,
        'bootstrap_f1_1_mean': np.mean(f1_scores_1),
        'bootstrap_f1_1_std': np.std(f1_scores_1),
        'bootstrap_f1_2_mean': np.mean(f1_scores_2),
        'bootstrap_f1_2_std': np.std(f1_scores_2),
        'bootstrap_diff_mean': np.mean(differences),
        'bootstrap_diff_std': np.std(differences),
        'bootstrap_diff_median': np.median(differences),
        'confidence_interval': (ci_lower, ci_upper),
        'p_value': p_value,
        'cohens_d': cohens_d,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha,
        'f1_scores_1': f1_scores_1,
        'f1_scores_2': f1_scores_2,
        'differences': differences
    }



def compare_f1_scores(file1_path, file2_path, model1_name="Model 1", model2_name="Model 2",
                        prob_key='pred_probs', label_key='labels', id_key='sample_ids',
                        n_bootstrap=1000, f1_average='macro', alpha=0.05, random_state=42):
    """
    Compare F1 scores between two models using Bootstrap Test.
    
    Parameters:
    file1_path: str, path to first model results (.npz file)
    file2_path: str, path to second model results (.npz file)
    model1_name: str, name for first model
    model2_name: str, name for second model
    prob_key: str, key for predicted probabilities
    label_key: str, key for true labels
    id_key: str, key for sample IDs
    n_bootstrap: int, number of bootstrap samples
    f1_average: str, F1 averaging method ('macro', 'micro', 'weighted', 'binary')
    alpha: float, significance level
    random_state: int, random seed for reproducibility
    
    Returns:
    dict: Bootstrap test results
    """
    
    print("Loading data...")
    # Load and match data (using your existing function)
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)
    
    # Use your existing match_samples_by_id function
    matched_data1, matched_data2, common_ids = match_samples_by_id(
        data1, data2, prob_key, label_key, id_key
    )
    
    # Extract matched data
    y_true = matched_data1[label_key]
    
    # Get predictions
    if matched_data1[prob_key].ndim == 2:
        y_pred1 = np.argmax(matched_data1[prob_key], axis=1)
    else:
        y_pred1 = (matched_data1[prob_key] > 0.5).astype(int)
        
    if matched_data2[prob_key].ndim == 2:
        y_pred2 = np.argmax(matched_data2[prob_key], axis=1)
    else:
        y_pred2 = (matched_data2[prob_key] > 0.5).astype(int)
    
    print("Performing Bootstrap Test...")
    # Perform bootstrap test
    bootstrap_results = bootstrap_f1_test(
        y_true, y_pred1, y_pred2, 
        n_bootstrap=n_bootstrap, 
        random_state=random_state,
        f1_average=f1_average, 
        alpha=alpha
    )
    
    # Print detailed results
    print("\n" + "="*80)
    print("F1 SCORE BOOTSTRAP TEST RESULTS")
    print("="*80)
    
    print(f"\n  SAMPLE INFORMATION:")
    print(f"  Total matched samples: {len(y_true)}")
    print(f"  Bootstrap samples: {n_bootstrap}")
    print(f"  F1 averaging method: {f1_average}")
    print(f"  Significance level (α): {alpha}")
    
    print(f"\n  ORIGINAL F1 SCORES (Single Test Set):")
    print(f"  {model1_name}: {bootstrap_results['original_f1_1']:.4f}")
    print(f"  {model2_name}: {bootstrap_results['original_f1_2']:.4f}")
    print(f"  Difference: {bootstrap_results['original_diff']:.4f}")
    
    print(f"\n  BOOTSTRAP F1 STATISTICS:")
    print(f"  {model1_name}:")
    print(f"    Mean: {bootstrap_results['bootstrap_f1_1_mean']:.4f}")
    print(f"    Std:  {bootstrap_results['bootstrap_f1_1_std']:.4f}")
    print(f"  {model2_name}:")
    print(f"    Mean: {bootstrap_results['bootstrap_f1_2_mean']:.4f}")
    print(f"    Std:  {bootstrap_results['bootstrap_f1_2_std']:.4f}")
    
    print(f"\n  BOOTSTRAP DIFFERENCE STATISTICS:")
    print(f"  Mean difference: {bootstrap_results['bootstrap_diff_mean']:.4f}")
    print(f"  Median difference: {bootstrap_results['bootstrap_diff_median']:.4f}")
    print(f"  Std of differences: {bootstrap_results['bootstrap_diff_std']:.4f}")
    print(f"  {(1-alpha)*100:.1f}% Confidence Interval: [{bootstrap_results['confidence_interval'][0]:.4f}, {bootstrap_results['confidence_interval'][1]:.4f}]")
    
    print(f"\n  BOOTSTRAP TEST RESULTS:")
    print(f"  P-value: {bootstrap_results['p_value']:.6f}")
    print(f"  Cohen's d (effect size): {bootstrap_results['cohens_d']:.4f}")
    
    # Interpret effect size
    abs_cohens_d = abs(bootstrap_results['cohens_d'])
    if abs_cohens_d < 0.2:
        effect_interpretation = "negligible"
    elif abs_cohens_d < 0.5:
        effect_interpretation = "small"
    elif abs_cohens_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"  Effect size interpretation: {effect_interpretation}")
    
    is_significant = bootstrap_results['p_value'] < alpha
    print(f"  Result: {'Significant' if is_significant else 'Not significant'} (α = {alpha})")
    
    # Check if confidence interval contains zero
    ci_contains_zero = (bootstrap_results['confidence_interval'][0] <= 0 <= 
                       bootstrap_results['confidence_interval'][1])
    
    print(f"\n  CONFIDENCE INTERVAL INTERPRETATION:")
    if ci_contains_zero:
        print(f"  The {(1-alpha)*100:.1f}% confidence interval CONTAINS zero")
        print(f"  This suggests NO significant difference between models")
    else:
        print(f"  The {(1-alpha)*100:.1f}% confidence interval does NOT contain zero")
        print(f"  This suggests a significant difference between models")
    
    print(f"\n  SUMMARY:")
    if bootstrap_results['bootstrap_diff_mean'] > 0:
        better_model = model1_name
        worse_model = model2_name
    else:
        better_model = model2_name
        worse_model = model1_name
    
    print(f"  Better performing model: {better_model}")
    print(f"  Absolute mean F1 difference: {abs(bootstrap_results['bootstrap_diff_mean']):.4f}")
    
    if is_significant:
        print(f"  Statistical significance: YES")
        print(f"  Conclusion: {better_model} performs significantly better than {worse_model}")
    else:
        print(f"  Statistical significance: NO")
        print(f"  Conclusion: No significant difference in F1 scores detected")
    
    # Add bootstrap test specific insights
    print(f"\n  BOOTSTRAP INSIGHTS:")
    print(f"  Proportion of bootstrap samples where {model1_name} > {model2_name}: {np.mean(bootstrap_results['differences'] > 0):.3f}")
    print(f"  Proportion of bootstrap samples where {model2_name} > {model1_name}: {np.mean(bootstrap_results['differences'] < 0):.3f}")
    print(f"  Proportion of bootstrap samples with tie: {np.mean(bootstrap_results['differences'] == 0):.3f}")
    
    return {
        'bootstrap_results': bootstrap_results,
        'matched_samples': len(y_true),
        'is_significant': is_significant,
        'better_model': better_model,
        'ci_contains_zero': ci_contains_zero
    }

# Example usage with your existing code structure
if __name__ == "__main__":
    # Compare F1 scores between two models using Wilcoxon test
    results = compare_f1_scores(
        file1_path="Results/FCNN_encoder_confounder_free_plots/test_results.npz",
        file2_path="Results/FCNN_plots/test_results.npz",
        model1_name="FCNN Encoder Confounder Free",
        model2_name="FCNN",
        n_bootstrap=1000,
        f1_average='macro',  # or 'micro', 'weighted', 'binary'
    )