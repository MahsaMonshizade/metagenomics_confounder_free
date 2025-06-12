### https://medium.com/statistics-in-machine-learning/comparing-roc-curves-in-machine-learning-model-with-delongs-test-a-practical-guide-using-python-e70b5d20abde

import numpy as np
import scipy.stats

def Delong_test(true, prob_A, prob_B):
    """
    Perform DeLong's test for comparing the AUCs of two models.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True binary labels in range {0, 1}.
    prob_A : array-like of shape (n_samples,)
        Predicted probabilities by the first model.
    prob_B : array-like of shape (n_samples,)
        Predicted probabilities by the second model.

    Returns
    -------
    z_score : float
        The z score from comparing the AUCs of two models.
    p_value : float
        The p value from comparing the AUCs of two models.

    Example
    -------
    >>> true = [0, 1, 0, 1]
    >>> prob_A = [0.1, 0.4, 0.35, 0.8]
    >>> prob_B = [0.2, 0.3, 0.4, 0.7]
    >>> z_score, p_value = Delong_test(true, prob_A, prob_B)
    >>> print(f"Z-Score: {z_score}, P-Value: {p_value}")
    """

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def compute_ground_truth_statistics(true):
        assert np.array_equal(np.unique(true), [0, 1]), "Ground truth must be binary."
        order = (-true).argsort()
        label_1_count = int(true.sum())
        return order, label_1_count

    # Prepare data
    order, label_1_count = compute_ground_truth_statistics(np.array(true))
    sorted_probs = np.vstack((np.array(prob_A), np.array(prob_B)))[:, order]

    # Fast DeLong computation starts here
    m = label_1_count  # Number of positive samples
    n = sorted_probs.shape[1] - m  # Number of negative samples
    k = sorted_probs.shape[0]  # Number of models (2)

    # Initialize arrays for midrank computations
    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        positive_examples = sorted_probs[r, :m]
        negative_examples = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(examples) for examples in [positive_examples, negative_examples, sorted_probs[r, :]]
        ]

    # Calculate AUCs
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    # Compute variance components
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    # Compute covariance matrices
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # Calculating z-score and p-value
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T)).flatten()
    p_value = scipy.stats.norm.sf(abs(z)) * 2

    z_score = -z[0].item()
    p_value = p_value[0].item()

    return z_score, p_value

# =====================
# Load Data and Run Test
# =====================
if __name__ == "__main__":
    # Edit these file paths
    file1_path = "Results/FCNN_encoder_confounder_free_plots/test_results.npz"
    file2_path = "Results/FCNN_plots/test_results.npz"

    # Keys in your npz files
    prob_key = 'pred_probs'
    label_key = 'labels'
    id_key = 'sample_ids'

    # Load npz data
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)

    # Match sample IDs
    ids1 = data1[id_key]
    ids2 = data2[id_key]

    if isinstance(ids1[0], bytes): ids1 = [x.decode('utf-8') for x in ids1]
    if isinstance(ids2[0], bytes): ids2 = [x.decode('utf-8') for x in ids2]

    ids1 = np.array(ids1)
    ids2 = np.array(ids2)

    common_ids = np.intersect1d(ids1, ids2)
    common_ids_sorted = sorted(common_ids)

    id_to_idx1 = {id_: i for i, id_ in enumerate(ids1)}
    id_to_idx2 = {id_: i for i, id_ in enumerate(ids2)}

    indices1 = [id_to_idx1[i] for i in common_ids_sorted]
    indices2 = [id_to_idx2[i] for i in common_ids_sorted]

    y_true = data1[label_key][indices1]
    prob1 = data1[prob_key][indices1]
    prob2 = data2[prob_key][indices2]

    y_true = np.asarray(y_true).astype(int).flatten()
    prob1 = np.asarray(prob1).astype(float).flatten()
    prob2 = np.asarray(prob2).astype(float).flatten()

    # print(y_true)
    # print("prob1")
    # print(prob1)
    # print("prob2")
    # print(prob2)

    # true = [0, 1, 0, 1]
    # prob_A = [0.1, 0.4, 0.35, 0.8]
    # prob_B = [0.2, 0.3, 0.4, 0.7]
    print(f"Running DeLong test on {len(y_true)} matched samples...")
    z_score, p_value = Delong_test(true, prob_B, prob_A)
    print(f"\nDeLong test results:")
    print(f"Z-score = {z_score:.4f}")
    print(f"P-value = {p_value:.6f}")
