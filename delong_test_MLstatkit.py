import numpy as np
from MLstatkit.stats import Delong_test
from sklearn.metrics import roc_auc_score

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

    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)
    sign = np.sign(auc1 - auc2)
    print(sign)
    
    print(f"Running DeLong test on {len(y_true)} matched samples...")
    z_score, p_value = Delong_test(y_true, prob1, prob2)
    z_score = z_score * sign
    print(f"\nDeLong test results:")
    print(f"Z-score = {z_score:.4f}")
    print(f"P-value = {p_value:.6f}")
