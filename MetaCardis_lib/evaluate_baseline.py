# evaluate_baseline.py

import os
import torch
import pandas as pd
from data_processing import set_seed, load_and_transform_data, preprocess_metadata, create_batch
from baseline_model import BaselineModel
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import numpy as np
import json

def evaluate_baseline():
    """Load saved baseline models and evaluate them on the test dataset."""
    set_seed(42)
    
    # Paths to test data
    test_file_path = 'MetaCardis_data/test_T2D_abundance.csv'
    test_metadata_file_path = 'MetaCardis_data/test_T2D_metadata.csv'
    
    # Load and transform test data
    X_clr_df_test = load_and_transform_data(test_file_path)
    test_metadata = pd.read_csv(test_metadata_file_path)
    
    # Preprocess metadata
    test_metadata = preprocess_metadata(test_metadata)
    
    # Ensure models directory exists
    models_dir = 'baseline_models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")
    
    # Get list of saved models
    encoder_files = sorted([f for f in os.listdir(models_dir) if f.startswith('encoder_fold')])
    classifier_files = sorted([f for f in os.listdir(models_dir) if f.startswith('disease_classifier_fold')])
    
    if not encoder_files or not classifier_files:
        raise FileNotFoundError("Saved model files not found in the 'baseline_models' directory.")
    
    # Initialize lists to store evaluation metrics
    accuracies = []
    aucs = []
    f1_scores = []
    
    for encoder_file, classifier_file in zip(encoder_files, classifier_files):
        # Load the saved models
        print(f"Loading models: {encoder_file}, {classifier_file}")
        encoder_state_dict = torch.load(os.path.join(models_dir, encoder_file))
        classifier_state_dict = torch.load(os.path.join(models_dir, classifier_file))
        
        # Initialize model architecture
        input_dim = X_clr_df_test.shape[1] - 1
        baseline_model = BaselineModel(input_dim=input_dim)
        baseline_model.encoder.load_state_dict(encoder_state_dict)
        baseline_model.disease_classifier.load_state_dict(classifier_state_dict)
        baseline_model.eval()  # Set model to evaluation mode
        
        # Prepare test data batch
        feature_batch, metadata_batch_disease = create_batch(
            X_clr_df_test, test_metadata, batch_size=test_metadata.shape[0], is_test=True
        )
        
        # Evaluate on test data
        with torch.no_grad():
            encoded_feature_batch = baseline_model.encoder(feature_batch)
            prediction_scores = baseline_model.disease_classifier(encoded_feature_batch).view(-1)
            pred_prob = torch.sigmoid(prediction_scores)
            pred_tag = (pred_prob > 0.5).float()
        
        # Compute evaluation metrics
        disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
        if len(torch.unique(metadata_batch_disease)) > 1:
            auc = roc_auc_score(metadata_batch_disease.cpu(), pred_prob.cpu())
        else:
            auc = np.nan  # Use NaN if AUC cannot be computed
        f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())
        
        print(f"Fold Evaluation --> Accuracy: {disease_acc:.4f}, AUC: {auc}, F1 Score: {f1:.4f}")
        accuracies.append(disease_acc)
        aucs.append(auc)
        f1_scores.append(f1)
    
    # Calculate average metrics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_auc = np.nanmean(aucs)  # Use nanmean to ignore NaN values
    std_auc = np.nanstd(aucs)   # Use nanstd to ignore NaN values
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print("\nFinal Evaluation on Test Data (Baseline Model):")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Save evaluation results
    eval_results = {
        'accuracies': accuracies,
        'aucs': aucs,
        'f1_scores': f1_scores,
        'average_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'average_auc': avg_auc,
        'std_auc': std_auc,
        'average_f1_score': avg_f1,
        'std_f1_score': std_f1
    }
    with open('baseline_test_evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    print("Test evaluation results saved to 'baseline_test_evaluation_results.json'.")

if __name__ == "__main__":
    evaluate_baseline()
