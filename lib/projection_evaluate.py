# evaluate_test_data.py

import os
import torch
import pandas as pd
from data_processing import set_seed, load_and_transform_data, preprocess_metadata, create_batch
from projection_model import GAN
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import numpy as np
import json

def evaluate_test_data():
    """Load saved models and evaluate them on the test dataset."""
    set_seed(42)
    
    # Paths to test data
    test_file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_test.csv'
    test_metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_test.csv'
    
    # Load and transform test data
    X_clr_df_test = load_and_transform_data(test_file_path)
    test_metadata = pd.read_csv(test_metadata_file_path)
    
    # Preprocess metadata
    test_metadata = preprocess_metadata(test_metadata)
    
    # Ensure models directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")
    
    # Get list of saved models
    encoder_files = sorted([f for f in os.listdir(models_dir) if f.startswith('encoder_fold')])
    classifier_files = sorted([f for f in os.listdir(models_dir) if f.startswith('disease_classifier_fold')])
    age_regressor_files = sorted([f for f in os.listdir(models_dir) if f.startswith('age_regressor_fold')])
    bmi_regressor_files = sorted([f for f in os.listdir(models_dir) if f.startswith('bmi_regressor_fold')])
    
    if not encoder_files or not classifier_files or not age_regressor_files or not bmi_regressor_files:
        raise FileNotFoundError("Saved model files not found in the 'models' directory.")
    
    # Initialize lists to store evaluation metrics
    accuracies = []
    aucs = []
    f1_scores = []
    
    for fold_idx in range(len(encoder_files)):
        encoder_file = encoder_files[fold_idx]
        classifier_file = classifier_files[fold_idx]
        age_regressor_file = age_regressor_files[fold_idx]
        bmi_regressor_file = bmi_regressor_files[fold_idx]
        
        # Load the saved models
        print(f"Loading models: {encoder_file}, {classifier_file}, {age_regressor_file}, {bmi_regressor_file}")
        encoder_state_dict = torch.load(os.path.join(models_dir, encoder_file))
        classifier_state_dict = torch.load(os.path.join(models_dir, classifier_file))
        age_regressor_state_dict = torch.load(os.path.join(models_dir, age_regressor_file))
        bmi_regressor_state_dict = torch.load(os.path.join(models_dir, bmi_regressor_file))
        
        # Initialize model architecture
        input_dim = X_clr_df_test.shape[1] - 1  # Subtract 1 for 'uid' column
        gan = GAN(input_dim=input_dim)
        gan.encoder.load_state_dict(encoder_state_dict)
        gan.disease_classifier.load_state_dict(classifier_state_dict)
        gan.age_regressor.load_state_dict(age_regressor_state_dict)
        gan.bmi_regressor.load_state_dict(bmi_regressor_state_dict)
        gan.eval()  # Set model to evaluation mode
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gan.to(device)
        
        # Prepare test data batch
        feature_batch, metadata_batch_disease = create_batch(
            X_clr_df_test, test_metadata, batch_size=test_metadata.shape[0], is_test=True, device=device
        )
        
        # Evaluate on test data
        with torch.no_grad():
            encoded_feature_batch = gan.encoder(feature_batch)
            projected_features = gan.orthogonal_projection(encoded_feature_batch)
            prediction_scores = gan.disease_classifier(projected_features).view(-1)
            pred_prob = torch.sigmoid(prediction_scores)
            pred_tag = (pred_prob > 0.5).float()
        
        # Compute evaluation metrics
        disease_acc = balanced_accuracy_score(metadata_batch_disease.cpu(), pred_tag.cpu())
        if len(torch.unique(metadata_batch_disease)) > 1:
            auc = roc_auc_score(metadata_batch_disease.cpu(), pred_prob.cpu())
        else:
            auc = np.nan  # Use NaN if AUC cannot be computed
        f1 = f1_score(metadata_batch_disease.cpu(), pred_tag.cpu())
        
        print(f"Fold {fold_idx + 1} Evaluation --> Accuracy: {disease_acc:.4f}, AUC: {auc:.4f}, F1 Score: {f1:.4f}")
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
    
    print("\nFinal Evaluation on Test Data:")
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
    with open('test_evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    print("Test evaluation results saved to 'test_evaluation_results.json'.")

if __name__ == "__main__":
    evaluate_test_data()
