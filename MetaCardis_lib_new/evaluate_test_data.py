# evaluate_test_data.py

import os
import torch
import pandas as pd
from data_processing import set_seed, load_and_transform_data, MixedDataset
from models import GAN
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import numpy as np
import json

def evaluate_test_data():
    """Load saved models and evaluate them on the test dataset."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths to test data
    test_file_path = 'MetaCardis_data/test_T2D_abundance.csv'
    test_metadata_file_path = 'MetaCardis_data/test_T2D_metadata.csv'
    
    # Load and transform test data
    X_clr_df_test = load_and_transform_data(test_file_path)
    test_metadata = pd.read_csv(test_metadata_file_path)
    
    # Ensure models directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found.")
    
    # Get list of saved models
    encoder_files = sorted([f for f in os.listdir(models_dir) if f.startswith('encoder_fold')])
    classifier_files = sorted([f for f in os.listdir(models_dir) if f.startswith('disease_classifier_fold')])
    
    if not encoder_files or not classifier_files:
        raise FileNotFoundError("Saved model files not found in the 'models' directory.")
    
    # Initialize lists to store evaluation metrics
    accuracies = []
    aucs = []
    f1_scores = []
    
    for encoder_file, classifier_file in zip(encoder_files, classifier_files):
        # Load the saved models
        print(f"Loading models: {encoder_file}, {classifier_file}")
        encoder_state_dict = torch.load(os.path.join(models_dir, encoder_file), map_location=device)
        classifier_state_dict = torch.load(os.path.join(models_dir, classifier_file), map_location=device)
        
        # Initialize model architecture
        input_dim = X_clr_df_test.shape[1]
        gan = GAN(input_dim=input_dim)
        gan.encoder.load_state_dict(encoder_state_dict)
        gan.disease_classifier.load_state_dict(classifier_state_dict)
        gan.to(device)
        gan.eval()  # Set model to evaluation mode
        
        # Prepare test data using MixedDataset and DataLoader
        test_dataset = MixedDataset(X_clr_df_test, test_metadata, device)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for feature_batch, metadata_batch_disease in test_loader:
                feature_batch = feature_batch.to(device)
                metadata_batch_disease = metadata_batch_disease.to(device)
                
                # Forward pass
                encoded_feature_batch = gan.encoder(feature_batch)
                prediction_scores = gan.disease_classifier(encoded_feature_batch).view(-1)
                pred_prob = torch.sigmoid(prediction_scores)
                pred_tag = (pred_prob > 0.5).float()
                
                # Collect predictions and labels
                all_preds.append(pred_tag)
                all_labels.append(metadata_batch_disease)
                all_scores.append(pred_prob)
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_scores = torch.cat(all_scores)
        
        # Compute evaluation metrics
        disease_acc = balanced_accuracy_score(all_labels.cpu(), all_preds.cpu())
        if len(torch.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels.cpu(), all_scores.cpu())
        else:
            auc = np.nan  # Use NaN if AUC cannot be computed
        f1 = f1_score(all_labels.cpu(), all_preds.cpu())
        
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
