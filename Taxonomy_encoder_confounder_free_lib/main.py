import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay

from data_utils import get_data
from models import GAN, PearsonCorrelationLoss
from utils import create_stratified_dataloader
from train import train_model

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title, save_path, class_names=None):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def main():
    # Hyperparameters
    input_size = 371  # 654 features
    latent_dim = 32
    num_layers = 1
    learning_rate = 0.0001
    num_epochs = 100
    batch_size = 64

    train_abundance_path = 'MetaCardis_data/new_train_T2D_abundance_with_taxon_ids.csv'
    train_metadata_path = 'MetaCardis_data/train_T2D_metadata.csv'
    test_abundance_path = 'MetaCardis_data/new_test_T2D_abundance_with_taxon_ids.csv'
    test_metadata_path = 'MetaCardis_data/test_T2D_metadata.csv'

    df = pd.read_csv('Default_Database/features_taxonomy_ranking.csv')

    # Correct taxonomic ranks
    taxonomic_ranks = ['species', 'genus', 'family', 'order']

    # Adjust data to include only the required ranks in the correct order
    adjusted_data = []
    for idx, row in df.iterrows():
        new_row = [
            row['species'],  # species
            row['genus'],
            row['family'],
            row['order'],
        ]
        adjusted_data.append(new_row)

    # Rebuild index maps for the adjusted taxonomic ranks
    index_maps = {}
    for i, rank in enumerate(taxonomic_ranks):
        unique_items = sorted(set(row[i] for row in adjusted_data))
        index_maps[rank] = {item: idx for idx, item in enumerate(unique_items)}

    # Build masks for each pair of consecutive ranks
    def build_mask(data, from_rank, to_rank, index_maps):
        from_size = len(index_maps[from_rank])
        to_size = len(index_maps[to_rank])
        mask = torch.zeros(to_size, from_size)
        pairs = set()
        from_idx = taxonomic_ranks.index(from_rank)
        to_idx = taxonomic_ranks.index(to_rank)
        for row in data:
            from_item = row[from_idx]
            to_item = row[to_idx]
            from_id = index_maps[from_rank][from_item]
            to_id = index_maps[to_rank][to_item]
            pairs.add((to_id, from_id))
        for to_id, from_id in pairs:
            mask[to_id, from_id] = 1
        return mask

    # Build masks
    masks = {}
    for i in range(len(taxonomic_ranks) - 1):
        from_rank = taxonomic_ranks[i]
        to_rank = taxonomic_ranks[i + 1]
        mask_name = f"mask_{from_rank}_{to_rank}"
        masks[mask_name] = build_mask(adjusted_data, from_rank, to_rank, index_maps)

    # Load merged data
    merged_data_all = get_data(train_abundance_path, train_metadata_path)
    merged_test_data_all = get_data(test_abundance_path, test_metadata_path)

    # Define feature columns
    metadata_columns = pd.read_csv(train_metadata_path).columns.tolist()
    feature_columns = [
        col for col in merged_data_all.columns if col not in metadata_columns and col != 'SampleID'
    ]

    X = merged_data_all[feature_columns].values
    y_all = merged_data_all['PATGROUPFINAL_C'].values  # Labels for disease classification

    # Prepare test data
    x_test_all = torch.tensor(merged_test_data_all[feature_columns].values, dtype=torch.float32)
    y_test_all = torch.tensor(merged_test_data_all['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

    # Disease group data (patients with PATGROUPFINAL_C == 1) in test set
    test_data_disease = merged_test_data_all[merged_test_data_all['PATGROUPFINAL_C'] == 1]
    x_test_disease = torch.tensor(test_data_disease[feature_columns].values, dtype=torch.float32)
    y_test_disease = torch.tensor(test_data_disease['METFORMIN_C'].values, dtype=torch.float32).unsqueeze(1)

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # To store metrics across folds
    train_metrics_per_fold = []
    val_metrics_per_fold = []
    test_metrics_per_fold = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_all)):
        print(f"Fold {fold+1}")

        # Split data into training and validation sets
        train_data = merged_data_all.iloc[train_index]
        val_data = merged_data_all.iloc[val_index]

        # Prepare training data
        x_all_train = torch.tensor(train_data[feature_columns].values, dtype=torch.float32)
        y_all_train = torch.tensor(train_data['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

        x_all_val = torch.tensor(val_data[feature_columns].values, dtype=torch.float32)
        y_all_val = torch.tensor(val_data['PATGROUPFINAL_C'].values, dtype=torch.float32).unsqueeze(1)

        # Disease group data (patients with PATGROUPFINAL_C == 1)
        train_data_disease = train_data[train_data['PATGROUPFINAL_C'] == 1]
        val_data_disease = val_data[val_data['PATGROUPFINAL_C'] == 1]

        x_disease_train = torch.tensor(train_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_train = torch.tensor(train_data_disease['METFORMIN_C'].values, dtype=torch.float32).unsqueeze(1)

        x_disease_val = torch.tensor(val_data_disease[feature_columns].values, dtype=torch.float32)
        y_disease_val = torch.tensor(val_data_disease['METFORMIN_C'].values, dtype=torch.float32).unsqueeze(1)

        # Create stratified DataLoaders
        data_loader = create_stratified_dataloader(x_disease_train, y_disease_train, batch_size)
        data_all_loader = create_stratified_dataloader(x_all_train, y_all_train, batch_size)
        data_val_loader = create_stratified_dataloader(x_disease_val, y_disease_val, batch_size)
        data_all_val_loader = create_stratified_dataloader(x_all_val, y_all_val, batch_size)
        data_test_loader = create_stratified_dataloader(x_test_disease, y_test_disease, batch_size)
        data_all_test_loader = create_stratified_dataloader(x_test_all, y_test_all, batch_size)

        # Compute positive class weights
        num_pos_disease = y_all_train.sum().item()
        num_neg_disease = len(y_all_train) - num_pos_disease
        pos_weight_value_disease = num_neg_disease / num_pos_disease
        pos_weight_disease = torch.tensor([pos_weight_value_disease], dtype=torch.float32).to(device)

        num_pos_drug = y_disease_train.sum().item()
        num_neg_drug = len(y_disease_train) - num_pos_drug
        pos_weight_value_drug = num_neg_drug / num_pos_drug
        pos_weight_drug = torch.tensor([pos_weight_value_drug], dtype=torch.float32).to(device)

        # Define model, loss, and optimizer
        model = GAN(input_size, masks, index_maps, taxonomic_ranks,  num_layers).to(device)
        criterion = PearsonCorrelationLoss().to(device)
        optimizer = optim.Adam(model.encoder.parameters(), lr=0.002)
        criterion_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drug).to(device)
        optimizer_classifier = optim.Adam(model.classifier.parameters(), lr=0.002)
        criterion_disease_classifier = nn.BCEWithLogitsLoss(pos_weight=pos_weight_disease).to(device)
        optimizer_disease_classifier = optim.Adam(
            list(model.encoder.parameters()) + list(model.disease_classifier.parameters()), lr=learning_rate
        )

        # Train the model
        Results = train_model(
            model, criterion, optimizer, data_loader, data_all_loader, data_val_loader, data_all_val_loader,
            data_test_loader, data_all_test_loader, num_epochs,
            criterion_classifier, optimizer_classifier, criterion_disease_classifier, optimizer_disease_classifier,
            device
        )

        # Store metrics for this fold
        train_metrics_per_fold.append(Results['train'])
        val_metrics_per_fold.append(Results['val'])
        test_metrics_per_fold.append(Results['test'])


        # Plot confusion matrices for the final epoch of this fold
        plot_confusion_matrix(Results['train']['confusion_matrix'][-1], 
                      title=f'Train Confusion Matrix - Fold {fold+1}', 
                      save_path=f'plots/fold_{fold+1}_train_conf_matrix.png',
                      class_names=['Class 0', 'Class 1'])

        plot_confusion_matrix(Results['val']['confusion_matrix'][-1], 
                      title=f'Validation Confusion Matrix - Fold {fold+1}', 
                      save_path=f'plots/fold_{fold+1}_val_conf_matrix.png',
                      class_names=['Class 0', 'Class 1'])

        plot_confusion_matrix(Results['test']['confusion_matrix'][-1], 
                      title=f'Test Confusion Matrix - Fold {fold+1}', 
                      save_path=f'plots/fold_{fold+1}_test_conf_matrix.png',
                      class_names=['Class 0', 'Class 1'])
        
        num_epochs_actual = len(Results['train']['gloss_history'])
        epochs = range(1, num_epochs_actual + 1)

        # Plot metrics for this fold
        plt.figure(figsize=(20, 15))

        plt.subplot(3, 3, 1)
        plt.plot(epochs, Results['train']['gloss_history'], label=f'Fold {fold+1}')
        plt.title("correlation g Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.plot(epochs, Results['train']['dcor_history'], label=f'Fold train {fold+1}')
        plt.plot(epochs, Results['val']['dcor_history'], label=f'Fold val {fold+1}')
        plt.plot(epochs, Results['test']['dcor_history'], label=f'Fold test {fold+1}')
        plt.title("Distance Correlation History")
        plt.xlabel("Epoch")
        plt.ylabel("Distance Correlation")
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(epochs, Results['train']['loss_history'], label=f'Fold train {fold+1}')
        plt.plot(epochs, Results['val']['loss_history'], label=f'Fold val {fold+1}')
        plt.plot(epochs, Results['test']['loss_history'], label=f'Fold test {fold+1}')
        plt.title("Disease Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Disease Loss")
        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(epochs, Results['train']['accuracy'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['accuracy'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['accuracy'], label=f'Fold {fold+1} Test')
        plt.title("Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(epochs, Results['train']['f1_score'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['f1_score'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['f1_score'], label=f'Fold {fold+1} Test')
        plt.title("F1 Score History")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(epochs, Results['train']['auc_pr'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['auc_pr'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['auc_pr'], label=f'Fold {fold+1} Test')
        plt.title("AUCPR Score History")
        plt.xlabel("Epoch")
        plt.ylabel("AUCPR Score")
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(epochs, Results['train']['precision'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['precision'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['precision'], label=f'Fold {fold+1} Test')
        plt.title("Precisions Score History")
        plt.xlabel("Epoch")
        plt.ylabel("Precisions Score")
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(epochs, Results['train']['recall'], label=f'Fold {fold+1} Train')
        plt.plot(epochs, Results['val']['recall'], label=f'Fold {fold+1} Val')
        plt.plot(epochs, Results['test']['recall'], label=f'Fold {fold+1} Test')
        plt.title("Recalls Score History")
        plt.xlabel("Epoch")
        plt.ylabel("Recalls Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'plots/fold_{fold+1}_metrics.png')
        plt.close()

    num_epochs_actual = len(train_metrics_per_fold[0]['gloss_history'])
    epochs = range(1, num_epochs_actual + 1)

    train_avg_metrics = {key: np.zeros(num_epochs_actual) for key in train_metrics_per_fold[0].keys() if key != 'confusion_matrix'}
    val_avg_metrics = {key: np.zeros(num_epochs_actual) for key in val_metrics_per_fold[0].keys() if key != 'confusion_matrix'}
    test_avg_metrics = { key: np.zeros(num_epochs_actual) for key in test_metrics_per_fold[0].keys() if key != 'confusion_matrix'}

    # Initialize confusion matrix averages separately
    train_conf_matrix_avg = [np.zeros_like(train_metrics_per_fold[0]['confusion_matrix'][0]) for _ in range(num_epochs_actual)]
    val_conf_matrix_avg = [np.zeros_like(val_metrics_per_fold[0]['confusion_matrix'][0]) for _ in range(num_epochs_actual)]
    test_conf_matrix_avg = [np.zeros_like(test_metrics_per_fold[0]['confusion_matrix'][0]) for _ in range(num_epochs_actual)]

    # Accumulate scalar metrics for averaging
    for train_fold_metrics in train_metrics_per_fold:
        for key in train_avg_metrics.keys():
            train_avg_metrics[key] += np.array(train_fold_metrics[key])
        for epoch_idx, cm in enumerate(train_fold_metrics['confusion_matrix']):
            train_conf_matrix_avg[epoch_idx] += cm

    for val_fold_metrics in val_metrics_per_fold:
        for key in val_avg_metrics.keys():
            val_avg_metrics[key] += np.array(val_fold_metrics[key])
        for epoch_idx, cm in enumerate(val_fold_metrics['confusion_matrix']):
            val_conf_matrix_avg[epoch_idx] += cm

    for test_fold_metrics in test_metrics_per_fold:
        for key in test_avg_metrics.keys():
            test_avg_metrics[key] += np.array(test_fold_metrics[key])
        for epoch_idx, cm in enumerate(test_fold_metrics['confusion_matrix']):
            test_conf_matrix_avg[epoch_idx] += cm

    # Compute averages across folds for scalar metrics
    num_train_folds = len(train_metrics_per_fold)
    num_val_folds = len(val_metrics_per_fold)
    num_test_folds = len(test_metrics_per_fold)

    for key in train_avg_metrics.keys():
        train_avg_metrics[key] /= num_train_folds

    for key in val_avg_metrics.keys():
        val_avg_metrics[key] /= num_val_folds

    for key in test_avg_metrics.keys():
        test_avg_metrics[key] /= num_test_folds

    # Plot average metrics across folds
    plt.figure(figsize=(20, 15))

    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_avg_metrics['gloss_history'], label='Average')
    plt.title("correlation g Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_avg_metrics['dcor_history'], label='Average train')
    plt.plot(epochs, val_avg_metrics['dcor_history'], label='Average val')
    plt.plot(epochs, test_avg_metrics['dcor_history'], label='Average test')
    plt.title("Average Distance Correlation History")
    plt.xlabel("Epoch")
    plt.ylabel("Distance Correlation")
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(epochs, train_avg_metrics['loss_history'], label='Average train')
    plt.plot(epochs, val_avg_metrics['loss_history'], label='Average val')
    plt.plot(epochs, test_avg_metrics['loss_history'], label='Average test')
    plt.title("Average Disease Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Disease Loss")
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_avg_metrics['accuracy'], label='Train Average')
    plt.plot(epochs, val_avg_metrics['accuracy'], label='Validation Average')
    plt.plot(epochs, test_avg_metrics['accuracy'], label='Test Average')
    plt.title("Average Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_avg_metrics['f1_score'], label='Train Average')
    plt.plot(epochs, val_avg_metrics['f1_score'], label='Validation Average')
    plt.plot(epochs, test_avg_metrics['f1_score'], label='Test Average')
    plt.title("Average F1 Score History")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_avg_metrics['auc_pr'], label='Train Average')
    plt.plot(epochs, val_avg_metrics['auc_pr'], label='Validation Average')
    plt.plot(epochs, test_avg_metrics['auc_pr'], label='Test Average')
    plt.title("Average AUCPR Score History")
    plt.xlabel("Epoch")
    plt.ylabel("AUCPR Score")
    plt.legend()

    plt.subplot(3, 3, 7)
    plt.plot(epochs, train_avg_metrics['precision'], label=f'Fold {fold+1} Train')
    plt.plot(epochs, val_avg_metrics['precision'], label=f'Fold {fold+1} Val')
    plt.plot(epochs, test_avg_metrics['precision'], label=f'Fold {fold+1} Test')
    plt.title("Average Precisions Score History")
    plt.xlabel("Epoch")
    plt.ylabel("Precision Score")
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.plot(epochs, train_avg_metrics['recall'], label=f'Fold {fold+1} Train')
    plt.plot(epochs, val_avg_metrics['recall'], label=f'Fold {fold+1} Val')
    plt.plot(epochs, test_avg_metrics['recall'], label=f'Fold {fold+1} Test')
    plt.title("Average Recalls Score History")
    plt.xlabel("Epoch")
    plt.ylabel("Recalls Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/average_metrics.png')
    plt.close()

    # Print average of the final epoch's test accuracy across folds

    # Compute averages for confusion matrices
    train_conf_matrix_avg = [cm / num_train_folds for cm in train_conf_matrix_avg]
    val_conf_matrix_avg = [cm / num_val_folds for cm in val_conf_matrix_avg]
    test_conf_matrix_avg = [cm / num_test_folds for cm in test_conf_matrix_avg]

    # Add the averaged confusion matrices back to the metrics dictionaries
    train_avg_metrics['confusion_matrix'] = train_conf_matrix_avg
    val_avg_metrics['confusion_matrix'] = val_conf_matrix_avg
    test_avg_metrics['confusion_matrix'] = test_conf_matrix_avg

    # Plot average confusion matrices
    plot_confusion_matrix(train_avg_metrics['confusion_matrix'][-1], 
                        title='Average Train Confusion Matrix', 
                        save_path='plots/average_train_conf_matrix.png',
                        class_names=['Class 0', 'Class 1'])

    plot_confusion_matrix(val_avg_metrics['confusion_matrix'][-1], 
                        title='Average Validation Confusion Matrix', 
                        save_path='plots/average_val_conf_matrix.png',
                        class_names=['Class 0', 'Class 1'])

    plot_confusion_matrix(test_avg_metrics['confusion_matrix'][-1], 
                        title='Average Test Confusion Matrix', 
                        save_path='plots/average_test_conf_matrix.png',
                        class_names=['Class 0', 'Class 1'])
    
    avg_test_accs = test_avg_metrics['accuracy'][-1]
    print(f"Average Test Accuracy over {n_splits} folds: {avg_test_accs:.4f}")

if __name__ == "__main__":
    main()