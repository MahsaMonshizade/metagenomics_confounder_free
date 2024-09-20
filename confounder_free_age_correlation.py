import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clr_transformation(X):
    """Perform CLR (Centered Log-Ratio) transformation on the data."""
    geometric_mean = np.exp(np.mean(np.log(X), axis=1))
    return np.log(X / geometric_mean[:, np.newaxis])


def load_and_transform_data(file_path):
    """Load data from CSV, apply CLR transformation, and return transformed DataFrame with 'loaded_uid'."""
    data = pd.read_csv(file_path)
    loaded_uid = data['loaded_uid']
    X = data.drop(columns=['loaded_uid']).values
    X += 1e-6  # Adding a small pseudocount to avoid log(0)
    X_clr = clr_transformation(X)
    X_clr_df = pd.DataFrame(X_clr, columns=data.columns[1:])
    X_clr_df['loaded_uid'] = loaded_uid
    return X_clr_df[['loaded_uid'] + list(X_clr_df.columns[:-1])]


def preprocess_metadata(metadata):
    """Converts categorical metadata into numeric and one-hot encoded features."""
    disease_dict = {'D006262': 0, 'D003093': 1}
    metadata['disease_numeric'] = metadata['disease'].map(disease_dict)
    return metadata


def create_batch(relative_abundance, metadata, batch_size, is_test=False):
    """Creates a batch of data by sampling from the metadata and relative abundance data."""
    metadata = preprocess_metadata(metadata)
    proportions = metadata['disease'].value_counts(normalize=True)
    num_samples_per_group = (proportions * batch_size).round().astype(int)

    # Sample metadata
    metadata_feature_batch = metadata.groupby('disease').apply(
        lambda x: x.sample(n=num_samples_per_group[x.name])
    ).reset_index(drop=True)

    # Sample relative abundance
    training_feature_batch = relative_abundance[relative_abundance['loaded_uid'].isin(metadata_feature_batch['uid'])]
    training_feature_batch = training_feature_batch.set_index('loaded_uid').reindex(metadata_feature_batch['uid']).reset_index()
    training_feature_batch.rename(columns={'loaded_uid': 'uid'}, inplace=True)
    training_feature_batch = training_feature_batch.drop(columns=['uid'])

    # Convert to tensors
    training_feature_batch = torch.tensor(training_feature_batch.values, dtype=torch.float32)
    metadata_batch_disease = torch.tensor(metadata_feature_batch['disease_numeric'].values, dtype=torch.float32)

    if is_test:
        return training_feature_batch, metadata_batch_disease

    # Control batch
    ctrl_metadata = metadata[metadata['disease'] == 'D006262']
    run_ids = ctrl_metadata['uid']
    ctrl_relative_abundance = relative_abundance[relative_abundance['loaded_uid'].isin(run_ids)]

    ctrl_idx = np.random.permutation(ctrl_metadata.index)[:batch_size]
    training_feature_ctrl_batch = ctrl_relative_abundance.loc[ctrl_idx].rename(columns={'loaded_uid': 'uid'}).drop(columns=['uid'])
    metadata_ctrl_batch = ctrl_metadata.loc[ctrl_idx]

    training_feature_ctrl_batch = torch.tensor(training_feature_ctrl_batch.values, dtype=torch.float32)
    metadata_ctrl_batch_age = torch.tensor(metadata_ctrl_batch['host_age'].values, dtype=torch.float32)

    return (training_feature_ctrl_batch, metadata_ctrl_batch_age, 
            training_feature_batch, metadata_batch_disease)



def correlation_coefficient_loss(x, y):
   
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    covariance = torch.mean((x - mean_x) * (y - mean_y))
    std_x = torch.std(x)
    std_y = torch.std(y)
    eps = 1e-5
    return (covariance / (std_x * std_y) +eps)**2




class GAN(nn.Module):
    def __init__(self, input_dim, latent_dim=128, lr=0.01, dropout_rate=0.3, pos_weight=2):
        """Initializes the GAN with an encoder, age regressor, and disease classifier."""
        super(GAN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.age_regressor = nn.Sequential(
           nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
             nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.age_regression_loss = nn.MSELoss()

        self.disease_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.disease_classifier_loss = nn.BCELoss()

        # Optimizers
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.disease_classifier.parameters()), lr
        )
        self.optimizer_distiller = optim.Adam(self.encoder.parameters(), lr)
        self.optimizer_regression_age = optim.Adam(self.age_regressor.parameters(), lr)

        # Schedulers
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.scheduler_distiller = lr_scheduler.ReduceLROnPlateau(self.optimizer_distiller, mode='min', factor=0.5, patience=5)
        self.scheduler_regression_age = lr_scheduler.ReduceLROnPlateau(self.optimizer_regression_age, mode='min', factor=0.5, patience=5)

    def initialize_weights(self):
        """Initialize weights using Kaiming initialization for layers with ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def train(self, epochs, relative_abundance, metadata, batch_size=64):
        best_acc = 0
        early_stop_step = 10
        early_stop_patience = 0
        
        X_clr_df_train, X_clr_df_val, train_metadata, val_metadata = train_test_split(relative_abundance, metadata, test_size=0.2, random_state=42)
        
        # Lists to store losses
        r_losses = []
        g_losses = []
        c_losses = []
        for epoch in range(epochs):
            # Create batches
            training_feature_ctrl_batch, metadata_ctrl_batch_age, training_feature_batch, metadata_batch_disease = create_batch(
                X_clr_df_train, train_metadata, batch_size
            )

            # Train age regressor
            self.optimizer_regression_age.zero_grad()
            for param in self.encoder.parameters():
                param.requires_grad = False
            encoded_feature_ctrl_batch = self.encoder(training_feature_ctrl_batch)
            age_prediction = self.age_regressor(encoded_feature_ctrl_batch)
            r_loss = self.age_regression_loss(metadata_ctrl_batch_age.view(-1,1), age_prediction)
            r_loss.backward()
            
            self.optimizer_regression_age.step()
            
            for param in self.encoder.parameters():
                param.requires_grad = True

            # Train distiller
            self.optimizer_distiller.zero_grad()
            for param in self.age_regressor.parameters():
                param.requires_grad = False
            encoder_features = self.encoder(training_feature_ctrl_batch)
            predicted_age = self.age_regressor(encoder_features)
            g_loss = correlation_coefficient_loss(encoder_features, metadata_ctrl_batch_age.view(-1,1))
            g_loss.backward()

            initial_params_dist = {name: param.clone() for name, param in self.encoder.named_parameters()}
            self.optimizer_distiller.step()

            for name, param in self.encoder.named_parameters():
                if not torch.equal(initial_params_dist[name], param):
                    print(f"Parameter {name} updated")
                else:
                    print(f"Parameter {name} did not update")

            for param in self.age_regressor.parameters():
                param.requires_grad = True

            # Train encoder & classifier
            self.optimizer.zero_grad()
            encoded_feature_batch = self.encoder(training_feature_batch)
            prediction_scores = self.disease_classifier(encoded_feature_batch)
            c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
            c_loss.backward()
            pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
            disease_acc = accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
            self.optimizer.step()
            self.scheduler.step(disease_acc)

            # Store the losses
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
            c_losses.append(c_loss.item())

            if disease_acc > best_acc:
                best_acc = disease_acc
                early_stop_patience = 0
            else:
                early_stop_patience += 1
            if early_stop_patience == early_stop_step:
                break

            print(f"Epoch {epoch + 1}/{epochs}, r_loss: {r_loss.item()}, g_loss: {g_loss.item()}, c_loss: {c_loss.item()}, disease_acc: {disease_acc}")

            self.evaluate(relative_abundance=X_clr_df_val, metadata=val_metadata, batch_size=val_metadata.shape[0], t='eval')
            self.evaluate(relative_abundance=X_clr_df_train, metadata=train_metadata, batch_size=train_metadata.shape[0], t='train')

        self.plot_losses(r_losses, g_losses, c_losses)


    def plot_losses(self, r_losses, g_losses, c_losses):
        """Plots r_loss, g_loss, and c_loss over epochs."""
        plt.figure(figsize=(12, 6))
        # plt.plot(r_losses, label='r_loss', color='r')
        plt.plot(g_losses, label='g_loss', color='g')
        print(g_losses)
        # plt.plot(c_losses, label='c_loss', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/confounder_free_age_linear_correlation_gloss.png')

        plt.figure(figsize=(12, 6))
        plt.plot(r_losses, label='r_loss', color='r')
        # plt.plot(g_losses, label='g_loss', color='g')
        plt.plot(c_losses, label='c_loss', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/confounder_free_age_linear_correlation_rloss.png')

        plt.figure(figsize=(12, 6))
        # plt.plot(r_losses, label='r_loss', color='r')
        # plt.plot(g_losses, label='g_loss', color='g')
        plt.plot(c_losses, label='c_loss', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/confounder_free_age_linear_correlation_closs.png')

    def evaluate(self, relative_abundance, metadata, batch_size, t):
        """Evaluates the trained GAN model on test data."""
        # Create batches
        feature_batch, metadata_batch_disease = create_batch(relative_abundance, metadata, batch_size, True)
    
        # Get encoded features
        encoded_feature_batch = self.encoder(feature_batch)
    
        # Get prediction scores (probabilities)
        prediction_scores = self.disease_classifier(encoded_feature_batch)
    
        # Convert probabilities to predicted classes
        pred_tag = [1 if p > 0.5 else 0 for p in prediction_scores]
    
        # Calculate accuracy
        disease_acc = accuracy_score(metadata_batch_disease.view(-1, 1), pred_tag)
    
        # Calculate classifier loss
        c_loss = self.disease_classifier_loss(prediction_scores, metadata_batch_disease.view(-1, 1))
    
        # Calculate AUC
        if len(np.unique(metadata_batch_disease)) > 1:
            auc = roc_auc_score(metadata_batch_disease.view(-1, 1), prediction_scores.detach().numpy())
            print(f"{t} result --> Accuracy: {disease_acc}, Loss: {c_loss.item()}, AUC: {auc}")
        else:
            print("Cannot compute ROC AUC as only one class is present.")
        # auc = roc_auc_score(metadata_batch_disease.view(-1, 1), prediction_scores.detach().numpy())
    
        # Print results
            print(f"{t} result --> Accuracy: {disease_acc}, Loss: {c_loss.item()}")

if __name__ == "__main__":
    set_seed(42)

    # Load and transform training data
    file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_train.csv'
    metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_train.csv'
    X_clr_df = load_and_transform_data(file_path)
    metadata = pd.read_csv(metadata_file_path)

    # X_clr_df_train, X_clr_df_val, train_metadata, val_metadata = train_test_split(X_clr_df, metadata, test_size=0.2, random_state=42)

    # Initialize and train GAN
    gan = GAN(input_dim=X_clr_df.shape[1] - 1)
    gan.initialize_weights()
    gan.train(epochs=1500, relative_abundance=X_clr_df, metadata=metadata, batch_size=64)

    # Load and transform test data
    test_file_path = 'GMrepo_data/UC_relative_abundance_metagenomics_test.csv'
    test_metadata_file_path = 'GMrepo_data/UC_metadata_metagenomics_test.csv'
    X_clr_df_test = load_and_transform_data(test_file_path)
    test_metadata = pd.read_csv(test_metadata_file_path)

    # Evaluate GAN on test data
    gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test')


    #  # Load and transform test data
    # test_file_path = 'GMrepo_data/test_relative_abundance_IBD_1.csv'
    # test_metadata_file_path = 'GMrepo_data/test_metadata_IBD_1.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)

    # # Evaluate GAN on test data
    # gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test_1')

    #  # Load and transform test data
    # test_file_path = 'GMrepo_data/test_relative_abundance_IBD_2.csv'
    # test_metadata_file_path = 'GMrepo_data/test_metadata_IBD_2.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)

    # # Evaluate GAN on test data
    # gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test_2')

    #  # Load and transform test data
    # test_file_path = 'GMrepo_data/test_relative_abundance_IBD_3.csv'
    # test_metadata_file_path = 'GMrepo_data/test_metadata_IBD_3.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)

    # # Evaluate GAN on test data
    # gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test_3')

    #  # Load and transform test data
    # test_file_path = 'GMrepo_data/test_relative_abundance_IBD_4.csv'
    # test_metadata_file_path = 'GMrepo_data/test_metadata_IBD_4.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)

    # # Evaluate GAN on test data
    # gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test_4')

    #  # Load and transform test data
    # test_file_path = 'GMrepo_data/test_relative_abundance_IBD_5.csv'
    # test_metadata_file_path = 'GMrepo_data/test_metadata_IBD_5.csv'
    # X_clr_df_test = load_and_transform_data(test_file_path)
    # test_metadata = pd.read_csv(test_metadata_file_path)

    # # Evaluate GAN on test data
    # gan.evaluate(relative_abundance=X_clr_df_test, metadata=test_metadata, batch_size=test_metadata.shape[0], t = 'test_5')