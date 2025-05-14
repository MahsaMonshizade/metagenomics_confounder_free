# Modified models.py
import torch
import torch.nn as nn
from models import get_activation, get_norm_layer, previous_power_of_two



class GAN(nn.Module):
    """
    GAN model with a configurable encoder, confounder classifier, and reconstructor.
    This version is designed for pre-training on unlabeled data using reconstruction.
    """
    def __init__(self, input_size, latent_dim, num_encoder_layers, num_classifier_layers,
                 dropout_rate, norm="batch", classifier_hidden_dims=None, activation="relu", last_activation="relu"):
        super(GAN, self).__init__()
        self.activation = activation  # Save the chosen activation
        self.last_activation = last_activation
        self.input_size = input_size  # Save input size for reconstructor
        
        self.encoder = self._build_encoder(input_size, latent_dim, num_encoder_layers,
                                           dropout_rate, norm)
        self.classifier = self._build_classifier(latent_dim, num_classifier_layers,
                                                 dropout_rate, norm, classifier_hidden_dims)
        # Add reconstructor component for self-supervised learning
        self.reconstructor = self._build_reconstructor(latent_dim, num_encoder_layers,
                                                      dropout_rate, norm)

    def _build_encoder(self, input_size, latent_dim, num_layers, dropout_rate, norm):
        layers = []
        # Starting layer: use the largest power of two ≤ input_size.
        first_layer_dim = previous_power_of_two(input_size)
        layers.append(nn.Linear(input_size, first_layer_dim))
        layers.append(get_norm_layer(norm, first_layer_dim))
        layers.append(get_activation(self.activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        current_dim = first_layer_dim
        # Add extra encoder layers if requested.
        for _ in range(num_layers - 1):
            new_dim = current_dim // 2
            layers.append(nn.Linear(current_dim, new_dim))
            layers.append(get_norm_layer(norm, new_dim))
            layers.append(get_activation(self.activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = new_dim
        
        # Final projection to the latent space.
        layers.append(nn.Linear(current_dim, latent_dim))
        layers.append(get_norm_layer(norm, latent_dim))
        layers.append(get_activation(self.activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _build_classifier(self, latent_dim, num_layers, dropout_rate, norm, hidden_dims):
        layers = []
        current_dim = latent_dim
        
        # If hidden dimensions are provided, use them.
        if hidden_dims and len(hidden_dims) > 0:
            for hd in hidden_dims:
                layers.append(nn.Linear(current_dim, hd))
                layers.append(get_norm_layer(norm, hd))
                layers.append(get_activation(self.activation))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = hd
        else:
            # Otherwise, reduce dimension by half in each layer.
            for i in range(num_layers):
                new_dim = current_dim // 2
                layers.append(nn.Linear(current_dim, new_dim))
                layers.append(get_norm_layer(norm, new_dim))
                if i == num_layers-1:
                    layers.append(get_activation(self.activation))
                else:
                    layers.append(get_activation(self.last_activation))
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = new_dim
        
        # Final output layer.
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)
    
    def _build_reconstructor(self, latent_dim, num_layers, dropout_rate, norm):
        """
        Build a decoder/reconstructor network that mirrors the encoder structure
        to reconstruct the original input from the latent representation.
        """
        layers = []
        # Start from latent dimension
        current_dim = latent_dim
        
        # Calculate dimensions for each layer by reversing the encoder structure
        dimensions = []
        temp_dim = previous_power_of_two(self.input_size)
        for _ in range(num_layers - 1):
            dimensions.append(temp_dim)
            temp_dim = temp_dim // 2
        dimensions.reverse()  # Reverse to get expanding dimensions
        
        # First layer from latent space
        first_hidden_dim = dimensions[0] if dimensions else previous_power_of_two(self.input_size) // 2
        layers.append(nn.Linear(current_dim, first_hidden_dim))
        layers.append(get_norm_layer(norm, first_hidden_dim))
        layers.append(get_activation(self.activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = first_hidden_dim
        
        # Add expanding layers
        for i, dim in enumerate(dimensions[1:]):
            layers.append(nn.Linear(current_dim, dim))
            layers.append(get_norm_layer(norm, dim))
            layers.append(get_activation(self.activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        # Final layer to reconstruct input
        layers.append(nn.Linear(current_dim, self.input_size))
        
        return nn.Sequential(*layers)