import torch
import torch.nn as nn
import torch.nn.functional as F

# Define MaskedLinear
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mask = nn.Parameter(mask, requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.bias is not None:
            self.bias.register_hook(self._zero_bias_grad)
                
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.weight.data *= self.mask  # Apply mask to initial weights
        
    def forward(self, input):
        masked_weight = self.weight * self.mask  # Apply mask to weights
        return F.linear(input, masked_weight, self.bias)
    
    def _zero_bias_grad(self, grad):
        # Hook function to zero out the bias gradient
        return torch.zeros_like(grad)
    
    def __repr__(self):
        return (f"MaskedLinear("
                f"in_features={self.weight.shape[1]}, "
                f"out_features={self.weight.shape[0]}, "
                f"bias={self.bias is not None}, "
                f"mask_nonzero={self.mask.nonzero().size(0)})")


def previous_power_of_two(x):
    """Return the largest power of two less than or equal to x."""
    return 1 << ((x - 1).bit_length() - 1)

def get_norm_layer(norm_type, num_features):
    """Return a normalization layer based on norm_type."""
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

def get_activation(act):
    """Return an activation layer based on the given activation function name."""
    act = act.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise ValueError(f"Unsupported activation: {act}")

class PearsonCorrelationLoss(nn.Module):
    """
    Custom loss function based on Pearson correlation.
    Returns 0 loss for perfect correlation.
    """
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, pred, target):
        x = target
        y = pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm = x - mx
        ym = y - my
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + 1e-5
        r = r_num / r_den
        r = torch.clamp(r, min=-1.0, max=1.0)
        # Loss is lower when correlation is high (perfect correlation gives 0 loss).
        return 1 - r ** 2

class preGAN(nn.Module):
    """
    GAN model with a configurable encoder and two classifier branches.
    (Despite the name GAN, this model defines an encoder with two parallel classifier heads—
     one of which, disease_classifier, is used for the main prediction.)
    """
    def __init__(self, mask, input_size, latent_dim, num_encoder_layers, num_classifier_layers,
                 dropout_rate, norm="batch", classifier_hidden_dims=None, activation="relu"):
        super(preGAN, self).__init__()
        self.activation = activation  # Save the chosen activation
        self.input_size = input_size  # Save input size for reconstructor

        self.encoder = self._build_encoder(mask, input_size, latent_dim, num_encoder_layers,
                                           dropout_rate, norm)
        # Add reconstructor component for self-supervised learning
        self.reconstructor = self._build_reconstructor(latent_dim, num_encoder_layers,
                                                      dropout_rate, norm)

    def _build_encoder(self, mask, input_size, latent_dim, num_layers, dropout_rate, norm):
        layers = []
        # Starting layer: use the largest power of two ≤ input_size.
        layers = []
        input_size = mask.shape[1]
        first_layer_dim = mask.shape[0]
            
        # Create a MaskedLinear layer and add to layers
        layers.append(MaskedLinear(input_size, first_layer_dim, mask))
        
        layers.append(get_norm_layer(norm, first_layer_dim))
        layers.append(get_activation(self.activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        current_dim = first_layer_dim
        # Add extra encoder layers if requested.
        for _ in range(num_layers - 1):
            new_dim = previous_power_of_two(current_dim) // 2
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
            for _ in range(num_layers):
                new_dim = current_dim // 2
                layers.append(nn.Linear(current_dim, new_dim))
                layers.append(get_norm_layer(norm, new_dim))
                layers.append(get_activation(self.activation))
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