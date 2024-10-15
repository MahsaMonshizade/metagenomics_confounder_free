# losses.py

import torch
import torch.nn as nn

class CorrelationCoefficientLoss(nn.Module):
    """
    Custom loss function based on the squared correlation coefficient.
    """
    def __init__(self):
        super(CorrelationCoefficientLoss, self).__init__()

    def forward(self, y_true, y_pred):
        mean_x = torch.mean(y_true)
        mean_y = torch.mean(y_pred)
        covariance = torch.mean((y_true - mean_x) * (y_pred - mean_y))
        std_x = torch.std(y_true)
        std_y = torch.std(y_pred)
        eps = 1e-5
        corr = covariance / (std_x * std_y + eps)
        # corr = torch.clamp(corr, min=-1.0, max=1.0)
        return torch.abs(corr)


# class CorrelationCoefficientLoss(nn.Module):
#     def __init__(self):
#         super(CorrelationCoefficientLoss, self).__init__()

#     def forward(self, y_true, y_pred):
#         x = y_true
#         y = y_pred

#         mx = torch.mean(x)
#         my = torch.mean(y)
#         xm = x - mx
#         ym = y - my

#         r_num = torch.sum(xm * ym)
#         r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)  + 1e-5)
#         r = r_num / r_den

#         r = torch.clamp(r, min=-1.0, max=1.0)
#         return r ** 2

# class CorrelationCoefficientLoss(nn.Module):
#     def __init__(self):
#         super(CorrelationCoefficientLoss, self).__init__()

#     def forward(self, y_true, y_pred):
#         # Ensure inputs are 1D tensors
#         y_true = y_true.view(-1)
#         y_pred = y_pred.view(-1)
        
#         # Compute means
#         mean_true = torch.mean(y_true)
#         mean_pred = torch.mean(y_pred)
        
#         # Compute covariance and variances
#         cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
#         var_true = torch.var(y_true, unbiased=False)
#         var_pred = torch.var(y_pred, unbiased=False)
        
#         # Compute Pearson correlation coefficient
#         corr = cov / (torch.sqrt(var_true) * torch.sqrt(var_pred) + 1e-8)  # Add epsilon to avoid division by zero
#         return corr**2

    
class calculate_pearson_correlation(nn.Module):
    def __init__(self):
        super(calculate_pearson_correlation, self).__init__()

    def forward(self, gender_labels, predictions):
        # Compute means of the predicted probabilities and gender labels
        pred_mean = torch.mean(predictions)
        gender_mean = torch.mean(gender_labels)

        # Compute covariance
        covariance = torch.mean((predictions - pred_mean) * (gender_labels - gender_mean))

        # Compute standard deviations
        pred_std = torch.std(predictions)
        gender_std = torch.std(gender_labels)

        # Compute Pearson correlation
        correlation = covariance / (pred_std * gender_std + 1e-6)  # Adding a small constant to avoid division by zero
        
        return correlation ** 2
    


class InvCorrelationCoefficientLoss(nn.Module):
    """
    Custom loss function inverse of squared correlation coefficient.
    """
    def __init__(self):
        super(InvCorrelationCoefficientLoss, self).__init__()

    def forward(self, y_true, y_pred):
        mean_x = torch.mean(y_true)
        mean_y = torch.mean(y_pred)
        covariance = torch.mean((y_true - mean_x) * (y_pred - mean_y))
        std_x = torch.std(y_true)
        std_y = torch.std(y_pred)
        eps = 1e-5
        corr = covariance / (std_x * std_y + eps)
        return 1 - corr ** 2
    

# import torch

# class PearsonCorrelationLoss(torch.nn.Module):
#     """
#     Differentiable Pearson correlation loss for use in PyTorch models.
#     This can be used to compute the correlation between two variables, such as actual and predicted labels.
#     """
#     def __init__(self):
#         super(PearsonCorrelationLoss, self).__init__()

#     def forward(self, x, y):
#         """
#         Forward pass for computing Pearson correlation between x and y.

#         :param x: Tensor of actual values (e.g., actual gender, binary)
#         :param y: Tensor of predicted values (e.g., predicted gender, can be binary or continuous)
#         :return: Pearson correlation coefficient (between -1 and 1)
#         """
#         # Flatten the tensors to ensure they are 1D
#         x = x.view(-1)
#         y = y.view(-1)

#         # Calculate mean of x and y
#         x_mean = torch.mean(x)
#         y_mean = torch.mean(y)

#         # Center the data by subtracting the mean
#         xm = x - x_mean
#         ym = y - y_mean

#         # Calculate covariance and standard deviations
#         cov = torch.sum(xm * ym)
#         x_std = torch.sqrt(torch.sum(xm ** 2))
#         y_std = torch.sqrt(torch.sum(ym ** 2))

#         # Pearson correlation coefficient
#         correlation = cov / (x_std * y_std)

#         return correlation ** 2