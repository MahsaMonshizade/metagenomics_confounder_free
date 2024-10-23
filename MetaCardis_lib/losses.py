# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dcor

    
class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, pred, target):
        print("correlation")
        print(target)
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
        return r ** 2


class CorrelationCoefficientLoss(nn.Module):
    def __init__(self):
        super(CorrelationCoefficientLoss, self).__init__()

    def forward(self, y_true, y_pred):
        bias_true = y_true
        bias_pred = y_pred
        bias_pred_centered = bias_pred - bias_pred.mean()
        bias_true_centered = bias_true - bias_true.mean()
        
        covariance = torch.mean(bias_pred_centered * bias_true_centered)
        std_pred = bias_pred.std()
        std_true = bias_true.std()
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        normalized_covariance = covariance / (std_pred * std_true + epsilon)
        
        # Since we want zero covariance, we can define the loss as the squared normalized covariance
        loss_adv = normalized_covariance ** 2
        return loss_adv





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