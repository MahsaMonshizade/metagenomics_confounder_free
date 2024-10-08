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
        return corr ** 2

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