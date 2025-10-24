import torch
import torch.nn as nn

import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size, dropout=0.1, use_batch_norm=True):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
       
        # self.activation = nn.ReLU()
        self.activation = nn.LeakyReLU()
        # self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
       
        self.layers = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.layers.extend([nn.Linear(layers_size, layers_size) for _ in range(1, self.num_layers-1)])
        self.layers.append(nn.Linear(layers_size, output_size))
        
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(layers_size) for _ in range(self.num_layers-1)])
   
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
def weighted_mse_loss(predictions, targets, weights):
        """MSE loss weighted by POD mode importance"""
        squared_errors = (predictions - targets) ** 2
        weighted_errors = squared_errors * weights.unsqueeze(0)
        return weighted_errors.mean()
