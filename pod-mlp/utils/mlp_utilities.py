import torch
import torch.nn as nn

import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size, dropout=0.1):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
        
        self.layers = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.layers.extend([nn.Linear(layers_size, layers_size) for i in range(1, self.num_layers-1)])
        self.layers.append(nn.Linear(layers_size, output_size))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
class NormalizationHandler:
    """
    Handles normalization and denormalization of data with multiple strategies.
    """
    def __init__(self, X, type='std', range=None):
        """
        Parameters
        ----------
        X : np.ndarray
            Data to normalize (can be vector or matrix)
        type : str
            'std' for standardization or 'bounds' for min-max normalization
        range : tuple, optional
            (min, max) for bounds normalization
        """
        self.X_original = X.copy()
        self.type = type
        self.range = range if range is not None else (0, 1)
        
        if type == 'std':
            self.mean = np.mean(X, axis=1, keepdims=True)
            self.std = np.std(X, axis=1, keepdims=True)
            self.std = np.where(self.std < 1e-10, 1.0, self.std)
            self._X_norm = (X - self.mean) / self.std
            self.X_min = np.min(X, axis=1, keepdims=True)
            self.X_max = np.max(X, axis=1, keepdims=True)
            
        elif type == 'bounds':
            self.X_min = np.min(X, axis=1, keepdims=True)
            self.X_max = np.max(X, axis=1, keepdims=True)
            self._X_norm = self.range[0] + ((X - self.X_min) * (self.range[1] - self.range[0]) / (self.X_max - self.X_min))
            self.mean = None
            self.std = None
            
        else:
            raise ValueError(f"Unknown normalization type: {type}")
    
    def denormalize(self, X_norm):
        """Denormalize data back to original scale"""
        # Handle 1D input by reshaping to column vector
        original_shape = X_norm.shape
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(-1, 1)
        
        if self.type == 'std':
            result = (X_norm * self.std) + self.mean
        elif self.type == 'bounds':
            result = ((X_norm - self.range[0]) * (self.X_max - self.X_min) / (self.range[1] - self.range[0])) + self.X_min
        
        # Return in original shape
        if len(original_shape) == 1:
            result = result.flatten()
        
        return result
    
    @property
    def X_norm(self):
        """Return normalized data"""
        return self._X_norm
    
    def to_torch(self, device='cpu'):
        """Convert normalization parameters to torch tensors"""
        params = {
            'type': self.type,
            'X_min': torch.tensor(self.X_min, dtype=torch.float32, device=device),
            'X_max': torch.tensor(self.X_max, dtype=torch.float32, device=device),
        }
        if self.type == 'std':
            params['mean'] = torch.from_numpy(self.mean).float().to(device)
            params['std'] = torch.from_numpy(self.std).float().to(device)
        elif self.type == 'bounds':
            params['range'] = torch.tensor(self.range, dtype=torch.float32, device=device)
        return params