import torch
import torch.nn as nn

import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.encoder(x)

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            # First block
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Second block
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=0, output_padding=0),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Third block
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0, output_padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Fourth block
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=0, output_padding=0),
            nn.Conv2d(32, 32, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.decoder(x)

class Bridge(nn.Module):
    def __init__(self):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.bridge(x)

class EncoderDecoderNetwork(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(EncoderDecoderNetwork, self).__init__()
        self.encoder = EncoderBlock()
        self.bridge = Bridge()
        self.decoder = DecoderBlock()
        self.output_conv = nn.Conv2d(32, output_channels, kernel_size=1)
    
    def forward(self, x):
        enc_out = self.encoder(x)
        bridge_out = self.bridge(enc_out)
        dec_out = self.decoder(bridge_out)
        out = self.output_conv(dec_out)
        
        return out

# Load utilities for the data
class NormalizationHandler:
    """
    Manages the normalization and denormalization of a set of data
    so that we don't need to store this information in other areas of the program
    """

    def __init__(self, X, type='std', range=None):
        """
        Parameters
        ----------
        X : np.ndarray
            Data to normalize (2D matrix, 3D matrix)
        type : str
            'std' for standardization or 'bounds' for min-max normalization
        range : tuple, optional
            (min, max) for bounds normalization
        """

        self.X_original = X.copy()
        self.type = type
        self.range = range  if range is not None else (0, 1)

        self.shape = self.X_original.shape
        self.dim = self.shape.ndim

        if self.type == 'std':
            self.mean = np.mean(X, axis = self.dim - 1, keepdims=True)
            self.std = np.std(X, axis = self.dim - 1, keepdims=True)
            self.std = np.where(self.std < 1e-10, 1.0, self.std)
            self._X_norm = (X - self.mean) / self.std
            self.X_min = np.min(X, axis=self.dim - 1, keepdims=True)
            self.X_max = np.max(X, axis=self.dim - 1, keepdims=True)

        elif self.type == 'bounds':
            self.X_min = np.min(X, axis=self.dim - 1, keepdims=True)
            self.X_max = np.max(X, axis=self.dim - 1, keepdims=True)
            self._X_norm = self.range[0] + ((X - self.X_min) * (self.range[1] - self.range[0]) / (self.X_max - self.X_min))
            self.mean = None
            self.std = None

        else:
            raise Exception("Incorrect matrix size for normalization")

    def denormalize(self, X_norm):
        """Denormalize data back to the original scale"""
        original_shape = self.shape
        if self.type == 'std':
            result = (X_norm * self.std) + self.mean
        elif self.type == 'bounds':
            result = ((X_norm - self.range[0]) * (self.X_max - self.X_min) / (self.range[1] - self.range[0])) + self.X_min
        
        return result
    
    @property
    def X_norm(self):
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
    

# Load the data 
