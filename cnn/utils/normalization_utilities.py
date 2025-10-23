import torch
import torch.nn as nn

import numpy as np

# Load utilities for the data
class NormalizationHandler:
    """
    Manages the normalization and denormalization of a set of data
    so that we don't need to store this information in other areas of the program
    """

    def __init__(self, X, method='std', bounds=None, excluded_axis = None):
        """
        Parameters
        ----------
        X : np.ndarray
            Data to normalize (1D vector, 2D matrix, 3D matrix, 4D matrix, etc)
        method : str
            'std' for standardization or 'bounds' for min-max normalization
        bounds : tuple, optional
            (min, max) for bounds normalization
        excluded_axis : List[int] (default: None)
            NumPy array axis to exclude from normalization
            Example:
                X = np.array([[1, 2], [3, 4], [5, 6]])
                self._std(X, dim=1, excluded_axis=[1])  # Per-feature normalization
        """

        self.X = X.copy()
        self.method = method
        self.bounds = bounds  if bounds is not None else (0, 1)

        if excluded_axis is None:
            excluded_axis = []

        self.shape = X.shape
        self.dim = X.ndim
        
        self.axis = tuple(i for i in range(self.dim) if i not in excluded_axis)

        methods = {
            'std' : self._std,
            'bounds' : self._bounds
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

        self._normalize = methods[method]
        self._normalize()

    # function to normalize a second matrix, of the same size, by the same amounts as the original set
    def normalize(self, X):
        """
        Normalize a separate set of data by the same values as the original set
        """
        self._check_dimensions(X)
        X = self._normalize(X)
        return X

    def denormalize(self, X_norm):
        """Denormalize data back to the original scale"""
        self._check_dimensions(X_norm)
        if self.method == 'std':
            result = (X_norm * self.std) + self.mean
        elif self.method == 'bounds':
            result = ((X_norm - self.bounds[0]) * (self.X_max - self.X_min) / (self.bounds[1] - self.bounds[0])) + self.X_min
        return result

    def _check_dimensions(self, X):
        """Check that the dimensions of the input and output matrix agree"""
        if X.shape[1:] != self.shape[1:]:
            raise ValueError(f"Normalization Error - mismatched array sizes\nRequired:{self.shape}\nProvided:{X.shape}")

    def _std(self, X=None):
        if X is None:
            self.mean = np.mean(self.X, axis = self.axis, keepdims=True)
            self.std = np.std(self.X, axis = self.axis, keepdims=True)
            self.std = np.where(self.std < 1e-10, 1.0, self.std)
            self._X_norm = (self.X - self.mean) / self.std
            self.X_min = None
            self.X_max = None
        else:
            return (X - self.mean) / self.std

    def _bounds(self, X=None):
        if X is None:
            self.X_min = np.min(self.X, axis=self.axis, keepdims=True)
            self.X_max = np.max(self.X, axis=self.axis, keepdims=True)
            self._X_norm = self.bounds[0] + ((self.X - self.X_min) * (self.bounds[1] - self.bounds[0]) / (self.X_max - self.X_min))
            self.mean = None
            self.std = None
        else:
            return self.bounds[0] + ((X - self.X_min) * (self.bounds[1] - self.bounds[0]) / (self.X_max - self.X_min))
    
    @property
    def X_norm(self):
        return self._X_norm
    
    def to_torch(self, device='cpu'):
        """Convert normalization parameters to torch tensors"""
        params = {
            'method': self.method,
            'X_min': torch.tensor(self.X_min, dtype=torch.float32, device=device),
            'X_max': torch.tensor(self.X_max, dtype=torch.float32, device=device),
        }
        if self.method == 'std':
            params['mean'] = torch.from_numpy(self.mean).float().to(device)
            params['std'] = torch.from_numpy(self.std).float().to(device)
        elif self.method == 'bounds':
            params['bounds'] = torch.tensor(self.bounds, dtype=torch.float32, device=device)
        return params