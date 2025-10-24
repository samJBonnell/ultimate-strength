import numpy as np
from typing import List, Dict, Tuple

def plot_field(axis, field, levels = 20, vmin=None, vmax=None):
    """
    Plot the field, directly as provided to this function
    
    Parameters:
    -----------
    axis : matplotlib axis
        The axis to plot on
    field : np.ndarray
        Field to plot
    vmin : float, optional
        Minimum value for colormap scaling
    vmax : float, optional
        Maximum value for colormap scaling
    
    Returns:
    --------
    contour : matplotlib contour object
        The contour plot object (useful for adding colorbars)
    """
    # Extract geometry
    width = field.shape[0]
    length = field.shape[1]
    
    # Create meshgrid
    x = np.linspace(0, width, width)
    y = np.linspace(0, length, length)
    X, Y = np.meshgrid(x, y)
    z = field.reshape(width, length)
    
    # Plot with optional vmin/vmax
    return axis.contourf(X, Y, z, cmap='RdBu_r', vmin=vmin, vmax=vmax, levels=levels)