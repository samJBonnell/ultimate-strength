"""
samples.py contains code related to the creation of sample sets of data, and can be used to separate your information
into training, validation, and test sets of data.
"""
# Generic imports
import numpy as np
from typing import Optional, Tuple, Generator

def create_folds(X: np.ndarray, num_folds: int = 0, relative_size: Optional[float] = None, save_to_file: bool = False, file_name: Optional[str] = None) -> Optional[np.ndarray]:
    """
    create_folds takes as input a set of data and creates a mapping in the form of an (N x k) matrix indicating
    whether or not a row (N_i) is part of the training or validation set for each fold (k)
    
    Parameters:
    -----------
    X : np.ndarray 
        Dataset that you wish to split into training and validation sets. Must be a 2D array.
    num_folds : int
        Number of folds you wish to create. Must be >= 2. In the case that the number of folds requested 
        is not a multiple of the number of rows of the input field, the remainder of points will be 
        distributed across the first (N % k) folds.
    relative_size : float, optional
        Creates folds using a percentage of the input field instead of a fixed number. 
        Ex. relative_size = 0.2 results in 5 folds. Must be between 0 and 1.
    save_to_file : bool
        If True, saves the created folds map to the file specified by 'file_name'
    file_name : str, optional
        The file name to save the folds mapping. Required if save_to_file is True.
        
    Returns:
    --------
    mask_matrix : np.ndarray
        Array where each column corresponds to a single fold and each row indicates whether the data is
        included in the training set (True) or the validation set (False). 
        Returns None if save_to_file is True and file_name is not provided.
        
    Raises:
    -------
    ValueError
        If input validation fails
    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy ndarray")
    
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}")
    
    if num_folds == 0 and relative_size is None:
        raise ValueError("Must specify either num_folds or relative_size")
    
    if relative_size is not None:
        if not 0 < relative_size < 1:
            raise ValueError(f"relative_size must be between 0 and 1, got {relative_size}")
        num_folds = int(1 / relative_size)
    
    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}")
    
    if save_to_file and file_name is None:
        raise ValueError("file_name must be provided when save_to_file is True")
    
    n, d = X.shape
    
    if num_folds > n:
        raise ValueError(f"num_folds ({num_folds}) cannot exceed number of samples ({n})")
    
    # Compute the width of a single fold and the remainder to be added to the first folds
    fold_width = n // num_folds
    fold_division_remainder = n % num_folds
    
    # Create a matrix that stores a selection mapping for each of the folds in the set
    mask_matrix = np.ones((n, num_folds), dtype=bool)
    
    # For each fold in the training data, we want to set the validation data region to False
    start = 0
    for fold in range(num_folds):
        # Add 1 extra sample to the first (fold_division_remainder) folds to evenly distribute remainder
        end = start + fold_width + (1 if fold < fold_division_remainder else 0)
        mask_matrix[start:end, fold] = False
        start = end
    
    # Save to file if requested
    if save_to_file:
        np.save(file_name, mask_matrix, allow_pickle=True)
    
    return mask_matrix


def iterate_folds(X: np.ndarray, mask_matrix: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields training and validation splits for each fold.
    
    Parameters:
    -----------
    X : np.ndarray
        Dataset to split
    mask_matrix : np.ndarray
        Boolean mask matrix from create_folds where True = training, False = validation
        
    Yields:
    -------
    X_train, X_val : tuple of np.ndarray
        Training and validation data for each fold
        
    Example:
    --------
    X = np.random.rand(100, 10)
    masks = create_folds(X, num_folds=5)
    for fold_idx, (X_train, X_val) in enumerate(iterate_folds(X, masks)):
        print(f"Fold {fold_idx}: train={len(X_train)}, val={len(X_val)}")
    """
    if X.shape[0] != mask_matrix.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but mask_matrix has {mask_matrix.shape[0]} rows")
    
    num_folds = mask_matrix.shape[1]
    
    for fold in range(num_folds):
        train_mask = mask_matrix[:, fold]
        val_mask = ~train_mask
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        
        yield (X_train, X_val)