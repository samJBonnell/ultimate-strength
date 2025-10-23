'''
Samuel Bonnell - 2025-10-22
LASE MASc Student
'''

# Generic Imports
import os
import string
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
np.set_printoptions(linewidth=200)
from datetime import datetime

import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from scipy.linalg import svd

# Personal Definitions
from utils.mlp_utilities import MLP, NormalizationHandler

from utils.json_utils import (
    load_random_records
)

from utils.data_utilities import (
    extract_von_mises_stress
    # filter_valid_snapshots,
    # training_data_constructor,
    # plot_field,
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CNN Training Script')
    
    # parser.add_argument('--num_layers', type=int, default=5,
    #                     help='Number of layers in the MLP (default: 5)')
    # parser.add_argument('--layer_size', type=int, default=10,
    #                     help='Size of each hidden layer (default: 10)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=bool, default=0,
                        help='Save (default: 0)')
    parser.add_argument('--path', type=str, default='data/non-var-thickness',
                        help='Path to trial data relative to pod-mlp.py')
    
    return parser.parse_args()

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Parse command line arguments
    args = parse_args()

    print(f"Training Path: {args.path}\n")
    
    print(f"Training with configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    
    # Define input and output data locations
    input_path = Path(f"{args.path}/input.jsonl")
    output_path = Path(f"{args.path}/output.jsonl")

    if not input_path.exists() or not output_path.exists():
        print(f"\nInput or output path does not exist\nExiting")
        return
    
    # Load data records
    records = load_random_records(input_path, output_path, n=250)
    stress_vectors = extract_von_mises_stress(records)
    element_indices = [r.output.element_counts for r in records]

    # Extract parameters
    parameters = []
    for rec in records:
        row = [
            # rec.input.t_panel,                    
            rec.input.pressure_location[0],       
            rec.input.pressure_location[1],       
            rec.input.pressure_patch_size[0],     
            rec.input.pressure_patch_size[1]      
        ]
        parameters.append(row)

    parameters = np.array(parameters)

    # parameter_names = ["t_panel", "pressure_x", "pressure_y", "patch_width", "patch_height"]
    parameter_names = ["pressure_x", "pressure_y", "patch_width", "patch_height"]

    stress_matrix = np.zeros((len(stress_vectors), int(np.sqrt(len(stress_vectors[0]))), int(np.sqrt(len(stress_vectors[0])))))
    

    # Need to create a patterning for the CNN interface
    # Currently, we have rows and columns that correspond to the size of input of the CNN
    # The snapshots are N x n vectors, parameters are d x 1 vectors. We need to create a 
    # sqrt(N) x sqrt(N) x n block for snapshots and a sqrt(N) x sqrt(N) x d input for each n_i
    # example.

if __name__ == '__main__':
    main()