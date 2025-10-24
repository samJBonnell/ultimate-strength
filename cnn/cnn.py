'''
Samuel Bonnell - 2025-10-22
LASE MASc Student
'''

# Generic Imports
import os
import string

import torch.optim.adam
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
from torchinfo import summary

from sklearn.model_selection import train_test_split

# Personal Definitions
from us_lib.normalization_utilities import NormalizationHandler

from us_lib.json_utilities import (
    load_records
)

from us_lib.data_utilities import (
    extract_attributes,
    # filter_valid_snapshots,
    # training_data_constructor,
    # plot_field,
)

from us_lib.cnn_utilities import EncoderBlock, DecoderBlock, Bridge, EncoderDecoderNetwork

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
    parser.add_argument('--path', type=str, default='./data/test/non-var-thickness',
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
    records = load_records(input_path, output_path)
    stress_vectors = extract_attributes(records, attributes= ['vm'])['vm']

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

    num_samples = len(parameters)
    num_features = len(parameters[0])
    # parameter_names = ["t_panel", "pressure_x", "pressure_y", "patch_width", "patch_height"]
    parameter_names = ["pressure_x", "pressure_y", "patch_width", "patch_height"]
    
    # Data input!
    input_matrix_size = int(np.sqrt(len(stress_vectors[0])))
    y = np.zeros((len(stress_vectors), input_matrix_size, input_matrix_size))
    for i, vector in enumerate(stress_vectors):
        np_vector = np.array(vector)
        y[i, :, :] = np_vector.reshape((input_matrix_size, input_matrix_size))
    # Need to create a patterning for the CNN interface
    # Currently, we have rows and columns that correspond to the size of input of the CNN
    # The snapshots are N x n vectors, parameters are d x 1 vectors. We need to create a 
    # sqrt(N) x sqrt(N) x n block for snapshots and a sqrt(N) x sqrt(N) x d input for each n_i example.

    # Create the n x d x N_i x N_i parameter input space:
    input_convolution_size = 80
    template_convolution = np.ones(shape=(input_convolution_size, input_convolution_size), dtype=float)
    X = np.ndarray(shape=(num_samples, num_features, input_convolution_size, input_convolution_size))

    for i, parameter_set in enumerate(parameters):
        # For each of the features, create an N_i x N_i input matrix that we set as the value of each feature across the entire matrix
        for j, value in enumerate(parameter_set):
            X[i, j, :, :] = (template_convolution.copy()) * value

    indices = np.arange(X.shape[0])
    # Create training and testing splits
    X_train, X_test, y_train, y_test, train_indicies, test_indicies = train_test_split(
        X, y, indices, 
        test_size = 0.2,
        random_state=None
    )

    # -------------------------------------------------------------------------------------------------------------------------
    # Normalize the data !AFTER! we split the data
    # -------------------------------------------------------------------------------------------------------------------------
    # We need to normalize the X_train and then normalize the X_test with the same values
    X_normalizer = NormalizationHandler(X_train, method = 'std', excluded_axis=[1])
    y_normalizer = NormalizationHandler(y_train, method = 'std', excluded_axis=[1, 2])

    # We need to noramlize the y_train and then normalize the y_test with the same values
    X_test = X_normalizer.normalize(X_test)
    y_test = y_normalizer.normalize(y_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------------------------------------------------------
    # Define the optimizer and the loss function
    # -------------------------------------------------------------------------------------------------------------------------
    model = EncoderDecoderNetwork(input_channels=num_features, output_channels = 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # We can now train a network!

    # -------------------------------------------------------------------------------------------------------------------------
    # Create the summary writer and Tensorboard writer
    # -------------------------------------------------------------------------------------------------------------------------
    summary(model, input_size=(1, 4, 80, 80))
    writer = SummaryWriter(log_dir="./cnn/runs/")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {num_params:,}\n")
    # -------------------------------------------------------------------------------------------------------------------------
    # Run the training of the model
    # -------------------------------------------------------------------------------------------------------------------------
    model.train()
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0

        for input, labels in train_loader:
            input = input.to(device)
            labels = labels.to(device)

            outputs = model(input)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

        avg_epoch_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)

    writer.flush()
    
    # -------------------------------------------------------------------------------------------------------------------------
    # Evaluate the performance of the model
    # -------------------------------------------------------------------------------------------------------------------------
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for input, labels in test_loader:
            input = input.to(device)
            labels = labels.to(device)

            outputs = model(input)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Error: {avg_test_loss:.4f}")


if __name__ == '__main__':
    main()