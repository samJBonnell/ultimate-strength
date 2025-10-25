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
from us_lib.data.normalization import NormalizationHandler
from us_lib.data.reader import load_records
from us_lib.data.parsing import extract_attributes
from us_lib.models.cnn import EncoderDecoderNetwork
from us_lib.visuals import plot_field

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CNN Training Script')
    
    # parser.add_argument('--num_layers', type=int, default=4,
    #                     help='Number of layers in the MLP (default: 4)')
    # parser.add_argument('--layer_size', type=int, default=16,
    #                     help='Size of each hidden layer (default: 16)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=bool, default=0,
                        help='Save (default: 0)')
    # parser.add_argument('--num_modes', type=int, default=10,
    #                     help='Number of POD modes (default: 10)')
    parser.add_argument('--path', type=str, default='./data/test/non-var-thickness',
                        help='Path to trial data relative to pod-mlp.py')
    parser.add_argument('--verbose', type=bool, default=0,
                        help='Print the structure of the network (default: 0)')
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
        y[i, :, :] = np_vector.reshape((input_matrix_size, input_matrix_size)) / 1e6
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

    parameters = np.array(parameters)
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
    X_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1])
    y_normalizer = NormalizationHandler(method = 'std', excluded_axis=[1, 2])

    # We need to noramlize the y_train and then normalize the y_test with the same values
    X_train = X_normalizer.fit_normalize(X_train)
    y_train = y_normalizer.fit_normalize(y_train)

    X_test = X_normalizer.normalize(X_test)
    y_test = y_normalizer.normalize(y_test)

    # Convert into torch compatible versions
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
    if args.verbose:
        summary(model, input_size=(1, 4, 80, 80))
    writer = SummaryWriter(log_dir=f"./cnn/runs/{timestamp}")

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

    # -------------------------------------------------------------------------------------------------------------------------
    # Visualize the compare the results of the model to the original and POD data
    # -------------------------------------------------------------------------------------------------------------------------

    test_idx = 9
    X_test = X_test[test_idx].unsqueeze(0).to(device)
    y_test = y_test[test_idx].cpu().numpy()

    with torch.no_grad():
        y_pred = model(X_test).squeeze(0).cpu().numpy()

    y_test = y_normalizer.denormalize(y_test)
    y_pred = y_normalizer.denormalize(y_pred)

    y_test = np.squeeze(y_test)
    y_pred = np.squeeze(y_pred)

    # Map test_idx back to the original dataset index
    test_idx = test_indicies[test_idx]

    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    # Plot comparison: Ground Truth vs MLP Prediction vs FEM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    im1 = plot_field(
        ax1,
        y_test,
        levels=10,
        vmin=vmin,
        vmax=vmax
    )
    ax1.set_title("FEM Stress Field")

    im2 = plot_field(
        ax2,
        y_pred,
        levels=10,
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_title("CNN Prediction")

    # fig.colorbar(im3, ax=[ax1, ax2, ax3], label='Von Mises Stress (Pa)', fraction=0.046, pad=0.04)

    for ax in (ax1, ax2):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')

    # Display the test parameters
    param_str = ", ".join([f"{name}={parameters[i, j]:.3f}" 
                        for j, name in enumerate(parameter_names)])
    fig.suptitle(f"Test Sample {test_idx}: {param_str}")
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    N = 80

    side_element_count = int(np.sqrt(len(y_test.flatten())))  # Or just use y_test.shape[0]
    panel_width_vector = np.linspace(-1.5, 1.5, side_element_count)

    # Extract cross-section at row N from 2D arrays
    y_test_section = y_test[N, :]  # Get row N
    y_pred_section = y_pred[N, :]  # Get row N

    ax1.plot(panel_width_vector, y_test_section, label="fem", lw=0.9)
    ax1.plot(panel_width_vector, y_pred_section, label="cnn", lw=0.9)
    ax1.set_title(f"Cross-Section at Row {N} (Example: {test_idx})")
    ax1.set_xlabel("Panel Width (m)")

    ax1.set_ylabel("Stress (MPa)")
    plt.legend(title="data set", fontsize='small', fancybox=True, title_fontsize=7, loc='best')
    # plt.grid(True, which="both", ls="-", color='0.95')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()