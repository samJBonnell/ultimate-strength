'''
Samuel Bonnell - 2025-09-29
LASE MASc Student
'''

# Generic Imports
import os
import argparse
import numpy as np
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
from utils.mlp_utilities import MLP

from utils.json_utils import (
    load_random_records
)

from utils.pod_utilities import (
    extract_von_mises_stress,
    filter_valid_snapshots,
    training_data_constructor,
    plot_field,
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='POD-MLP Training Script')
    
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of layers in the MLP (default: 5)')
    parser.add_argument('--layer_size', type=int, default=10,
                        help='Size of each hidden layer (default: 10)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs (default: 2000)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save', type=bool, default=0,
                        help='Save (default: 0)')
    parser.add_argument('--num_modes', type=int, default=10,
                        help='Number of POD modes (default: 10)')
    
    return parser.parse_args()

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Parse command line arguments
    args = parse_args()
    
    print(f"Training with configuration:")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Layer size: {args.layer_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print()
    
    # Define input and output data locations
    input_path = Path("data/input.jsonl")
    output_path = Path("data/output.jsonl")

    # Load data records
    records = load_random_records(input_path, output_path, n=250)
    stress_vectors = extract_von_mises_stress(records)
    element_indices = [r.output.element_counts for r in records]

    # Extract parameters
    parameters = []
    for rec in records:
        row = [
            rec.input.t_panel,                    
            rec.input.pressure_location[0],       
            rec.input.pressure_location[1],       
            rec.input.pressure_patch_size[0],     
            rec.input.pressure_patch_size[1]      
        ]
        parameters.append(row)

    parameters = np.array(parameters)

    # Define parameter names that match your 5 variables
    parameter_names = ["t_panel", "pressure_x", "pressure_y", "patch_width", "patch_height"]
    print(f"Parameters shape: {parameters.shape}")
    print(f"Parameter names: {parameter_names}")

    # Organize the stress data into a format that can be easily read by the PODs algorithm

    max_field_index, max_field_indices = max(enumerate(element_indices), key=lambda x: sum(x[1]))
    template_stress_field = np.zeros((int(sum(max_field_indices))))

    training_data = []
    object_index_maps = []
    for i in range(len(stress_vectors)):
        field, index_map = training_data_constructor(
            stress_vectors[i],
            template_stress_field,
            max_field_indices,
            element_indices[i]
        )
        training_data.append(field)
        object_index_maps.append(index_map)

    expected_snapshot_length = int(len(training_data[max_field_index]))
    snapshots, parameters = filter_valid_snapshots(training_data, parameters, expected_snapshot_length)

    # Save the original uncentered snapshots
    original_snapshots_uncentered = [snap.copy() for snap in snapshots]

    # Now continue with your existing code
    snapshots = np.array(snapshots, dtype=np.float32)
    parameters = np.array(parameters, dtype=np.float32)

    # We want each snapshot to exist in the columns of our space
    snapshots = np.transpose(snapshots)
    parameters = np.transpose(parameters)

    # Centre the data about the mean of each sample across all samples
    mean_field = np.mean(snapshots, axis=1, keepdims=True)
    snapshots -= mean_field

    min_norm_parameters = 0
    max_norm_parameters = 1

    X_min_parameters = np.min(parameters)
    X_max_parameters = np.max(parameters)

    parameters = min_norm_parameters + ((parameters - X_min_parameters) * (max_norm_parameters - min_norm_parameters) / (X_max_parameters - X_min_parameters))

    # ---------------------------------------------------------------------------------------------------------
    # PODS
    U, s, Vt = svd(snapshots, full_matrices=True)

    # ---------------------------------------------------------------------------------------------------------
    # Energy contribution
    energy_per_mode = s**2
    total_energy = np.sum(s**2)
    relative_energy = energy_per_mode / total_energy
    cumulative_energy = np.cumsum(relative_energy)

    num_modes = args.num_modes

    U_reduced = U[:,:num_modes]
    modal_coefficients = U_reduced.T @ snapshots

    min_norm_coefficients = 0
    max_norm_coefficients = 1

    X_min_coefficients = np.min(modal_coefficients)
    X_max_coefficients = np.max(modal_coefficients)

    # Normalize the model_coefficents to range [a, b]
    modal_coefficients = min_norm_coefficients + ((modal_coefficients - X_min_coefficients) * (max_norm_coefficients - min_norm_coefficients) / (X_max_coefficients - X_min_coefficients))

    # Convert numpy.ndarray into Tensor for MLP
    snapshots = torch.from_numpy(snapshots).float()
    parameters = torch.from_numpy(parameters).float()
    modal_coefficients = torch.from_numpy(modal_coefficients).float()

    # Create an MLP object
    model = MLP(input_size=5, num_layers=args.num_layers, layers_size=args.layer_size, output_size=num_modes)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Send the training to our GPU if at all available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    indices = np.arange(parameters.shape[1])

    # Split data AND indices
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        parameters.T, modal_coefficients.T, indices,
        test_size=0.2,
        random_state=42
    )

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create dataloaders using command-line batch_size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter()

    # Training using command-line epochs
    model.train()
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
       
        for input, labels in train_loader:
            input = input.to(device)
            labels = labels.to(device)
           
            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, labels)
           
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            total_loss += loss.item()
        
        # Write the training loss to the SummaryWriter object
        avg_epoch_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)

    writer.flush()

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for input, labels in test_loader:
            input = input.to(device)
            labels = labels.to(device)
           
            # Forward pass
            outputs = model(input)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Visualization
    test_sample_idx = 5

    test_parameters = X_test[test_sample_idx]
    test_coefficients = y_test[test_sample_idx]

    # Denormalize test_coefficients
    test_coefficients = X_min_coefficients + (test_coefficients - min_norm_coefficients) * (X_max_coefficients - X_min_coefficients) / (max_norm_coefficients - min_norm_coefficients)

    U_reduced_tensor = torch.from_numpy(U_reduced).float().to(device)

    with torch.no_grad():
        test_parameters_tensor = test_parameters.unsqueeze(0).to(device)
        predicted_coefficients = model(test_parameters_tensor) 
        predicted_coefficients = predicted_coefficients.squeeze(0)

        # Renormalize the coefficients
        predicted_coefficients = X_min_coefficients + ((predicted_coefficients - min_norm_coefficients) / (max_norm_coefficients - min_norm_coefficients)) * (X_max_coefficients - X_min_coefficients)
        predicted_snapshot = U_reduced_tensor @ predicted_coefficients
        ground_truth_snapshot = U_reduced_tensor @ test_coefficients.to(device)

    predicted_snapshot = predicted_snapshot.cpu().numpy()
    ground_truth_snapshot = ground_truth_snapshot.cpu().numpy()

    # Denormalize test parameters
    test_parameters = X_min_parameters + (test_parameters.cpu().numpy() - min_norm_parameters) * (X_max_parameters - X_min_parameters) / (max_norm_parameters - min_norm_parameters)

    # Add the mean back to the predictions
    predicted_snapshot = predicted_snapshot + mean_field.squeeze()
    ground_truth_snapshot = ground_truth_snapshot + mean_field.squeeze()

    # Plot comparison: Ground Truth vs MLP Prediction
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_field(
        ax1,
        ground_truth_snapshot,
        records[0].input,
        object_index=0,
        object_index_map=object_index_maps[0]
    )
    ax1.set_title("Ground Truth (POD Reconstruction)")

    plot_field(
        ax2,
        predicted_snapshot,
        records[0].input,
        object_index=0,
        object_index_map=object_index_maps[0]
    )
    ax2.set_title(f"MLP-POD Prediction")

    # Map test_sample_idx back to the original dataset index
    original_idx = test_indices[test_sample_idx]

    # Get the original FEM data for this specific snapshot
    original_snapshot = original_snapshots_uncentered[original_idx]

    # Difference plot
    difference = ground_truth_snapshot - predicted_snapshot
    plot_field(
        ax3,
        original_snapshot,
        records[0].input,
        object_index=0,
        object_index_map=object_index_maps[0]
    )
    ax3.set_title("FEM Stress Field")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')

    # Display the test parameters
    param_str = ", ".join([f"{name}={test_parameters[i].item():.3f}" 
                            for i, name in enumerate(parameter_names)])
    fig.suptitle(f"Test Sample {test_sample_idx}: {param_str}")
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    N = 54

    # Map test_sample_idx back to the original dataset index
    original_idx = test_indices[test_sample_idx]

    # Get the original FEM data for this specific snapshot
    original_snapshot = original_snapshots_uncentered[original_idx]

    side_element_count = int(np.sqrt(len(original_snapshot)))
    panel_width_vector = np.linspace(-1.5, 1.5, side_element_count)

    # Extract cross-section at row N from all three datasets
    original_stress = original_snapshot[N*side_element_count: (N + 1)*side_element_count]
    pod_stress = ground_truth_snapshot[N*side_element_count: (N + 1)*side_element_count]
    predicted_stress = predicted_snapshot[N*side_element_count: (N + 1)*side_element_count]

    ax1.plot(panel_width_vector, original_stress / 1e6, label="fem", lw=0.9)
    ax1.plot(panel_width_vector, pod_stress / 1e6, label="pod", lw=0.9)
    ax1.plot(panel_width_vector, predicted_stress / 1e6, label="pod-mlp", lw=0.9)

    ax1.set_title(f"Cross-Section at Row {N} (Original Index: {original_idx})")
    ax1.set_xlabel("Panel Width (m)")
    ax1.set_ylabel("Stress (MPa)")
    plt.legend(title="data set", fontsize='small', fancybox=True, title_fontsize=7, loc='best')

    # plt.grid(True, which="both", ls="-", color='0.95')

    plt.tight_layout()
    plt.show()















    if args.save == 1:
        os.makedirs('mlp-models', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': 5,
            'num_layers': args.num_layers,
            'layer_size': args.layer_size,
            'output_size': num_modes,
        }, f'mlp-models/model_epoch_{epoch}_{timestamp}.pth')

if __name__ == '__main__':
    main()