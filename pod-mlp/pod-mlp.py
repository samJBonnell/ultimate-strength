'''
Samuel Bonnell - 2025-09-29
LASE MASc Student
'''

# Generic Imports
import os
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

    # Save the original snapshots
    original_snapshots = [snap.copy() for snap in snapshots]

    # After loading and organizing data, convert to MPa
    snapshots = np.array(snapshots, dtype=np.float32) / 1e6  # Convert Pa to MPa
    original_snapshots = [snap / 1e6 for snap in original_snapshots]

    parameters = np.array(parameters, dtype=np.float32)

    snapshots = np.transpose(snapshots)
    parameters = np.transpose(parameters)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Normalize the data
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Normalize parameters
    # Center snapshots (POD works best with centered data)
    mean_field = np.mean(snapshots, axis=1, keepdims=True)
    snapshots_centered = snapshots - mean_field

    # POD on centered (but not scaled) data
    U, s, Vt = svd(snapshots_centered, full_matrices=True)
    U_reduced = U[:, :args.num_modes]
    modal_coefficients = U_reduced.T @ snapshots_centered

    # NOW normalize only the coefficients and parameters
    param_normalizer = NormalizationHandler(parameters, type='bounds', range=(0, 1))
    parameters_norm = param_normalizer.X_norm

    coef_normalizer = NormalizationHandler(modal_coefficients, type='bounds', range=(0, 1))
    modal_coefficients_norm = coef_normalizer.X_norm

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create MLP object
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    model = MLP(input_size=5, num_layers=args.num_layers, layers_size=args.layer_size, output_size=args.num_modes)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Convert the data into a torch-compatible format
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Convert to torch
    # !!! AFTER THIS POINT, WE MUST REFER TO THE NORMALIZED VERSIONS OF VALUES
    parameters_norm = torch.from_numpy(parameters_norm).float()
    modal_coefficients_norm = torch.from_numpy(modal_coefficients_norm).float()

    # Send the training to our GPU if at all available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create a tracker for the index of each training and test sample so we can recover for comparison after training
    indices = np.arange(parameters_norm.shape[1])

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create datasets
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Split data AND indices
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        parameters_norm.T, modal_coefficients_norm.T, indices,
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

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Run the training of the model
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Evaluate the performance of the model
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Visualize the compare the results of the model to the original and POD data
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    test_sample_idx = 6
    test_parameters = X_test[test_sample_idx]
    test_coefficients = y_test[test_sample_idx]

    # Try with the training data, because the training error is SO low
    test_parameters = X_train[test_sample_idx]
    test_coefficients = y_train[test_sample_idx]

    with torch.no_grad():
        # Get predictions (normalized)
        test_parameters_tensor = test_parameters.unsqueeze(0).to(device)
        predicted_coefficients_norm = model(test_parameters_tensor).squeeze(0).cpu().numpy()
        
        # Denormalize coefficients
        test_coef_denorm = coef_normalizer.denormalize(test_coefficients.cpu().numpy())
        pred_coef_denorm = coef_normalizer.denormalize(predicted_coefficients_norm)
        
        # Reconstruct (scaled space) - using NumPy
        pod_snapshot = (U_reduced @ test_coef_denorm) + mean_field.squeeze()
        predicted_snapshot = (U_reduced @ pred_coef_denorm) + mean_field.squeeze()

    # Map test_sample_idx back to the original dataset index
    original_idx = test_indices[test_sample_idx]
    train_idx = train_indices[test_sample_idx]

    # Get the original FEM data for this specific snapshot
    fem_snapshot = original_snapshots[original_idx]
    fem_snapshot = original_snapshots[train_idx]

    # Calculate global min and max across all three fields for uniform scale
    vmin = min(fem_snapshot.min(), pod_snapshot.min(), predicted_snapshot.min())
    vmax = max(fem_snapshot.max(), pod_snapshot.max(), predicted_snapshot.max())

    # Plot comparison: Ground Truth vs MLP Prediction vs FEM
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    im1 = plot_field(
        ax1,
        fem_snapshot,
        records[0].input,
        object_index=0,
        object_index_map=object_index_maps[0],
        levels=10,
        vmin=vmin,
        vmax=vmax
    )
    ax1.set_title("FEM Stress Field")

    im2 = plot_field(
        ax2,
        pod_snapshot,
        records[0].input,
        object_index=0,
        object_index_map=object_index_maps[0],
        levels=10,
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_title("Ground Truth (POD Reconstruction)")

    im3 = plot_field(
        ax3,
        predicted_snapshot,
        records[0].input,
        object_index=0,
        object_index_map=object_index_maps[0],
        levels=10,
        vmin=vmin,
        vmax=vmax
    )
    ax3.set_title(f"MLP-POD Prediction")

    # Add a colorbar (you can choose which image to use, they all have the same scale)
    # fig.colorbar(im3, ax=[ax1, ax2, ax3], label='Von Mises Stress (Pa)', fraction=0.046, pad=0.04)

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

    side_element_count = int(np.sqrt(len(fem_snapshot)))
    panel_width_vector = np.linspace(-1.5, 1.5, side_element_count)

    # Extract cross-section at row N from all three datasets
    original_stress = fem_snapshot[N*side_element_count: (N + 1)*side_element_count]
    pod_stress = pod_snapshot[N*side_element_count: (N + 1)*side_element_count]
    predicted_stress = predicted_snapshot[N*side_element_count: (N + 1)*side_element_count]

    ax1.plot(panel_width_vector, original_stress, label="fem", lw=0.9)
    ax1.plot(panel_width_vector, pod_stress, label="pod", lw=0.9)
    ax1.plot(panel_width_vector, predicted_stress, label="pod-mlp", lw=0.9)

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
            'output_size': args.num_modes,
        }, f'mlp-models/model_epoch_{epoch}_{timestamp}.pth')

if __name__ == '__main__':
    main()