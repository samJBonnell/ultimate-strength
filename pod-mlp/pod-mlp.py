'''
Samuel Bonnell - 2025-09-29
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
from torchinfo import summary

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from scipy.linalg import svd

# Personal Definitions
from us_lib.mlp_utilities import MLP, weighted_mse_loss
from us_lib.normalization_utilities import NormalizationHandler
from us_lib.json_utilities import load_records
from us_lib.pod_utilities import training_data_constructor, plot_field
from us_lib.data_utilities import extract_attributes

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
    parser.add_argument('--path', type=str, default='./data/test/non-var-thickness',
                        help='Path to trial data relative to pod-mlp.py')
    
    return parser.parse_args()

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Parse command line arguments
    args = parse_args()

    print(f"Training Path: {args.path}\n")
    
    print(f"Training with configuration:")
    print(f"  - Number of layers: {args.num_layers}")
    print(f"  - Layer size: {args.layer_size}")
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

    num_samples = len(parameters)
    num_features = len(parameters[0])

    parameter_names = ["t_panel", "pressure_x", "pressure_y", "patch_width", "patch_height"]
    parameter_names = ["pressure_x", "pressure_y", "patch_width", "patch_height"]

    max_field_index, max_field_indices = max(enumerate(element_indices), key=lambda x: sum(x[1]))
    template_stress_field = np.zeros((int(sum(max_field_indices))))

    stress_fields = []
    object_index_maps = []
    for i in range(len(stress_vectors)):
        field, index_map = training_data_constructor(
            stress_vectors[i],
            template_stress_field,
            max_field_indices,
            element_indices[i]
        )
        stress_fields.append(field)
        object_index_maps.append(index_map)

    expected_snapshot_length = int(len(stress_fields[max_field_index]))
    # X, y = filter_valid_snapshots(stress_fields, parameters, expected_snapshot_length)

    X = np.array(parameters, dtype=np.float32)
    stress_fields = np.array(stress_fields, dtype=np.float32) / 1e6 # Convert Pa to MPa

    # Center the stress fields
    mean_field = np.mean(stress_fields, axis=1, keepdims=True)
    stress_fields = stress_fields - mean_field

    # Perform SVD on the stress fields set to compute the major modes
    U, s, Vt = svd(stress_fields.T, full_matrices=True)
    U_reduced = U[:, :args.num_modes]
    y = U_reduced.T @ stress_fields.T

    # -------------------------------------------------------------------------------------------------------------------------
    # Break the data into training and test sets
    # -------------------------------------------------------------------------------------------------------------------------
    indices = np.arange(X.shape[0])
    print(len(indices))
    X_train, X_test, y_train, y_test, train_indicies, test_indicies = train_test_split(
        X, y.T, indices, 
        test_size = 0.2,
        random_state=None
    )

    # -------------------------------------------------------------------------------------------------------------------------
    # Normalize the data !AFTER! we split the data
    # -------------------------------------------------------------------------------------------------------------------------

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
    # Create MLP object
    # -------------------------------------------------------------------------------------------------------------------------
    model = MLP(input_size=len(parameter_names), num_layers=args.num_layers, layers_size=args.layer_size, output_size=args.num_modes, dropout=0.05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # -------------------------------------------------------------------------------------------------------------------------
    # Define the optimizer and the loss function
    # -------------------------------------------------------------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    singular_values = s[:args.num_modes]
    mode_weights = torch.from_numpy(singular_values / singular_values.sum()).float().to(device)
    
    summary(model, input_size=(250, 4))
    writer = SummaryWriter(log_dir="./pod-mlp/runs/")

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
           
            # Forward pass
            outputs = model(input)
            # loss = criterion(outputs, labels)
            loss = weighted_mse_loss(outputs, labels, mode_weights)
           
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            total_loss += loss.item()
        
        # Write the training loss to the SummaryWriter object
        avg_epoch_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)

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
           
            # Forward pass
            outputs = model(input)
            # loss = criterion(outputs, labels)
            loss = weighted_mse_loss(outputs, labels, mode_weights)
            
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")

    # -------------------------------------------------------------------------------------------------------------------------
    # Visualize the compare the results of the model to the original and POD data
    # -------------------------------------------------------------------------------------------------------------------------
    test_sample_idx = 9
    test_parameters = X_test[test_sample_idx]
    test_coefficients = y_test[test_sample_idx]

    with torch.no_grad():
        test_parameters_tensor = test_parameters.unsqueeze(0).to(device)
        predicted_coefficients_norm = model(test_parameters_tensor).squeeze(0).cpu().numpy()
        
        # Denormalize coefficients
        test_coef_denorm = coef_normalizer.denormalize(test_coefficients.cpu().numpy())
        pred_coef_denorm = coef_normalizer.denormalize(predicted_coefficients_norm)
        
        pod_snapshot = (U_reduced @ test_coef_denorm) + mean_field.squeeze()
        predicted_snapshot = (U_reduced @ pred_coef_denorm) + mean_field.squeeze()

    # Map test_sample_idx back to the original dataset index
    original_idx = test_indices[test_sample_idx]

    # Get the original FEM data for this specific snapshot
    fem_snapshot = original_snapshots[original_idx]

    vmin = min(fem_snapshot.min(), pod_snapshot.min(), predicted_snapshot.min())
    vmax = max(fem_snapshot.max(), pod_snapshot.max(), predicted_snapshot.max())

    # After POD, check reconstruction error
    pod_error = np.mean([np.linalg.norm(snapshots[:, i] - (U_reduced @ modal_coefficients[:, i] + mean_field.squeeze())) 
                        for i in range(snapshots.shape[1])])
    print(f"Average POD reconstruction error: {pod_error:.4f} MPa")

    # After the prediction section
    print("\nCoefficient Comparison:")
    print("Mode | True (norm) | Pred (norm) | True (denorm) | Pred (denorm)")
    print("-" * 70)
    for i in range(args.num_modes):
        true_norm = test_coefficients[i].item()
        pred_norm = predicted_coefficients_norm[i]
        true_denorm = test_coef_denorm[i]
        pred_denorm = pred_coef_denorm[i]
        print(f"{i:4d} | {true_norm:11.4f} | {pred_norm:11.4f} | {true_denorm:13.4f} | {pred_denorm:13.4f}")

    # Calculate per-coefficient error
    coef_errors = np.abs(test_coef_denorm - pred_coef_denorm)
    print(f"\nCoefficient-wise MAE: {coef_errors.mean():.4f}")
    print(f"Max coefficient error: {coef_errors.max():.4f} (mode {coef_errors.argmax()})")

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

    N = 80

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
            'input_size': len(parameter_names),
            'num_layers': args.num_layers,
            'layer_size': args.layer_size,
            'output_size': args.num_modes,
        }, f'mlp-models/model_epoch_{epoch}_{timestamp}.pth')

if __name__ == '__main__':
    main()