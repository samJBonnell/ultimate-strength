'''
Samuel Bonnell - 2025-09-29
LASE MASc Student
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from pathlib import Path
import math
from utils.mlp_utilities import MLP
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from utils.json_utils import (
    load_random_records
)

from utils.pod_utilities import (
    extract_von_mises_stress,
    filter_valid_snapshots,
    training_data_constructor,
    plot_field,
)

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

snapshots = np.array(snapshots, dtype=np.float32)
parameters = np.array(parameters, dtype=np.float32)

# We want each snapshot to exist in the columns of our space
snapshots = np.transpose(snapshots)
parameters = np.transpose(parameters)

# Centre the data about the mean of each sample across all samples
mean_field = np.mean(snapshots, axis=1, keepdims=True) # Centre the data about the mean of each sample across all samples
snapshots -= mean_field # centered_snapshots = snapshots - mean_field

# ---------------------------------------------------------------------------------------------------------
# PODS
U, s, Vt = svd(snapshots, full_matrices=True)

# ---------------------------------------------------------------------------------------------------------
# Energy contribution
energy_per_mode = s**2
total_energy = np.sum(s**2)
relative_energy = energy_per_mode / total_energy
cumulative_energy = np.cumsum(relative_energy)

# n_modes_to_plot = 10

# # For 2D data (e.g., 100x100 grid flattened to 10000 points):
# nx, ny = 108, 108  # Replace with your actual grid dimensions

# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# axes = axes.flatten()

# for i in range(n_modes_to_plot):
#     # Reshape the mode from 1D to 2D
#     mode = U[:, i].reshape(nx, ny)
    
#     im = axes[i].imshow(mode, cmap='RdBu_r', aspect='auto')
#     axes[i].set_title(f'Mode {i+1}\n({relative_energy[i]*100:.2f}% energy)')
#     axes[i].axis('off')
#     # plt.colorbar(im, ax=axes[i], fraction=0.046)

# plt.suptitle('First 10 POD Modes', fontsize=16)
# plt.tight_layout()
# plt.show()

num_modes = 50

U_reduced = U[:,:num_modes] # Select the first N modes from the U matrix
modal_coefficients = U_reduced.T @ snapshots # Compute the modal_coefficients that will represent the ground truth of our model

# Convert numpy.ndarray into Tensor for MLP
snapshots = torch.from_numpy(snapshots).float()
parameters = torch.from_numpy(parameters).float()
modal_coefficients = torch.from_numpy(modal_coefficients).float()

# Create an MLP object that takes as input the number of parameters and output the number of modes we expect
model = MLP(input_size=5, num_layers=27, layers_size=200, output_size=num_modes)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")

# Send the training to our GPU if at all available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create training and validation sets using the snapshot and parameter data
# We want to compare the coefficients computed for each snapshots as the labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    parameters.T, modal_coefficients.T,
    test_size=0.2,
    random_state=42
)

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training
num_epochs = 1000
model.train()
for epoch in tqdm(range(num_epochs)):
    total_loss = 0
   
    for field, labels in train_loader:
        field = field.to(device)
        labels = labels.to(device)
       
        # Forward pass
        outputs = model(field)
        loss = criterion(outputs, labels)
       
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        total_loss += loss.item()

# Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for field, labels in test_loader:
        field = field.to(device)
        labels = labels.to(device)
       
        # Forward pass
        outputs = model(field)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# We now have a model that will predict the modal coefficients of our system to match the reduced order system
# We should plot the prediction against the ROM to determine how well it is working

test_sample_idx = 5

test_parameters = X_test[test_sample_idx]
test_coefficients = y_test[test_sample_idx]

U_reduced_tensor = torch.from_numpy(U_reduced).float().to(device)

with torch.no_grad():
    test_parameters_tensor = test_parameters.unsqueeze(0).to(device)
    predicted_coefficients = model(test_parameters_tensor) 
    predicted_coefficients = predicted_coefficients.squeeze(0)
    predicted_snapshot = U_reduced_tensor @ predicted_coefficients
    
    ground_truth_snapshot = U_reduced_tensor @ test_coefficients.to(device)

predicted_snapshot = predicted_snapshot.cpu().numpy()
ground_truth_snapshot = ground_truth_snapshot.cpu().numpy()

# Add the mean back to the predictions
predicted_snapshot = predicted_snapshot + mean_field.squeeze()
ground_truth_snapshot = ground_truth_snapshot + mean_field.squeeze()

# Plot comparison: Ground Truth vs MLP Prediction
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Note: You'll need to find the original record index if you want to use the exact input/index_map
# For now, using a representative one
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

# Difference plot
difference = ground_truth_snapshot - predicted_snapshot
plot_field(
    ax3,
    difference,
    records[0].input,
    object_index=0,
    object_index_map=object_index_maps[0]
)
ax3.set_title("Difference (Ground Truth - Predicted)")

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