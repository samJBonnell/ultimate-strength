'''
Samuel Bonnell - 2025-09-29
LASE MASc Student
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from pathlib import Path
import math

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

# Extract parameters with proper handling of lists
parameters = []
for rec in records:
    row = [
        rec.input.t_panel,                    # Single value
        rec.input.pressure_location[0],       # First element of list
        rec.input.pressure_location[1],       # Second element of list  
        rec.input.pressure_patch_size[0],     # First element of list
        rec.input.pressure_patch_size[1]      # Second element of list
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

# We want each snapshot to exist in the columns of our space
snapshots = list(np.transpose(snapshots))

# Centre the data about the mean of each sample across all samples
mean_field = np.mean(snapshots, axis=1, keepdims=True)
# centered_snapshots = snapshots - mean_field
snapshots -= mean_field

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

from utils.mlp_utilities import MLP
import torch
import torch.nn as nn

num_modes = 10

U_reduced = U[:,:num_modes] # Select the first N modes from the U matrix
modal_coefficients = U_reduced.T @ snapshots # Compute the modal_coefficients that will represent the ground truth of our model

# Convert numpy.ndarray into Tensor for MLP
snapshots = torch.from_numpy(snapshots)
parameters = torch.from_numpy(parameters)

# Create an MLP object that takes as input the number of parameters and output the number of modes we expect
model = MLP(input_size=5, num_layers=10, layers_size=50, output_size=num_modes)

print(model)
print(model(parameters).shape)