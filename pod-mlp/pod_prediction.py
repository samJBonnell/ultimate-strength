import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from ezyrb import POD, RBF
from ezyrb import ReducedOrderModel as ROM

from utils.json_utils import (
    load_random_records
)

from utils.pod_utilities import (
    extract_von_mises_stress,
    filter_valid_snapshots,
    create_ROM,
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

model_order = min(100, len(snapshots))  # Ensure we don't exceed number of snapshots

print(f"\nCreating POD Model with {len(snapshots)} snapshots")
print(f"Model Order: {model_order}")

# Create sample for training
sample_points = np.random.choice(a=len(snapshots), size=min(model_order, len(snapshots)), replace=False)
training_parameters = parameters[sample_points, :]
training_snapshots = [snapshots[i] for i in sample_points]

try:
    db, rom, pod, rbf = create_ROM(parameters, snapshots)
    print("ROM creation successful")
except Exception as e:
    print(f"ROM creation failed: {e}")
    exit()

# Define a loop to allow for parameter modification
while True:
    rank_input = input("Enter the POD truncation rank (or 'q' to quit): ")
    if rank_input.lower() in {'q', 'quit'}:
        print("Exiting.")
        break
    try:
        N = int(rank_input)
        if N > len(snapshots):
            print(f"Rank {N} exceeds number of snapshots ({len(snapshots)}). Using max available.")
            N = len(snapshots)
    except ValueError:
        print("Invalid input. Try again.")
        continue

    # Truncated POD + ROM
    pod_truncated = POD('svd', rank=N)
    pod_truncated.fit(snapshots)
    rom_truncated = ROM(db, pod_truncated, RBF())
    rom_truncated.fit()

    while True:
        print("\nCurrent parameter options:")
        for i, name in enumerate(parameter_names):
            print(f"{i}: {name}")
        
        param_input = input("Enter parameter index to modify (0-4) or 'q' to go back: ")
        if param_input.lower() in {'q', 'quit'}:
            break
            
        try:
            param_idx = int(param_input)
            if param_idx < 0 or param_idx >= len(parameter_names):
                print("Invalid parameter index.")
                continue
        except ValueError:
            print("Invalid input. Try again.")
            continue

        value_input = input(f"Enter new value for {parameter_names[param_idx]}: ")
        try:
            new_value = float(value_input)
        except ValueError:
            print("Invalid value. Try again.")
            continue

        # Use a test case
        test_index = min(10, len(records)-1)  # Ensure test_index is valid
        new_parameters = parameters[test_index, :].copy()
        new_parameters[param_idx] = new_value
        
        print(f"Modified parameters: {dict(zip(parameter_names, new_parameters))}")

        # Predict with both models
        try:
            full_snapshot = rom.predict(new_parameters).snapshots_matrix[0]
            pod_snapshot = rom_truncated.predict(new_parameters).snapshots_matrix[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            continue

        # Get a valid object index for plotting
        test_object_index_map = object_index_maps[test_index]
        available_object_indices = list(test_object_index_map.keys())
        plot_object_index = available_object_indices[0] if available_object_indices else 0
        
        print(f"Using object index {plot_object_index} for plotting")
        print(f"Available object indices: {available_object_indices}")

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        try:
            plot_field(
                ax1,
                full_snapshot,
                records[test_index].input,
                object_index=plot_object_index,
                object_index_map=test_object_index_map
            )
            ax1.set_title("Full-order Approximation")

            plot_field(
                ax2,
                pod_snapshot,
                records[test_index].input,
                object_index=plot_object_index,
                object_index_map=test_object_index_map
            )
            ax2.set_title(f"Reduced-order Approximation (Rank {N})")

            for ax in (ax1, ax2):
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect('equal')

            param_str = f"{parameter_names[param_idx]}={new_value:.3f}"
            plt.suptitle(f"POD Stress Model | {param_str} | Rank {N}")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")
            print("Check if plot_field function is working correctly")