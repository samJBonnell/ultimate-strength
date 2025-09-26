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
        print(f"\nAvailable records: 0 to {len(records)-1}")
        record_input = input("Enter record index to compare against (or 'q' to go back): ")
        if record_input.lower() in {'q', 'quit'}:
            break
            
        try:
            test_index = int(record_input)
            if test_index < 0 or test_index >= len(records):
                print(f"Invalid record index. Must be between 0 and {len(records)-1}.")
                continue
        except ValueError:
            print("Invalid input. Try again.")
            continue

        # Show the parameters for the selected record
        print(f"\nSelected record {test_index} parameters:")
        for i, name in enumerate(parameter_names):
            print(f"  {name}: {parameters[test_index, i]:.4f}")
        
        # Use the original parameters from the selected record
        original_parameters = parameters[test_index, :].copy()
        
        # Get original data for comparison
        original_snapshot = training_data[test_index]
        
        # Predict with reduced-order model using the same parameters
        try:
            pod_snapshot = rom_truncated.predict(original_parameters).snapshots_matrix[0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            continue

        # Get a valid object index for plotting
        test_object_index_map = object_index_maps[test_index]
        available_object_indices = list(test_object_index_map.keys())
        plot_object_index = available_object_indices[0] if available_object_indices else 0
        
        print(f"Using object index {plot_object_index} for plotting")

        # Plot comparison: Original vs POD prediction
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        try:
            # Original data
            plot_field(
                ax1,
                original_snapshot,
                records[test_index].input,
                object_index=plot_object_index,
                object_index_map=test_object_index_map
            )
            ax1.set_title("Original FEM Data")

            # POD prediction
            plot_field(
                ax2,
                pod_snapshot,
                records[test_index].input,
                object_index=plot_object_index,
                object_index_map=test_object_index_map
            )
            ax2.set_title(f"POD Prediction (Rank {N})")
            
            # Difference plot
            difference = original_snapshot - pod_snapshot
            plot_field(
                ax3,
                difference,
                records[test_index].input,
                object_index=plot_object_index,
                object_index_map=test_object_index_map
            )
            ax3.set_title("Difference (Original - POD)")

            for ax in (ax1, ax2, ax3):
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                ax.set_aspect('equal')

            # Calculate error metrics
            rmse = np.sqrt(np.mean(difference**2))
            max_error = np.max(np.abs(difference))
            
            param_str = ", ".join([f"{name}={parameters[test_index, i]:.3f}" 
                                 for i, name in enumerate(parameter_names)])
            plt.suptitle(f"Record {test_index} | Rank {N} | RMSE: {rmse:.2e} | Max Error: {max_error:.2e}")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")
            
        except Exception as e:
            print(f"Plotting failed: {e}")
            print("Check if plot_field function is working correctly")