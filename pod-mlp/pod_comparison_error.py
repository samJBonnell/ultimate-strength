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

import numpy as np
import matplotlib.pyplot as plt

def analyze_rmse_vs_order(records, training_data, parameters, parameter_names, snapshots, db):
    """
    Interactive analysis of RMSE vs POD order for selected records.
    
    Parameters:
        records (list): List of available records (can be indices or metadata).
        training_data (np.ndarray): Original snapshot data [n_records x ...].
        parameters (np.ndarray): Parameter matrix [n_records x n_parameters].
        parameter_names (list of str): Names of each parameter.
        snapshots (np.ndarray): Snapshot matrix used for POD basis.
        db: Database or structure required by ROM.
    """

    while True:
        print(f"\nAvailable records: 0 to {len(records) - 1}")
        record_input = input("Enter record index to analyze RMSE vs order (or 'q' to quit): ")

        if record_input.lower() in {'q', 'quit'}:
            print("Exiting RMSE vs POD order analysis.")
            return  # Exit the function

        try:
            test_index = int(record_input)
            if test_index < 0 or test_index >= len(records):
                print(f"Invalid record index. Must be between 0 and {len(records) - 1}.")
                continue
        except ValueError:
            print("Invalid input. Try again.")
            continue

        # Display selected record parameters
        print(f"\nSelected record {test_index} parameters:")
        for i, name in enumerate(parameter_names):
            print(f"  {name}: {parameters[test_index, i]:.4f}")

        original_snapshot = training_data[test_index]
        original_parameters = parameters[test_index, :].copy()

        # Compute max possible POD rank for this snapshot
        max_order = min(len(snapshots) - 1, 249)  # Cap to avoid long runtimes

        orders = range(1, max_order + 1, 2)  # Test every 2nd order

        rmse_values = []
        max_error_values = []

        print(f"\nTesting POD orders from 1 to {max_order}...")

        for order in orders:
            try:
                pod_order = POD('svd', rank=order)
                pod_order.fit(snapshots)

                rom_order = ROM(db, pod_order, RBF())
                rom_order.fit()

                pod_snapshot = rom_order.predict(original_parameters).snapshots_matrix[0]

                difference = original_snapshot - pod_snapshot
                rmse = np.sqrt(np.mean(difference ** 2))
                max_error = np.max(np.abs(difference))

                rmse_values.append(rmse)
                max_error_values.append(max_error)

                if order % 10 == 1:
                    print(f"  Order {order}: RMSE = {rmse:.2e}")

            except Exception as e:
                print(f"  Failed at order {order}: {e}")
                rmse_values.append(np.nan)
                max_error_values.append(np.nan)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        valid_orders = [o for o, r in zip(orders, rmse_values) if not np.isnan(r)]
        valid_rmse = [r for r in rmse_values if not np.isnan(r)]
        valid_max_error = [e for e in max_error_values if not np.isnan(e)]

        ax1.semilogy(valid_orders, valid_rmse, 'b-o', markersize=4)
        ax1.set_xlabel('POD Order')
        ax1.set_ylabel('RMSE')
        ax1.set_title(f'RMSE vs POD Order (Record {test_index})')
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(valid_orders, valid_max_error, 'r-s', markersize=4)
        ax2.set_xlabel('POD Order')
        ax2.set_ylabel('Maximum Absolute Error')
        ax2.set_title(f'Max Error vs POD Order (Record {test_index})')
        ax2.grid(True, alpha=0.3)

        param_str = ", ".join([f"{name}={parameters[test_index, i]:.3f}" 
                               for i, name in enumerate(parameter_names)])
        plt.suptitle(f"Error Analysis for Record {test_index}\n{param_str}")
        plt.tight_layout()
        plt.show()

        # Summary
        if valid_rmse:
            print(f"\nError Analysis Summary:")
            print(f"  Minimum RMSE: {min(valid_rmse):.2e} at order {valid_orders[np.argmin(valid_rmse)]}")
            print(f"  RMSE at order 10: {valid_rmse[4] if len(valid_rmse) > 4 else 'N/A':.2e}")
            print(f"  RMSE reduction from order 1 to {max(valid_orders)}: {valid_rmse[0]/valid_rmse[-1]:.1f}x")

            if len(valid_rmse) > 5:
                rmse_ratios = [valid_rmse[i] / valid_rmse[i + 1] for i in range(len(valid_rmse) - 1)]
                elbow_idx = np.argmax(np.array(rmse_ratios) < 1.1)
                if elbow_idx > 0:
                    elbow_order = valid_orders[elbow_idx]
                    print(f"  Suggested order (elbow point): {elbow_order} (RMSE = {valid_rmse[elbow_idx]:.2e})")


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

analyze_rmse_vs_order(records, training_data, parameters, parameter_names, snapshots, db)