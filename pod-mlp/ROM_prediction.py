import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import fields

from ezyrb import POD, RBF
from ezyrb import ReducedOrderModel as ROM

from utils.JSON_utils import (
    load_random_records
)

from utils.ROM_utils import (
    extract_stress_vectors,
    filter_valid_snapshots,
    create_ROM,
    training_data_constructor,
    plot_field,
)

# Define input and output data locations
input_path = Path("data/hydrostatic/input.jsonl")
output_path = Path("data/hydrostatic/output.jsonl")
n_samples = 500

# Selected features for training
feature_names = [
    "num_transverse",
    "num_longitudinal",
    "t_panel",
    "t_transverse_web",
    "t_transverse_flange",
    "t_longitudinal_web",
    "t_longitudinal_flange"
]

# Load data into memory by randomly sampling the input and output files
records = load_random_records(input_path, output_path, n_samples)
print(f"Loaded {len(records)} records.")

# Exclude ID from variables and create a 2D Numpy array to store the information
all_vars = np.array([
    [getattr(r.input, f.name) for f in fields(r.input) if f.name != "id"]
    for r in records
], dtype=float)

# Build feature name → index map
all_feature_names = [f.name for f in fields(records[0].input) if f.name != "id"]
feature_index_map = {name: i for i, name in enumerate(all_feature_names)}
selected_indices = [feature_index_map[name] for name in feature_names]
parameters = all_vars[:, selected_indices]

# Convert stress fields to slices
stress_vectors = extract_stress_vectors(records)
element_indices = [r.output.element_counts for r in records]

# Determine the max element shape for padding/template
max_field_index, max_field_shape = max(
    enumerate(element_indices), key=lambda x: sum(x[1][:-1])
)
template_length = int(sum(max_field_shape[:-1]))
template_stress_field = np.zeros(template_length)

# Construct padded fields and store object index maps
training_data = []
object_index_maps = []
for i in range(len(stress_vectors)):
    field, index_map = training_data_constructor(
        stress_vectors[i],
        template_stress_field,
        max_field_shape[:-1],
        element_indices[i]
    )
    training_data.append(field)
    object_index_maps.append(index_map)

# Filter snapshots by expected shape
snapshots, parameters = filter_valid_snapshots(training_data, parameters, expected_length=template_length)

# Train ROM
snapshots = np.array(snapshots)
parameters = np.array(parameters)

db, rom, pod, rbf = create_ROM(parameters, snapshots)

# Define a loop to allow for the creation of variable truncation rank and variable stiffener panels
while True:
    rank_input = input("Enter the POD truncation rank (or 'q' to quit): ")
    if rank_input.lower() in {'q', 'quit'}:
        print("Exiting.")
        break
    try:
        N = int(rank_input)
    except ValueError:
        print("Invalid input. Try again.")
        continue

    # Truncated POD + ROM
    pod_truncated = POD('svd', rank=N)
    pod_truncated.fit(snapshots)
    rom_truncated = ROM(db, pod_truncated, RBF())
    rom_truncated.fit()

    while True:
        tran_input = input("Enter number of transverse stiffeners (or 'q'): ")
        if tran_input.lower() in {'q', 'quit'}:
            break
        long_input = input("Enter number of longitudinal stiffeners (or 'q'): ")
        if long_input.lower() in {'q', 'quit'}:
            break

        try:
            tran = int(tran_input)
            long = int(long_input)
        except ValueError:
            print("Invalid stiffener count. Try again.")
            continue

        test_index = 10  # Example test index
        new_parameters = parameters[test_index, :].copy()
        new_parameters[feature_names.index("num_transverse")] = tran
        new_parameters[feature_names.index("num_longitudinal")] = long

        full_snapshot = rom.predict(new_parameters).snapshots_matrix[0]
        pod_snapshot = rom_truncated.predict(new_parameters).snapshots_matrix[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))        
        plot_field(
            ax1,
            full_snapshot,
            records[test_index].input,
            object_index=2,
            object_index_map=object_index_maps[test_index]
        )
        ax1.set_title("Full-order Approximation")

        plot_field(
            ax2,
            pod_snapshot,
            records[test_index].input,
            object_index=2,
            object_index_map=object_index_maps[test_index]
        )
        ax2.set_title("Reduced-order Approximation")

        for ax in (ax1, ax2):
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_aspect('equal')

        plt.suptitle(f"POD Stress Model | T: {tran}, L: {long} | Rank {N}")
        plt.tight_layout()
        plt.show()