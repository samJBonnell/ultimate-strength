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

# Update column names to match
parameter_names = ["t_panel", "pressure_x", "pressure_y", "patch_width", "patch_height"]

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

model_order = 100

print(f"\nPlottong POD Modes for Model Order: {model_order}")
sample_points = np.random.choice(a=model_order, size=model_order, replace=False)
training_parameters = parameters[sample_points, :]
training_snapshots = [snapshots[i] for i in sample_points]

try:
    db, rom, pod, rbf = create_ROM(parameters, snapshots)
except Exception as e:
    print(f"ROM creation failed at order {model_order}: {e}")
    
# Plot the first N modes of the plate
n_modes = 10
modes_per_row = 5

object_index = 0  # Index of part you want to visualize

truncation_rank = -1  # Use last rank
order = -1            # Use last order

rows = math.ceil(n_modes / modes_per_row)
fig, axes = plt.subplots(rows, modes_per_row, figsize=(5 * modes_per_row, 4 * rows))
axes = axes.flatten()
for i in range(n_modes):
    plot_field(
        axes[i],
        pod.modes[:, i],
        records[object_index].input,
        object_index=object_index,
        object_index_map=object_index_maps[object_index]
    )
    axes[i].set_title(f"Mode {i+1}")
    axes[i].set_xlabel("x (m)")
    axes[i].set_ylabel("y (m)")
    axes[i].set_aspect("equal")

for j in range(n_modes, len(axes)):
    axes[j].axis("off")

plt.suptitle(f"POD Plate First {n_modes} Stress Modes", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()