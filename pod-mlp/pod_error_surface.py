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
    training_data_constructor
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

# Fit ROMs
step_size = 24
num_samples = parameters.shape[0]
orders = range(11, 101, step_size)
max_rank = 100
K = 5

models = []
pods = []

for order in orders:
    print(f"\nModel Order: {order}")
    sample_points = np.random.choice(num_samples, order, replace=False)
    temp_parameters = parameters[sample_points, :]
    temp_snapshots = [snapshots[i] for i in sample_points]

    try:
        db, rom, pod, rbf = create_ROM(parameters, snapshots)
    except Exception as e:
        print(f"ROM creation failed at order {order}: {e}")
        models.append([np.nan] * (max_rank // step_size))
        continue

    test_candidates = [i for i in range(num_samples) if i not in sample_points]
    if len(test_candidates) < K:
        print("Not enough test candidates. Skipping.")
        models.append([np.nan] * (max_rank // step_size))
        continue

    test_models = np.random.choice(test_candidates, K, replace=False)
    truncation_ranks = range(1, max_rank + 1, step_size)
    errors = []
    temp_pod = []

    for N in truncation_ranks:
        print(f"  Truncation Rank: {N}")
        try:
            pod_truncated = POD('svd', rank=N)
            pod_truncated.fit(temp_snapshots)
            rom_truncated = ROM(db, pod_truncated, RBF())
            rom_truncated.fit()
            temp_pod.append(pod_truncated)

            total_error = 0
            for model in test_models:
                new_parameters = parameters[model, :]
                pred_snapshot = rom_truncated.predict(new_parameters).snapshots_matrix[0]
                true_snapshot = snapshots[model]
                total_error += np.sqrt(np.mean((pred_snapshot - true_snapshot) ** 2))

            average_error = total_error / K
            errors.append(average_error)
        except Exception as e:
            print(f"    Error at N={N}: {e}")
            errors.append(np.nan)

    models.append(errors)
    pods.append(temp_pod)

# Plot ROM Error Surface
models_array = np.array(models)
orders = np.array(list(orders))
truncation_ranks = np.arange(1, max_rank + 1, step_size)
X, Y = np.meshgrid(truncation_ranks, orders)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, models_array, cmap='viridis')

ax.set_xlabel("Truncation Rank")
ax.set_ylabel("Training Set Size (Order)")
ax.set_zlabel("Average L2 Error")
ax.set_title("ROM Error Surface")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.show()