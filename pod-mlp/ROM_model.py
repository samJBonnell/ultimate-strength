import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

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

# Load data records
records = load_random_records(input_path, output_path, n=500)
stress_vectors = extract_stress_vectors(records)
element_indices = [r.output.element_counts for r in records]

feature_names = [f for f in vars(records[0].input) if f != "id"]
all_vars = np.array([
    [getattr(r.input, f) for f in feature_names]
    for r in records
], dtype=float)

selected_keys = ["num_transverse", "num_longitudinal", "t_panel", "t_transverse_web", "t_transverse_flange", "t_longitudinal_web", "t_longitudinal_flange"]
parameters = np.array([
    [getattr(rec.input, k) for k in selected_keys]
    for rec in records
])

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
step_size = 50
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

# Plot the first N modes of the plate
n_modes = 10
modes_per_row = 5
object_index = 2  # Index of record you want to visualize

truncation_rank = -1  # Use last rank
order = -1            # Use last order

rows = math.ceil(n_modes / modes_per_row)
fig, axes = plt.subplots(rows, modes_per_row, figsize=(5 * modes_per_row, 4 * rows))
axes = axes.flatten()

for i in range(n_modes):
    plot_field(
        axes[i],
        pods[order][truncation_rank].modes[:, i],
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