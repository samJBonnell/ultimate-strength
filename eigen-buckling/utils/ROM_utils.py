import numpy as np
from typing import List, Dict, Tuple
from ezyrb import POD, RBF, Database, ReducedOrderModel as ROM

from utils.JSON_utils import Record

# --- 1. Extract stress vectors from Record list ---
def extract_stress_vectors(records: List[Record]) -> List[List[float]]:
    return [[s.stress for s in r.output.stress_field] for r in records]

# --- 2. Extract selected features from PanelInput ---
def extract_input_features(records: List[Record], feature_names: List[str]) -> np.ndarray:
    return np.array([
        [getattr(r.input, f) for f in feature_names]
        for r in records
    ], dtype=float)

# --- 3. Extract element index lists from output ---
def extract_element_indices(records: List[Record]) -> List[List[int]]:
    return [r.output.element_counts for r in records]

# --- 4. Pad one stress vector into a template ---
def slice_single_stress_vector(stress_vector: List[float], element_indices: List[int], target_part: int) -> List[float]:
    start = int(sum(element_indices[:target_part]))
    end = start + int(element_indices[target_part])
    return stress_vector[start:end]

def slice_stress_vectors(stress_vectors, element_indices_list, target_part=None):
    return [
        slice_single_stress_vector(vec, indices, target_part)
        for vec, indices in zip(stress_vectors, element_indices_list)
    ]

def training_data_constructor(
    stress_vector: List[float],
    template_stress_field: np.ndarray,
    template_element_indices: List[int],
    element_indices: List[int]
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    training_field = template_stress_field.copy()
    object_start_indices = {}

    index = 0
    for i in range(len(template_element_indices)):
        sub_field = slice_single_stress_vector(stress_vector, element_indices, i)
        training_field[index: index + len(sub_field)] = sub_field
        object_start_indices[i] = (index, index + len(sub_field))
        index += int(template_element_indices[i])

    return training_field, object_start_indices

# --- 5. Wrap full training field generation ---
def build_training_fields(
    stress_vectors: List[List[float]],
    element_indices: List[List[int]],
    expected_length: int
) -> Tuple[List[np.ndarray], List[Dict[int, Tuple[int, int]]]]:
    # Find max-sized field for padding template
    max_index, max_shape = max(enumerate(element_indices), key=lambda x: sum(x[1]))
    template = np.zeros(int(sum(max_shape)))
    padded_fields = []
    object_maps = []

    for vec, shape in zip(stress_vectors, element_indices):
        field, index_map = training_data_constructor(vec, template, max_shape, shape)
        padded_fields.append(field)
        object_maps.append(index_map)

    return padded_fields, object_maps

# --- 6. Filter valid snapshots based on length ---
def filter_valid_snapshots(
    snapshots: List[np.ndarray],
    parameters: np.ndarray,
    expected_length: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    valid_indices = [i for i, s in enumerate(snapshots) if len(s) == expected_length]
    return [snapshots[i] for i in valid_indices], parameters[valid_indices]

# --- 7. ROM Creation ---
def create_ROM(parameters: np.ndarray, snapshots: List[np.ndarray]):
    snapshots = np.array(snapshots)
    parameters = np.array(parameters)

    db = Database(parameters, snapshots)
    pod = POD('svd')
    rbf = RBF()
    rom = ROM(db, pod, rbf)
    rom.fit()

    return db, rom, pod, rbf

# --- 8. Plotting Functions ---
def plot_field(axis, training_field, panel_input, object_index, object_index_map):
    """
    Plot a specific subfield from the full training field using geometry info from PanelInput.
    """
    # Extract geometry
    width = float(panel_input.width)
    length = float(panel_input.length)
    mesh_size = float(panel_input.mesh_plate)

    # Slice the field
    start, end = object_index_map[object_index]
    sub_field = np.array(training_field[start:end])

    print(f"Plotting sub_field[{start}:{end}] of size {len(sub_field)}")

    # Compute grid shape
    base_w_el = int(width / mesh_size)
    base_l_el = int(length / mesh_size)
    total_elements = len(sub_field)

    if total_elements % base_w_el == 0:
        num_w_el = base_w_el
        num_l_el = total_elements // base_w_el
    elif total_elements % base_l_el == 0:
        num_l_el = base_l_el
        num_w_el = total_elements // base_l_el
    else:
        raise ValueError(
            f"Cannot reshape sub_field of length {total_elements} into 2D grid. "
            f"Check stiffener count or mesh size."
        )

    # Create meshgrid
    x = np.linspace(0, width, num_w_el)
    y = np.linspace(0, length, num_l_el)
    X, Y = np.meshgrid(x, y)
    z = sub_field.reshape(num_l_el, num_w_el)

    return axis.contourf(X, Y, z)