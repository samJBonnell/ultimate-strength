import numpy as np
from typing import List, Dict, Tuple

from us_lib.data.parsing import slice_single_stress_vector

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

def filter_valid_snapshots(
    snapshots: List[np.ndarray],
    parameters: np.ndarray,
    expected_length: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    valid_indices = [i for i, s in enumerate(snapshots) if len(s) == expected_length]
    return parameters[valid_indices], [snapshots[i] for i in valid_indices]

def plot_field(axis, training_field, panel_input, object_index, object_index_map, levels = 20, vmin=None, vmax=None):
    """
    Plot a specific subfield from the full training field using geometry info from PanelInput.
    
    Parameters:
    -----------
    axis : matplotlib axis
        The axis to plot on
    training_field : array-like
        The full field data
    panel_input : PanelInput
        Panel geometry information
    object_index : int
        Index of the object to plot
    object_index_map : dict
        Mapping of object indices to field slices
    vmin : float, optional
        Minimum value for colormap scaling
    vmax : float, optional
        Maximum value for colormap scaling
    
    Returns:
    --------
    contour : matplotlib contour object
        The contour plot object (useful for adding colorbars)
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
    
    # Plot with optional vmin/vmax
    return axis.contourf(X, Y, z, cmap='RdBu_r', vmin=vmin, vmax=vmax, levels=levels)