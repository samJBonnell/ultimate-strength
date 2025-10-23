import numpy as np
from typing import List, Dict, Tuple
from ezyrb import POD, RBF, Database, ReducedOrderModel as ROM

from utils.json_utils import Record

def extract_von_mises_stress(records: List[Record], step = 'Step-1') -> List[List[float]]:
    return [[s.attributes['stress'].vm for s in r.output.elements_by_step[step]] for r in records]

def extract_input_features(records: List[Record], feature_names: List[str]) -> np.ndarray:
    return np.array([
        [getattr(r.input, f) for f in feature_names]
        for r in records
    ], dtype=float)

def extract_element_indices(records: List[Record]) -> List[List[int]]:
    return [r.output.element_counts for r in records]

def slice_single_stress_vector(stress_vector: List[float], element_indices: List[int], target_part: int) -> List[float]:
    start = int(sum(element_indices[:target_part]))
    end = start + int(element_indices[target_part])
    return stress_vector[start:end]

def slice_stress_vectors(stress_vectors, element_indices_list, target_part=None):
    return [
        slice_single_stress_vector(vec, indices, target_part)
        for vec, indices in zip(stress_vectors, element_indices_list)
    ]