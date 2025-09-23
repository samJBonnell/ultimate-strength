from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PanelInput:
    id: str

    # Global Geometry
    num_transverse: int
    num_longitudinal: int

    width: float
    length: float

    # Thickness List
    t_panel: float
    t_transverse_web: float
    t_transverse_flange: float
    t_longitudinal_web: float
    t_longitudinal_flange: float

    # Local stiffener geometry
    h_transverse_web: float
    h_longitudinal_web: float

    w_transverse_flange: float
    w_longitudinal_flange: float

    # Applied Pressure
    pressure_magnitude: float

    # Mesh Settings
    mesh_plate: float
    mesh_transverse_web: float
    mesh_transverse_flange: float
    mesh_longitudinal_web: float
    mesh_longitudinal_flange: float

@dataclass
class ElementStress:
    element_id: int
    stress: float

@dataclass
class PanelOutput:
    id: str
    max_stress: float
    assembly_mass: float
    element_counts: Dict[str, int]
    stress_field: List[ElementStress]
    job_name: str
    step: str

    @staticmethod
    def from_dict(d: dict) -> 'PanelOutput':
        stress_objs = [ElementStress(**s) for s in d.get("stress_field", [])]
        return PanelOutput(
            id=d["id"],
            max_stress=d["max_stress"],
            assembly_mass=d["assembly_mass"],
            element_counts=d["element_counts"],
            stress_field=stress_objs,
            job_name=d["job_name"],
            step=d["step"],
        )