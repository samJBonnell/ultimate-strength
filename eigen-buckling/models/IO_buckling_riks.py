from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PanelInput:
    id: str

    # Global Geometry
    num_longitudinal: int

    width: float
    length: float

    # Thickness List
    t_panel: float
    t_longitudinal_web: float
    t_longitudinal_flange: float

    # Local stiffener geometry
    h_longitudinal_web: float

    w_longitudinal_flange: float

    # Applied Pressure
    axial_force: float

    # Mesh Settings
    mesh_plate: float
    mesh_longitudinal_web: float
    mesh_longitudinal_flange: float

@dataclass
class ElementStress:
    element_id: int
    stress: float

@dataclass
class ElementDisplacement:
    element_id: int
    x: float
    y: float
    z: float

@dataclass
class PanelOutput:
    id: str
    element_counts: Dict[str, int]
    stress_field: Dict[str, List[ElementStress]]  # step -> list of stress results
    displacement_field: Dict[str, List[ElementDisplacement]]  # step -> list of disp results
    job_name: str
    steps: List[str]  # List of exported step names

    @staticmethod
    def from_dict(d: dict) -> 'PanelOutput':
        stress_field = {
            step: [ElementStress(**s) for s in stresses]
            for step, stresses in d.get("stress_field", {}).items()
        }
        displacement_field = {
            step: [ElementDisplacement(**s) for s in disps]
            for step, disps in d.get("displacement_field", {}).items()
        }
        return PanelOutput(
            id=d["id"],
            element_counts=d["element_counts"],
            stress_field=stress_field,
            displacement_field=displacement_field,
            job_name=d["job_name"],
            steps=d["steps"],
        )