from utils.FiniteElementModel import FiniteElementModel
from models.IO_hydrostatic import PanelInput, PanelOutput

panel_input = PanelInput(
    id = "",

    # Global Geometry
    num_transverse = 3,
    num_longitudinal = 6,

    width = 6.0,
    length = 9.0,

    # Thickness List
    t_panel = 0.025,
    t_transverse_web = 0.025,
    t_transverse_flange = 0.025,
    t_longitudinal_web = 0.025,
    t_longitudinal_flange = 0.025,

    # Local stiffener geometry
    h_transverse_web = 0.45,
    h_longitudinal_web = 0.40,

    w_transverse_flange = 0.25,
    w_longitudinal_flange = 0.175,

    # Applied Pressure
    pressure_magnitude = -10000,

    # Mesh Settings
    mesh_plate = 0.05,
    mesh_transverse_web = 0.05,
    mesh_transverse_flange = 0.05,
    mesh_longitudinal_web = 0.05,
    mesh_longitudinal_flange = 0.05
)

fem = FiniteElementModel("models\\hydrostatic.py", "data\\hydrostatic\\input.jsonl", "data\\hydrostatic\\output.jsonl", PanelInput, PanelOutput)
result = fem.evaluate(panel_input)