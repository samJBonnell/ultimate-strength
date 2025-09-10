from utils.FiniteElementModel import FiniteElementModel
from models.IO_buckling import PanelInput, PanelOutput

from datetime import datetime
now = datetime.now()
print(f"Start Time: {now}")

panel = PanelInput(
    id = "",

    # Global Geometry
    num_longitudinal = 4,

    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.0078,
    t_longitudinal_flange = 0.004,

    # Local stiffener geometry
    h_longitudinal_web = 0.125,
    w_longitudinal_flange = 0.100,

    # Applied Pressure
    axial_force = 1e6,

    # Mesh Settings
    mesh_plate = 0.02,
    mesh_longitudinal_web = 0.125 / 5,
    mesh_longitudinal_flange = 0.025
)

fem = FiniteElementModel("models\\buckling_eigen_panel_merge.py", "data\\input.jsonl", "data\\output.jsonl", PanelInput, PanelOutput)
fem.write(panel)
fem.run()