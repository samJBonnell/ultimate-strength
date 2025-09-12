from utils.FEMPipeline import FEMPipeline
from utils.IO_utils import PanelInput, PanelOutput

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
    mesh_longitudinal_web = 0.125 / 6,
    mesh_longitudinal_flange = 0.025
)

fem_model = FEMPipeline(
    model="models\\temp.py",
    input_path="data\\input.jsonl", 
    output_path="data\\output.jsonl",
    input_class=PanelInput,
    output_class=PanelOutput
)

fem_model.write(panel)
fem_model.run()