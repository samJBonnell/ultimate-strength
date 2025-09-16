from utils.FEMPipeline import FEMPipeline
from utils.IO_utils import PanelInput, PanelOutput

from datetime import datetime
now = datetime.now()
print(f"Start Time: {now}")

model_name = 'eigen'
# trial_id = re.sub(r'[:\s\-\.]', '_', str(now))
trial_id = '1'
job_name = model_name + "_" + trial_id

panel = PanelInput(
    model_name=model_name,
    job_name=job_name,

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
    mesh_longitudinal_web = 13.228E-03,
    mesh_longitudinal_flange = 0.025,

    # Model Parameters
    numCpus=4,
    numGpus=0
)

fem_model = FEMPipeline(
    model="models\\eigen.py",
    input_path="data\\input.jsonl", 
    output_path="data\\output.jsonl",
    input_class=PanelInput,
    output_class=PanelOutput
)

fem_model.write(panel)
fem_model.run()