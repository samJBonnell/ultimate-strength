from utils.FEMPipeline import FEMPipeline
from utils.IO_utils import ModelInput, ModelOutput
from datetime import datetime
import re

now = datetime.now()
print(f"Start Time: {now}")

model_name = 'flat_panel'
trial_id = re.sub(r'[:\s\-\.]', '_', str(now))
# trial_id = '1'
job_name = model_name + "_" + trial_id

panel = ModelInput(
    model_name=model_name,
    job_name=job_name,

    # Global Geometry
    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,

    # Force Information
    pressure = 1e2,
    pressure_location = [0.5, 0],
    pressure_patch_size = [0.2, 0.2],

    # Mesh Settings
    mesh_plate = 0.05,

    # Model Parameters
    numCpus=4,
    numGpus=0,
)

fem_model = FEMPipeline(
    model="models\\flat_panel.py",
    input_path="data\\input.jsonl", 
    output_path="data\\output.jsonl",
    input_class=ModelInput,
    output_class=ModelOutput)

fem_model.write(panel)
fem_model.run()