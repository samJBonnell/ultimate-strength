import subprocess
import os

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

trial = 'riks'
imperfection_file = 'eigen'

# Define the imperfection block
imperfection_block = [
    f'*IMPERFECTION, FILE={imperfection_file}, STEP=1\n',
    '1, 0.00061\n'
]

# Define the field output block
field_output_block = [
    '*OUTPUT, FIELD\n',
    '*NODE OUTPUT, NSET=load_set\n',
    'U, CF\n'
]

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Write the initial input file
fem_model = FEMPipeline(
    model="models\\riks.py",
    input_path="data\\input.jsonl", 
    output_path="data\\output.jsonl",
    input_class=PanelInput,
    output_class=PanelOutput
)
fem_model.write(panel)
fem_model.run()

# Read the .inp file from the disk and modify it with the imperfection
with open(f'{trial}.inp', 'r') as f:
    lines = f.readlines()

insert_index = next(i for i, line in enumerate(lines) if line.strip().upper().startswith('*STEP'))

# Find the corresponding *END STEP
for j in range(insert_index + 1, len(lines)):
    if lines[j].strip().upper().startswith('*END STEP'):
        end_step_index = j
        break

else:
    raise ValueError("No '*END STEP' found after first '*STEP'")

# Insert imperfection block before *STEP
lines = lines[:insert_index] + imperfection_block + lines[insert_index:]

# Insert the output block just before *END STEP
lines[end_step_index:end_step_index] = field_output_block

with open(f'{trial}.inp', 'w') as f:
    f.writelines(lines)

# Run the analysis using the .inp file
functionCall = subprocess.Popen(f"abaqus job={trial} ask_delete=OFF", shell=True)
functionCall.communicate()