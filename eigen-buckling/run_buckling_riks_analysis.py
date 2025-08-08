import subprocess
import os

from utils.FiniteElementModel import FiniteElementModel
from models.IO_buckling_eigen import PanelInput, PanelOutput

UC1 = PanelInput(
    id = "",

    # Global Geometry
    num_longitudinal = 1,

    width = 0.6,
    length = 0.6,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.078,
    t_longitudinal_flange = 0.004,

    # Local stiffener geometry
    h_longitudinal_web = 0.125,
    w_longitudinal_flange = 0.100,

    # Applied Pressure
    axial_force = 10000000,

    # Mesh Settings
    mesh_plate = 0.025,
    mesh_longitudinal_web = 0.025,
    mesh_longitudinal_flange = 0.025
)

panel = PanelInput(
    id = "",

    # Global Geometry
    num_longitudinal = 4,

    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.078,
    t_longitudinal_flange = 0.004,

    # Local stiffener geometry
    h_longitudinal_web = 0.125,
    w_longitudinal_flange = 0.100,

    # Applied Pressure
    axial_force = 10000000,

    # Mesh Settings
    mesh_plate = 0.025,
    mesh_longitudinal_web = 0.025,
    mesh_longitudinal_flange = 0.025
)

trial = 'buckling_riks_panel'
imperfection_file = 'buckling_eigen_panel'

# Write the initial input file
fem = FiniteElementModel(f"models\\{trial}.py", "data\\input.jsonl", "data\\output.jsonl", PanelInput, PanelOutput)
fem.write(panel)
fem.run()

# Read the .inp file from the disk and modify it with the imperfection
with open(f'{trial}.inp', 'r') as f:
    lines = f.readlines()

insert_index = next(i for i, line in enumerate(lines) if line.strip().upper().startswith('*STEP'))

# Define the imperfection block
imperfection_block = [
    f'*IMPERFECTION, FILE={imperfection_file}, STEP=1\n',
    '1, 0.006\n'
]

# Define the field output block
field_output_block = [
    '*OUTPUT, FIELD\n',
    '*NODE OUTPUT, NSET=Monitoring-Set\n',
    'U, RF\n'
]

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