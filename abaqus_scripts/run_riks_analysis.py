import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from os.path import join
from uuid import uuid4
from pathlib import Path
import subprocess
import os

from abq_lib.abaqus_writer import load_last_input
from abq_lib.model_wrapper import ModelWrapper
from abq_lib.abaqus_imports import ModelClass, ModelOutput

# Specify direction which model we are going to use
from abq_lib.abaqus_imports import Model_02

# -------------------------------------------------------------------------------------
#   Grab the imperfection file name from eigen analysis results
# -------------------------------------------------------------------------------------

input_path = Path('data/model_02/eigen/input.jsonl')
last_input = load_last_input(input_path)
imperfection_file = last_input.job_name

print("Using imperfection file from eigen analysis: {}".format(imperfection_file))
# -------------------------------------------------------------------------------------
#   Configure the Riks analysis
# -------------------------------------------------------------------------------------

job_name = str(uuid4())

base_input = Model_02(
    model_name='model_02',
    job_name=job_name,
    job_type="riks",

    # Global Geometry
    num_longitudinal = 4,

    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.0078,
    t_longitudinal_flange = 0.004,

    # Local stiffener geometry
    h_longitudinal_web = 0.125 + (1/2) * (0.010 + 0.004),
    w_longitudinal_flange = 0.100,

    # Applied Pressure
    axial_force = 1e6,

    # Mesh Settings
    mesh_plate = 0.02,
    mesh_longitudinal_web = 20.833E-03,
    mesh_longitudinal_flange = 0.025,

    # Model Parameters
    numCpus=4,
    numGpus=0,

    centroid=-1
)

print("Riks analysis file: {}".format(job_name))

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

# -------------------------------------------------------------------------------------
#   Create the riks step .inp file
# -------------------------------------------------------------------------------------

fem_model = ModelWrapper(
    model="abaqus_scripts/models/model_02_riks.py",
    input_path="data/model_02/riks/input.jsonl",
    output_path="data/model_02/riks/output.jsonl",
    input_class=ModelClass,
    output_class=ModelOutput
)

try:
    fem_model.write(base_input)
    fem_model.run()
except Exception as e:
    print("Failed to generate .inp file: {}".format(e))
    raise

# -------------------------------------------------------------------------------------
#   Insert the imperfection data into the .inp file created above
# -------------------------------------------------------------------------------------

# Path to working directory and .inp file
working_dir = Path('abaqus_scripts/working')
inp_file = working_dir / '{}.inp'.format(job_name)

if not inp_file.exists():
    raise FileNotFoundError("Input file not found: {}".format(inp_file))

# Read the .inp file
with open(str(inp_file), 'r') as f:
    lines = f.readlines()

# Find the first *STEP
insert_index = next(i for i, line in enumerate(lines) if line.strip().upper().startswith('*STEP'))
if insert_index is None:
    raise ValueError("No '*STEP' found in .inp file")

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

with open(str(inp_file), 'w') as f:
    f.writelines(lines)

original_dir = os.getcwd()
os.chdir(str(working_dir))

# -------------------------------------------------------------------------------------
#   Run the riks .inp file directly
# -------------------------------------------------------------------------------------

# Run the analysis using the .inp file
# functionCall = subprocess.Popen(
#     "abaqus job={} input={}.inp ask_delete=OFF cpus={}".format(
#         job_name, job_name, base_input.numCpus
#     ),
#     shell=True,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE
# )

functionCall = subprocess.Popen(
    f"abaqus job={job_name} input={job_name}.inp ask_delete=OFF cpus={base_input.numCpus} gpus={base_input.numGpus}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

stdout, stderr = functionCall.communicate()

# Change back to original directory
os.chdir(original_dir)

if functionCall.returncode == 0:
    pass
else:
    print("Analysis failed with return code: {}".format(functionCall.returncode))
    if stderr:
        print("Error output: {}".format(stderr))