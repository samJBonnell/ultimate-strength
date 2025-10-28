import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from uuid import uuid4

from us_lib.new_io import FlatPanelInput
from abq_lib.model_wrapper import ModelWrapper
from abq_lib.abaqus_imports import ModelInput, ModelOutput

n_samples = 250
dim = 5

# Bounds for the trial variables
bounds = np.array([
    # Plate thickness
    [0.005, 0.075],    # plate thickness``

    # Pressure Location
    [-1.34, 1.34],  # x-axis pressure location
    [-1.34, 1.34],  # y-axis pressure location

    # Pressure patch size
    [0.10, 0.25],
    [0.10, 0.25]
])

# Generate Latin Hypercube Samples
sampler = qmc.LatinHypercube(d=dim)
raw_samples = sampler.random(n=n_samples)
samples = qmc.scale(raw_samples, bounds[:, 0], bounds[:, 1])

# -------------------------------
# Static / baseline input values
# -------------------------------

base_input = FlatPanelInput(
    model_name='flat_panel',
    job_name="",

    # Global Geometry
    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,

    # Force Information
    pressure = 1e2,
    pressure_location = [0, 0],
    pressure_patch_size = [0.1, 0.1],

    # Mesh Settings
    mesh_plate = 0.0277,

    # Model Parameters
    numCpus=4,
    numGpus=1,
)

# Create handeler for FEM model
fem_model = ModelWrapper(
    model="abaqus_scripts/models/flat_panel.py",
    input_path="data/input.jsonl",
    output_path="data/output.jsonl",
    input_class=ModelInput,
    output_class=ModelOutput
)

for i, sample in enumerate(tqdm(samples, desc="Evaluating panel samples")):
    variable_input = base_input.copy(
        job_name=str(uuid4()),
        t_panel=sample[0],
        pressure_location=[sample[1], sample[2]],
        pressure_patch_size=[sample[3], sample[4]]
    )

    try:
        fem_model.write(variable_input)
        fem_model.run() 
    except Exception as e:
        # print(f"[{i}] Evaluation failed: {e}")
        print("[{}] Evaluation failed: {}".format(i, e))