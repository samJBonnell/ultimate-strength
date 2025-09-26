import numpy as np
from scipy.stats import qmc
from uuid import uuid4
from utils.FEMPipeline import FEMPipeline
from dataclasses import replace
from utils.IO_utils import ModelInput, ModelOutput
from tqdm import tqdm

n_samples = 250
dim = 5

# Bounds for the trial variables
bounds = np.array([
    # Plate thickness
    [2, 10],    # plate thickness

    # Pressure Location
    [-2.65, 2.65],  # x-axis pressure location
    [-2.65, 2.65],  # y-axis pressure location

    # Pressure patch size
    [0.075, 0.225],
    [0.075, 0.225]
])

# Generate Latin Hypercube Samples
sampler = qmc.LatinHypercube(d=dim)
raw_samples = sampler.random(n=n_samples)
samples = qmc.scale(raw_samples, bounds[:, 0], bounds[:, 1])

# -------------------------------
# Static / baseline input values
# -------------------------------
base_input = ModelInput(
    model_name='panel',
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
    mesh_plate = 0.05,

    # Model Parameters
    numCpus=4,
    numGpus=1,
)

# Create handeler for FEM model
fem_model = FEMPipeline(
    model="models\\flat_panel.py",
    input_path="data\\input.jsonl", 
    output_path="data\\output.jsonl",
    input_class=ModelInput,
    output_class=ModelOutput)

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
        print(f"[{i}] Evaluation failed: {e}")

# # ---------------------------------------------------------------------------------------------
# # Full Panel Model
# # ---------------------------------------------------------------------------------------------

# n_samples = 1000
# dim = 7

# # Bounds for the trial variables
# bounds = np.array([
#     # Stiffener Numbers
#     [2, 10],    # transverse stiffener number
#     [2, 10],    # longitudinal stiffener number

#     # Thicknesses
#     [0.005, 0.050],  # t_panel
#     [0.005, 0.050],  # t_transverse_web
#     [0.005, 0.050],  # t_transverse_flange
#     [0.005, 0.050],  # t_longitudinal_web
#     [0.005, 0.050],  # t_longitudinal_flange
# ])

# # Generate Latin Hypercube Samples
# sampler = qmc.LatinHypercube(d=dim)
# raw_samples = sampler.random(n=n_samples)
# samples = qmc.scale(raw_samples, bounds[:, 0], bounds[:, 1])

# # -------------------------------
# # Static / baseline input values
# # -------------------------------
# base_input = ModelInput(
#     job_name="",  # will be overwritten
#     num_transverse=3,
#     num_longitudinal=6,
#     width=6.0,
#     length=9.0,
#     t_panel=0.025,
#     t_transverse_web=0.025,
#     t_transverse_flange=0.025,
#     t_longitudinal_web=0.025,
#     t_longitudinal_flange=0.025,
#     h_transverse_web=0.45,
#     h_longitudinal_web=0.40,
#     w_transverse_flange=0.25,
#     w_longitudinal_flange=0.175,
#     pressure_magnitude=-10000,
#     mesh_plate=0.05,
#     mesh_transverse_web=0.05,
#     mesh_transverse_flange=0.05,
#     mesh_longitudinal_web=0.05,
#     mesh_longitudinal_flange=0.05,
# )

# # Create handeler for FEM model
# fem_model = FEMPipeline(
#     model="models\\dual_stiffened_panel.py",
#     input_path="data\\input.jsonl", 
#     output_path="data\\output.jsonl",
#     input_class=ModelInput,
#     output_class=ModelOutput)

# for i, sample in enumerate(tqdm(samples, desc="Evaluating panel samples")):
#     variable_input = base_input.copy(
#         base_input,
#         id=str(uuid4()),
#         num_transverse=int(sample[0]),
#         num_longitudinal=int(sample[1]),
#         t_panel=sample[2],
#         t_transverse_web=sample[3],
#         t_transverse_flange=sample[4],
#         t_longitudinal_web=sample[5],
#         t_longitudinal_flange=sample[6],
#     )

#     try:
#         fem_model.write(variable_input)
#         fem_model.run() 
#     except Exception as e:
#         print(f"[{i}] Evaluation failed: {e}")