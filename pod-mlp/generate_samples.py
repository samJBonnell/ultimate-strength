import numpy as np
from scipy.stats import qmc
from uuid import uuid4
from utils.FiniteElementModel import FiniteElementModel
from dataclasses import replace
from models.IO_hydrostatic import PanelInput, PanelOutput
from tqdm import tqdm

n_samples = 1000
dim = 7

# Bounds for the trial variables
bounds = np.array([
    # Stiffener Numbers
    [2, 10],    # transverse stiffener number
    [2, 10],    # longitudinal stiffener number

    # Thicknesses
    [0.005, 0.050],  # t_panel
    [0.005, 0.050],  # t_transverse_web
    [0.005, 0.050],  # t_transverse_flange
    [0.005, 0.050],  # t_longitudinal_web
    [0.005, 0.050],  # t_longitudinal_flange
])

# Generate Latin Hypercube Samples
sampler = qmc.LatinHypercube(d=dim)
raw_samples = sampler.random(n=n_samples)
samples = qmc.scale(raw_samples, bounds[:, 0], bounds[:, 1])

# -------------------------------
# Static / baseline input values
# -------------------------------
base_input = PanelInput(
    id="",  # will be overwritten
    num_transverse=3,
    num_longitudinal=6,
    width=6.0,
    length=9.0,
    t_panel=0.025,
    t_transverse_web=0.025,
    t_transverse_flange=0.025,
    t_longitudinal_web=0.025,
    t_longitudinal_flange=0.025,
    h_transverse_web=0.45,
    h_longitudinal_web=0.40,
    w_transverse_flange=0.25,
    w_longitudinal_flange=0.175,
    pressure_magnitude=-10000,
    mesh_plate=0.05,
    mesh_transverse_web=0.05,
    mesh_transverse_flange=0.05,
    mesh_longitudinal_web=0.05,
    mesh_longitudinal_flange=0.05,
)

# Create handeler for FEM model
fem = FiniteElementModel("models\\hydrostatic.py", "data\\hydrostatic\\input.jsonl", "data\\hydrostatic\\output.jsonl", PanelInput, PanelOutput)

for i, sample in enumerate(tqdm(samples, desc="Evaluating panel samples")):
    new_input = replace(
        base_input,
        id=str(uuid4()),
        num_transverse=int(sample[0]),
        num_longitudinal=int(sample[1]),
        t_panel=sample[2],
        t_transverse_web=sample[3],
        t_transverse_flange=sample[4],
        t_longitudinal_web=sample[5],
        t_longitudinal_flange=sample[6],
    )

    try:
        result = fem.evaluate(new_input)
    except Exception as e:
        print(f"[{i}] Evaluation failed: {e}")