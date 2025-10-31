import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from uuid import uuid4

from abq_lib.model_wrapper import ModelWrapper
from abq_lib.abaqus_imports import ModelClass, ModelOutput

# Specify direction which model we are going to use
from abq_lib.abaqus_imports import Model_02

"""
Sample creation for CPSC540 & my research:

set_1:
    t_panel                 [0.001, 0.010]
    t_longitudinal_web      [0.001, 0.010]
    t_longitudinal_flange   [0.001, 0.010]

    h_longitudinal_web      [0.025, 0.500]
    w_longitudinal_flange   [0.050, 0.125]
"""

n_samples = 600
dim = 5

# Bounds for the trial variables
bounds = np.array([
    [0.001, 0.010], # t_panel
    [0.001, 0.010],  # t_longitudinal_web
    [0.001, 0.010],  # t_longitudinal_flange
    [0.025, 0.500],   # h_longitudinal_web
    [0.050, 0.125]    # w_longitudinal_flange
])

# Generate Latin Hypercube Samples
sampler = qmc.LatinHypercube(d=dim)
raw_samples = sampler.random(n=n_samples)
samples = qmc.scale(raw_samples, bounds[:, 0], bounds[:, 1])

# -------------------------------
# Static / baseline input values
# -------------------------------

# Specify direction which model we are going to use
from abq_lib.abaqus_imports import Model_02

base_input = Model_02(
    model_name='model_02',
    job_name='',
    job_type="eigen",

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
    mesh_plate = 0.025,
    mesh_longitudinal_web = 0.025,
    mesh_longitudinal_flange = 0.025,

    # Model Parameters
    numCpus=4,
    numGpus=0,

    centroid=-1
)

fem_model = ModelWrapper(
    model="abaqus_scripts/models/model_02_eigen.py",
    input_path= "data/cpsc540/set_1/input.jsonl",
    output_path="data/cpsc540/set_1/output.jsonl",
    input_class=ModelClass,
    output_class=ModelOutput
)

for i, sample in enumerate(tqdm(samples, desc="Evaluating panel samples")):
    variable_input = base_input.copy(
        job_name=str(uuid4()),
        t_panel=sample[0],
        t_longitudinal_web = sample[1],
        t_longitudinal_flange = sample[2],
        h_longitudinal_web = sample[3],
        w_longitudinal_flange = sample[4],
    )

    try:
        fem_model.write(variable_input)
        fem_model.run() 
    except Exception as e:
        print(f"[{i}] Evaluation failed: {e}")