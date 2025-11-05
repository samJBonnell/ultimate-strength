import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from uuid import uuid4

from abq_lib.model_wrapper import ModelWrapper
from abq_lib.abaqus_imports import ModelClass, ModelOutput

# Specify direction which model we are going to use
from abq_lib.abaqus_imports import Model_02, Model_03


job_name = str(uuid4())

# base_input = Model_02(
#     model_name='model_02',
#     job_name=job_name,
#     job_type="eigen",

#     # Global Geometry
#     num_longitudinal = 4,

#     width = 3.0,
#     length = 3.0,

#     # Thickness List
#     t_panel = 0.010,
#     t_longitudinal_web = 0.0078,
#     t_longitudinal_flange = 0.004,

#     # Local stiffener geometry
#     h_longitudinal_web = 0.125,
#     w_longitudinal_flange = 0.100,

#     # Applied Pressure
#     axial_force = 1e6,

#     # Mesh Settings
#     mesh_plate = 0.02,
#     mesh_longitudinal_web = 15e-03,
#     mesh_longitudinal_flange = 0.025,

#     # Model Parameters
#     numCpus=4,
#     numGpus=0,

#     centroid=-1
# )

base_input = Model_03(
    model_name='model_03',
    job_name=job_name,
    job_type="eigen",

    # Global Geometry
    num_longitudinal = 4,
    num_transverse= 2,

    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.0078,
    t_longitudinal_flange = 0.004,
    t_transverse_web= 0.003,
    t_transverse_flange= 0.002,

    # Local stiffener geometry
    h_longitudinal_web = 0.125,
    w_longitudinal_flange = 0.100,
    h_transverse_web= 0.225,
    w_transverse_flange= 0.100,

    # Applied Pressure
    axial_force = 1e6,

    # Mesh Settings
    mesh_plate = 0.02,
    mesh_longitudinal_web = 15e-03,
    mesh_longitudinal_flange = 0.025,
    mesh_transverse_web=0.02,
    mesh_transverse_flange= 0.01,

    # Model Parameters
    numCpus=4,
    numGpus=0,

    centroid=-1
)


print("Eigen analysis file: {}".format(job_name))


fem_model = ModelWrapper(
    model="abaqus_scripts/models/model_03_eigen.py",
    input_path= "data/model_03/eigen/input.jsonl",
    output_path="data/model_03/eigen/output.jsonl",
    input_class=ModelClass,
    output_class=ModelOutput
)
try:
    fem_model.write(base_input)
    fem_model.run() 
except Exception as e:
    print(f"Evaluation failed: {e}")