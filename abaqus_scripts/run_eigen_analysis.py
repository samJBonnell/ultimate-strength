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


# Geometry because the Paik paper define the thicknesses as manufactured dimensions and we have to factor in the thickness of the panels
# Plate
t = 0.010

# Longitudinals
h_w_l = 0.290
b_f_l = 0.090
t_w_l = 0.010
t_f_l = 0.010

h_longitudinal_web = h_w_l + (1/2)*(t_f_l + t)
w_longitudinal_flange = b_f_l

# Transverse
h_w_t = 0.665
b_f_t = 0.150
t_w_t = 0.010
t_f_t = 0.010

h_transverse_web = h_w_t + (1/2)*(t_f_t + t)
w_transverse_flange = b_f_t

Paik_Test = Model_03(
    model_name='model_03',
    job_name=job_name,
    job_type="eigen",

    # Global Geometry
    num_transverse= 2,

    location_longitudinals= [0.24, 0.720, 0.720, 0.720],

    width = 2.640,
    length = 9.450,

    # Thickness List
    t_panel = t,
    t_longitudinal_web = t_w_l,
    t_longitudinal_flange = t_f_l,
    t_transverse_web= t_w_t,
    t_transverse_flange= t_f_t,

    # Local stiffener geometry
    h_longitudinal_web = h_longitudinal_web,
    w_longitudinal_flange = w_longitudinal_flange,
    h_transverse_web= h_transverse_web,
    w_transverse_flange= w_transverse_flange,

    # Applied Pressure
    axial_force = 1e6,

    # Mesh Settings
    mesh_plate = 0.025,
    mesh_longitudinal_web = 0.025,
    mesh_longitudinal_flange = 0.025,
    mesh_transverse_web=0.025,
    mesh_transverse_flange= 0.025,

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
    fem_model.write(Paik_Test)
    fem_model.run() 
except Exception as e:
    print(f"Evaluation failed: {e}")