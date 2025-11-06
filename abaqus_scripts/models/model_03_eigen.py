# Sam Bonnell - UBC Labratory for Structural Efficiency MASc Student
# 2025-11-04

# ----------------------------------------------------------------------------------------------------------------------------------
# Library Import
import numpy as np
import os
import math

# ABAQUS Prefactory Information
from abaqus import *
from abaqusConstants import *
from sympy import capture
backwardCompatibility.setValues(includeDeprecated=True, reportDeprecated=False)

# Import module information from ABAQUS
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import regionToolset
import odbAccess

# ----------------------------------------------------------------------------------------------------------------------------------

import sys
from os.path import join, exists
import os

# Get project root from environment variable (set by ModelWrapper) or fallback to hardcoded
project_root = os.environ.get('PROJECT_ROOT')

if project_root is None:
    # Fallback for manual CAE runs - hardcode it
    project_root = 'C:/Users/sbonnell/Desktop/lase/projects/ultimate_strength'

# Add abaqus_scripts to path
abaqus_dir = join(project_root, 'abaqus_scripts')
if abaqus_dir not in sys.path:
    sys.path.insert(0, abaqus_dir)

# Define paths
working_directory = join(project_root, 'abaqus_scripts', 'working')
input_directory = join(project_root, 'data', 'model_03', 'eigen', 'input.jsonl')
output_directory = join(project_root,'data', 'model_03', 'eigen', 'output.jsonl')

# Create working directory if it doesn't exist
if not exists(working_directory):
    os.makedirs(working_directory)

# Now import from abq_lib
from abq_lib.abaqus_imports import ModelOutput, Element, Stress
from abq_lib.abaqus_writer import write_trial_ndjson, load_last_input

from abq_lib.node_utilities import *
from abq_lib.mesh_utilities import *
from abq_lib.section_utilities import *
from abq_lib.transformation_utilities import homogenous_transform
from abq_lib.constraint_utilities import *

# Configure coordinate output
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

project_root = os.path.abspath(project_root)
working_directory = os.path.abspath(join(project_root, working_directory))

# ----------------------------------------------------------------------------------------------------------------------------------
# Create the panel object
panel = load_last_input(input_directory)

# Change to working directory
os.chdir(working_directory)

model_name = panel.model_name
job_name = panel.job_name

# Creation of a list used to create section assignments for each component of the panel
component_thickness_map = {
    'panel': panel.t_panel,
    'l_web': panel.t_longitudinal_web,
    'l_flange': panel.t_longitudinal_flange,
    't_web' : panel.t_transverse_web,
    't_flange' : panel.t_transverse_flange
}

for component, thickness in component_thickness_map.items():
    if thickness is None:
        raise ValueError("{} thickness not set!".format(component))

# ----------------------------------------------------------------------------------------------------------------------------------
# Start of Definition of Panel Model

# Create model object
model = mdb.Model(name=model_name)

# ----------------------------------------------------------------------------------------------------------------------------------
# Part Definitions
# Dimensional definitions
# l_web_spacing = float(panel.width) / (panel.num_longitudinal + 1)
t_web_spacing = float(panel.length) / (panel.num_transverse + 1)
half_width = float(panel.width) / 2
half_length = float(panel.length) / 2

l_web_locations = np.cumsum(panel.location_longitudinals) - half_width
t_web_locations = np.arange(-half_length + t_web_spacing, half_length, t_web_spacing)

half_l_flange_width = float(panel.w_longitudinal_flange) / 2
half_t_flange_width = float(panel.w_transverse_flange) / 2

# base plate line
sketch = model.ConstrainedSketch(name='geometry', sheetSize=4.0)
sketch.Line(point1=(-half_width, 0.0), point2=(half_width, 0.0))

for web_location in l_web_locations:
    # vertical web
    web_line = sketch.Line(point1=(web_location, 0.0), point2=(web_location, panel.h_longitudinal_web))
    sketch.VerticalConstraint(entity=web_line)

    # flange
    flange_line = sketch.Line(point1=(web_location - half_l_flange_width, panel.h_longitudinal_web), point2=(web_location + half_l_flange_width, panel.h_longitudinal_web))
    sketch.HorizontalConstraint(entity=flange_line)

p_l = model.Part(name='longitudinal', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p_l.BaseShellExtrude(sketch=sketch, depth=panel.length)
del model.sketches['geometry']

# Transverse Stiffeners
sketch = model.ConstrainedSketch(name='geometry', sheetSize=4.0)

for web_location in t_web_locations:
    # Web
    web_line = sketch.Line(point1=(web_location, 0.0), point2=(web_location, panel.h_transverse_web))
    sketch.VerticalConstraint(entity=web_line)

    # flange
    flange_line = sketch.Line(point1=(web_location - half_t_flange_width, panel.h_transverse_web), point2=(web_location + half_t_flange_width, panel.h_transverse_web))
    sketch.HorizontalConstraint(entity=flange_line)

p_t = model.Part(name = 'transverse', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p_t.BaseShellExtrude(sketch=sketch, depth=panel.width)
del model.sketches['geometry']

# --- Face sets ---

capture_offset = 1e-5

# Capture the plate faces
plate_faces = p_l.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax=half_width + capture_offset,
    yMin=-capture_offset, yMax=capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p_l.Set(faces=plate_faces, name='plate_face')

# Capture the longitudinal flange faces
flange_faces = p_l.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax = half_width + capture_offset,
    yMin=panel.h_longitudinal_web - capture_offset, yMax=panel.h_longitudinal_web + capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p_l.Set(faces=flange_faces, name='l_flange_faces')

# Capture the transverse flange faces
flange_faces = p_t.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax = half_width + capture_offset,
    yMin=panel.h_transverse_web - capture_offset, yMax=panel.h_transverse_web + capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p_t.Set(faces=flange_faces, name='t_flange_faces')

# Capture each of the longitudinal web faces
all_faces = p_l.faces.getByBoundingBox(
    xMin=1e6, xMax=-1e6,
    yMin=1e6, yMax=-1e6,
    zMin=1e6, zMax=-1e6
)

for web_location in l_web_locations:
    faces = p_l.faces.getByBoundingBox(
        xMin=web_location-1e-6, xMax=web_location+1e-6,
        yMin=0.0, yMax=panel.h_longitudinal_web+1e-6,
        zMin=0.0, zMax=panel.length+1e-6
    )
    all_faces = all_faces + faces

p_l.Set(faces=all_faces, name="l_web_faces")

# Capture each of the transverse web faces
all_faces = p_t.faces.getByBoundingBox(
    xMin=1e6, xMax=-1e6,
    yMin=1e6, yMax=-1e6,
    zMin=1e6, zMax=-1e6
)

for web_location in t_web_locations:
    faces = p_t.faces.getByBoundingBox(
        xMin=web_location-1e-6, xMax=web_location+1e-6,
        yMin=0.0, yMax=panel.h_transverse_web+1e-6,
        zMin=0.0, zMax=panel.length+1e-6
    )
    all_faces = all_faces + faces

p_t.Set(faces=all_faces, name="t_web_faces")

# ----------------------------------------------------------------------------------------------------------------------------------
# Material & Section Definitions
material = model.Material(name='steel')

# Elastic properties
E = 205e9  # Pa
nu = 0.3
material.Elastic(table=((E, nu),))

rho = 7850
stp_T = 296.15
material.Density(table=((rho,float(stp_T)),))

# Plasticity constants
sigma0 = 355e6      # Initial yield stress in Pa
K = 530e6           # Strength coefficient in Pa
n = 0.26            # Hardening exponent
eps_L = 0.006       # Plateau strain

# Calculate epsilon_0 from equation
eps_0 = (sigma0 / K)**(1.0 / n) - eps_L
plastic_data = []

# Plateau point
plastic_data.append((sigma0, 0.0))  # (stress, plastic strain = 0)
plastic_data.append((sigma0, eps_L))  # plateau ends

# Generate points beyond plateau (e.g., 5 more points up to ~0.08 plastic strain)
eps_plastic_range = [0.01, 0.02, 0.04, 0.06, 0.08]
for eps in eps_plastic_range:
    stress = K * (eps_0 + eps)**n
    plastic_data.append((stress, eps_L + eps))  # plastic strain includes plateau

# Assign to material
material.Plastic(table=plastic_data)

# ----------------------------------------------------------------------------------------------------------------------------------
# Section Defintions
for component, thickness in component_thickness_map.items():
    model.HomogeneousShellSection(
        idealization=NO_IDEALIZATION,
        integrationRule=SIMPSON,
        material='steel',
        name=component,
        nodalThicknessField='',
        numIntPts=5,
        poissonDefinition=DEFAULT,
        preIntegrate=OFF,
        temperature=GRADIENT,
        thickness=float(thickness),
        thicknessField='',
        thicknessModulus=None, 
        thicknessType=UNIFORM, 
        useDensity=OFF
    )

# Create a new shell section that is N times the thickness of the web for local stiffness increases
thickness_multiplier = 15
model.HomogeneousShellSection(
    idealization=NO_IDEALIZATION,
    integrationRule=SIMPSON,
    material='steel',
    name="local-thickness",
    nodalThicknessField='',
    numIntPts=5,
    poissonDefinition=DEFAULT,
    preIntegrate=OFF,
    temperature=GRADIENT,
    thickness=thickness_multiplier * panel.t_longitudinal_web,
    thicknessField='',
    thicknessModulus=None, 
    thicknessType=UNIFORM, 
    useDensity=OFF
)

# ----------------------------------------------------------------------------------------------------------------------------------

# Assembly & Instances
model.rootAssembly.DatumCsysByDefault(CARTESIAN)
assembly = model.rootAssembly

# ----------------------------------------------------------------------------------------------------------------------------------

# The operations performed on the web from the previous code are the same as for this new panel design. We will reuse this code.

rot_1 = math.pi / 2
rot_2 = math.pi / 2
rot_plate = math.pi / 2

web_displacement = -float(panel.length) / 2
plate_displacement = float(panel.length) / 2

rot_transverse = math.pi / 2
disp_transverse = (panel.length - panel.width) / 2
rot_disp = panel.width / 2

# Create a homogenous transformation matrix to tranfer the part coordinate system into the assembly coordinate system and vise-versa
'''
The following matrices are used to reference the local and global reference frames found between the assembly and the flange and web parts.
All reference to the coordinates of a feature in either of these reference frames must be connected via the np.dot() function to ensure correspondance
of all points
'''
T_l = np.array([
    [math.cos(rot_1)*math.cos(rot_2), -math.cos(rot_2)*math.sin(rot_1), math.sin(rot_2), web_displacement],
    [math.sin(rot_1), math.cos(rot_1), 0, 0],
    [-math.cos(rot_1)*math.sin(rot_2), math.sin(rot_1)*math.sin(rot_2), math.cos(rot_2), 0],
    [0, 0, 0, 1]
])

T_inv_l = np.array([
    [math.cos(rot_1)*math.cos(rot_2), math.sin(rot_1), -math.cos(rot_1)*math.sin(rot_2), -web_displacement*math.cos(rot_1)*math.cos(rot_2)],
    [-math.cos(rot_2)*math.sin(rot_1), math.cos(rot_1), math.sin(rot_1)*math.sin(rot_2), web_displacement*math.cos(rot_2)*math.sin(rot_1)],
    [math.sin(rot_2), 0, math.cos(rot_2), -web_displacement*math.sin(rot_2)],
    [0, 0, 0, 1]
])

T_t = np.array([
    [math.cos(rot_transverse), 0, math.sin(rot_transverse), -rot_disp*math.sin(rot_transverse)],
    [0, 1, 0, 0],
    [-math.sin(rot_transverse), 0, math.cos(rot_transverse), disp_transverse + rot_disp * (1 - math.cos(rot_transverse))],
    [0, 0, 0, 1]
])

T_inv_t = np.array([
    [math.cos(rot_transverse), 0, -math.sin(rot_transverse), math.sin(rot_transverse)*(disp_transverse + rot_disp)],
    [0, 1, 0, 0],
    [math.sin(rot_transverse), 0, math.cos(rot_transverse), rot_disp - math.cos(rot_transverse)*(disp_transverse + rot_disp)],
    [0, 0, 0, 1]
])

# ----------------------------------------------------------------------------------------------------------------------------------
# Assign sections to the original geometry prior to creating the orphan mesh
assign_section(container= p_l, section_name = "panel", method='sets', set_name = 'plate_face')
assign_section(container= p_l, section_name = "l_web", method='sets', set_name = 'l_web_faces')
assign_section(container= p_l, section_name = "l_flange", method='sets', set_name = 'l_flange_faces')
assign_section(container= p_t, section_name = "t_web", method='sets', set_name = 't_web_faces')
assign_section(container= p_t, section_name = "t_flange", method='sets', set_name = 't_flange_faces')

# Create two orphan mesh parts, align the mesh of each, and then InstanceFromBooleanMerge the result
face_seed_map = {
    'plate_face': panel.mesh_plate,
    'l_web_faces': panel.mesh_longitudinal_web,
    'l_flange_faces': panel.mesh_longitudinal_flange,
}

mesh_from_faces(model.parts['longitudinal'], face_seed_map)

face_seed_map = {
    't_web_faces': panel.mesh_transverse_web,
    't_flange_faces': panel.mesh_transverse_flange
}

mesh_from_faces(model.parts['transverse'], face_seed_map)

# Align the mesh of the stiffeners and the panel
for index, location in enumerate(l_web_locations):
    _, _ = move_closest_nodes_to_axis(part=model.parts['longitudinal'], target_point=(location,0.0,0.0), axis_dof=3, free_dof=1)

for index, location in enumerate(t_web_locations):
    # Transfer from the web_locations into the reference frame of the panel
    web_edge = np.array([
        [location], 
        [0.0], 
        [0.0]
    ], dtype=float)
    web_face_point = (
        float(location),
        float(panel.h_longitudinal_web),
        0.0
    )

    panel_point = homogenous_transform(T_t, web_edge)
    # Move the panel points to align with the locations of the transverse web
    _, _ = move_closest_nodes_to_axis(part=model.parts['longitudinal'], target_point=panel_point, axis_dof=1, free_dof=3)
    
    # Move the mesh on the face of the transverse web to align with the longitudinal web
    _, _ = move_closest_nodes_to_axis(part=model.parts['transverse'], target_point=web_face_point, axis_dof=3, free_dof=2)

# Create a mesh part!
longitudinal = model.parts['longitudinal'].PartFromMesh(name='longitudinal', copySets=TRUE)
transverse = model.parts['transverse'].PartFromMesh(name='transverse', copySets=TRUE)

assembly.Instance(name = 'longitudinal', part = model.parts['longitudinal'], dependent= OFF)
assembly.Instance(name = 'transverse', part = model.parts['transverse'], dependent= OFF)

assembly.rotate(instanceList=['transverse'], axisPoint= (0,0,panel.width / 2), axisDirection= (0,1,0), angle=math.degrees(math.pi / 2))
assembly.translate(instanceList=['transverse'], vector=(0.0, 0.0, (panel.length - panel.width) / 2))

assembly.InstanceFromBooleanMerge(
    name='panel', 
    instances=(assembly.instances['longitudinal'], 
               assembly.instances['transverse']),
    domain=MESH, 
    originalInstances=SUPPRESS,
    mergeNodes=NONE,
    nodeMergingTolerance=0.001
)

# Reset the working environment
instanceNames = assembly.instances.keys()
for instanceName in instanceNames:
    del assembly.instances[instanceName]

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
assembly.Instance(dependent=ON, name='panel', part=model.parts['panel'])

# Position the flange and web properly
assembly.Instance(dependent=ON, name='panel', part=model.parts['panel'])

# Position the flange and web properly
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_1))
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,1,0), angle=math.degrees(rot_2))
assembly.translate(instanceList=['panel'], vector=(web_displacement, 0.0, 0.0))

# Boundary Conditions are more important for this trial:
# We want to support the entire sides of the panel such as they have in the paper

# ----------------------------------------------------------------------------------------------------------------------------------

# Boundary conditions

# ----------------------------------------------------------------------------------------------------------------------------------
# Rigid End
end_nodes, _ = get_nodes(assembly, instance_name='panel', bounds = [panel.length / 2 - capture_offset, panel.length / 2 + capture_offset, -panel.width / 2 - capture_offset, panel.width / 2 + capture_offset, -capture_offset, panel.h_transverse_web + capture_offset])
constraint_set = assembly.Set(name='fixed_end', nodes=(end_nodes,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='fixed_end', region=assembly.sets['fixed_end'], u1 = 0.0, u2=0.0, u3 = 0.0, ur1 = 0.0, ur2 = 0.0, ur3 = 0.0)

# Load End
load_nodes, load_node_labels = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 - capture_offset, -panel.length / 2 + capture_offset, -panel.width / 2 - capture_offset, panel.width / 2 + capture_offset, -capture_offset, panel.h_transverse_web + capture_offset])
constraint_set = assembly.Set(name='load_end', nodes=(load_nodes,))

assembly.Set(name = 'Load-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((load_node_labels[0],)))
assembly.Set(name = 'Load-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(load_node_labels[1:]))

equation_sets(
    model=model,
    name='Load-End',
    set_one='Load-Follower',
    set_two='Load-Main',
    linked_dof=[1]
)

model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='load_end', region=assembly.sets['load_end'], u2=0.0, u3 = 0.0, ur1 = 0.0, ur2 = 0.0, ur3 = 0.0)

# Side Supports
side_nodes_1, _ = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 + capture_offset, panel.length / 2 - capture_offset, -panel.width / 2 - capture_offset, l_web_locations[0] - capture_offset, -capture_offset, capture_offset])
constraint_set = assembly.Set(name='side_nodes_1', nodes=(side_nodes_1,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='side_nodes_1`', region=assembly.sets['side_nodes_1'], u2=0.0, u3 = 0.0, ur1 = 0.0, ur2 = 0.0, ur3 = 0.0)

side_nodes_2, _ = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 + capture_offset, panel.length / 2 - capture_offset, l_web_locations[-1] + capture_offset, panel.width / 2 + capture_offset, -capture_offset, capture_offset])
constraint_set = assembly.Set(name='side_nodes_2', nodes=(side_nodes_2,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='side_nodes_2`', region=assembly.sets['side_nodes_2'], u2=0.0, u3 = 0.0, ur1 = 0.0, ur2 = 0.0, ur3 = 0.0)
