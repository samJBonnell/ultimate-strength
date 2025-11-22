# Sam Bonnell - UBC Labratory for Structural Efficiency MASc Student
# 2025-10-28

# ----------------------------------------------------------------------------------------------------------------------------------
# Library Import
import numpy as np
import os
import math

# ABAQUS Prefactory Information
from abaqus import *
from abaqusConstants import *
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
input_directory = join(project_root, 'data', 'model_02', 'riks', 'input.jsonl')
output_directory = join(project_root,'data', 'model_02', 'riks', 'output.jsonl')

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
    'longitudinal_web': panel.t_longitudinal_web,
    'longitudinal_flange': panel.t_longitudinal_flange
}

for component, thickness in component_thickness_map.items():
    if thickness is None:
        raise ValueError("{} thickness not set!".format(component))

# ----------------------------------------------------------------------------------------------------------------------------------
# Start of Definition of Panel Model

# Create model object
model = mdb.Model(name=model_name)

# ----------------------------------------------------------------------------------------------------------------------------------

web_spacing = float(panel.width) / (panel.num_longitudinal + 1)
half_width = float(panel.width) / 2
web_locations = np.arange(-half_width + web_spacing, half_width, web_spacing)
half_flange_width = float(panel.w_longitudinal_flange) / 2

# base plate line
sketch = model.ConstrainedSketch(name='geometry', sheetSize=4.0)
sketch.Line(point1=(-half_width, 0.0), point2=(half_width, 0.0))

for web_location in web_locations:
    # vertical web
    web_line = sketch.Line(point1=(web_location, 0.0), point2=(web_location, panel.h_longitudinal_web))
    sketch.VerticalConstraint(entity=web_line)

    # flange
    flange_line = sketch.Line(point1=(web_location - half_flange_width, panel.h_longitudinal_web), point2=(web_location + half_flange_width, panel.h_longitudinal_web))
    sketch.HorizontalConstraint(entity=flange_line)

p = model.Part(name='plate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShellExtrude(sketch=sketch, depth=panel.length)
del model.sketches['geometry']

# --- Face sets ---

capture_offset = 1e-5

# Capture the plate faces
plate_faces = p.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax=half_width + capture_offset,
    yMin=-capture_offset, yMax=capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p.Set(faces=plate_faces, name='plate_face')

# Capture the flange faces
flange_faces = p.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax = half_width + capture_offset,
    yMin=panel.h_longitudinal_web - capture_offset, yMax=panel.h_longitudinal_web + capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p.Set(faces=flange_faces, name='flange_faces')

# Capture each of the web faces
all_faces = p.faces.getByBoundingBox(
    xMin=1e6, xMax=-1e6,
    yMin=1e6, yMax=-1e6,
    zMin=1e6, zMax=-1e6
)

for web_location in web_locations:
    faces = p.faces.getByBoundingBox(
        xMin=web_location-1e-6, xMax=web_location+1e-6,
        yMin=0.0, yMax=panel.h_longitudinal_web+1e-6,
        zMin=0.0, zMax=panel.length+1e-6
    )
    all_faces = all_faces + faces

p.Set(faces=all_faces, name="web_faces")

# ----------------------------------------------------------------------------------------------------------------------------------
# Material & Section Definitions
material = model.Material(name='steel')

# Elastic properties
E = 210e9  # Pa
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
thickness_multiplier = 12
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

face_seed_map = {
    'plate_face': panel.mesh_plate,
    'web_faces': panel.mesh_longitudinal_web,
    'flange_faces': panel.mesh_longitudinal_flange,
}

mesh_from_faces(model.parts['plate'], face_seed_map)

# The operations performed on the web from the previous code are the same as for this new panel design. We will reuse this code.

rot_1 = math.pi / 2
rot_2 = math.pi / 2
rot_plate = math.pi / 2
web_displacement = -float(panel.length) / 2
plate_displacement = float(panel.length) / 2

# Create a homogenous transformation matrix to tranfer the part coordinate system into the assembly coordinate system and vise-versa
'''
The following matrices are used to reference the local and global reference frames found between the assembly and the flange and web parts.
All reference to the coordinates of a feature in either of these reference frames must be connected via the np.dot() function to ensure correspondance
of all points
'''
T_web = np.array([
    [math.cos(rot_1)*math.cos(rot_2), -math.cos(rot_2)*math.sin(rot_1), math.sin(rot_2), web_displacement],
    [math.sin(rot_1), math.cos(rot_1), 0, 0],
    [-math.cos(rot_1)*math.sin(rot_2), math.sin(rot_1)*math.sin(rot_2), math.cos(rot_2), 0],
    [0, 0, 0, 1]
])

T_inv_web = np.array([
    [math.cos(rot_1)*math.cos(rot_2), math.sin(rot_1), -math.cos(rot_1)*math.sin(rot_2), -web_displacement*math.cos(rot_1)*math.cos(rot_2)],
    [-math.cos(rot_2)*math.sin(rot_1), math.cos(rot_1), math.sin(rot_1)*math.sin(rot_2), web_displacement*math.cos(rot_2)*math.sin(rot_1)],
    [math.sin(rot_2), 0, math.cos(rot_2), -web_displacement*math.sin(rot_2)],
    [0, 0, 0, 1]
])

# ----------------------------------------------------------------------------------------------------------------------------------
# Assign sections to the original geometry prior to creating the orphan mesh
assign_section(container= p, section_name = "panel", method='sets', set_name = 'plate_face')
assign_section(container= p, section_name = "longitudinal_web", method='sets', set_name = 'web_faces')
assign_section(container= p, section_name = "longitudinal_flange", method='sets', set_name = 'flange_faces')

# Create a mesh part!
p = model.parts['plate'].PartFromMesh(name='panel', copySets=TRUE)

# ----------------------------------------------------------------------------------------------------------------------------------
# Find the node closest to the centroid of the face
assembly.regenerate()
capture_offset = 0.001

case_number = 1

# Case One
if case_number == 1:
    A_p = panel.t_panel * panel.width
    A_w = 4 * panel.t_longitudinal_web * panel.h_longitudinal_web
    A_f = 4 * panel.t_longitudinal_flange * panel.w_longitudinal_flange

    y_p = (1/2) * panel.t_panel
    y_w = (1/2) * (panel.h_longitudinal_web + panel.t_panel)
    y_f = panel.h_longitudinal_web + (1/2) * panel.t_panel

# Case Two
elif case_number == 2:
    A_p = panel.t_panel * panel.width
    A_w = 4 * panel.t_longitudinal_web * (panel.h_longitudinal_web - (1/2)*(panel.t_panel + panel.t_longitudinal_flange))
    A_f = 4 * panel.t_longitudinal_flange * panel.w_longitudinal_flange

    y_p = (1/2) * panel.t_panel
    y_w = (1/2) * (panel.h_longitudinal_web + panel.t_panel)
    y_f = panel.h_longitudinal_web + (1/2) * panel.t_panel

# Case Three
elif case_number == 3:
    A_p = panel.t_panel * panel.width
    A_w = 4 * panel.t_longitudinal_web * (panel.h_longitudinal_web - (1/2)*(panel.t_panel + panel.t_longitudinal_flange))
    A_f = 4 * panel.t_longitudinal_flange * panel.w_longitudinal_flange

    y_p = (1/2) * panel.t_panel
    y_w = panel.h_longitudinal_web + (1/2) * panel.t_panel
    y_f = (1/2) * panel.w_longitudinal_flange + panel.h_longitudinal_web + panel.t_panel
else:
    raise ValueError("Failed to capture a centroid case")

if panel.centroid == -1:
    centroid = (A_p * y_p + A_w * y_w + A_f * y_f) / (A_p + A_w + A_f) - ((1/2) * panel.t_panel)
else:
    centroid = panel.centroid

# Apply load to the left most edge of the panel
web_step = panel.width / (panel.num_longitudinal + 1)
current_step = web_step

# Define left and right target points at web locations in the assembly reference frame
centroid_free_end = np.array([
    [-float(panel.length / 2)],
    [-(float(panel.width) / 2) + web_step],
    [centroid]
])
centroid_fixed_end = np.array([
    [float(panel.length / 2)],
    [-(float(panel.width) / 2) + web_step],
    [centroid]
])

centroid_fixed_end = homogenous_transform(T_inv_web, centroid_fixed_end)
centroid_free_end = homogenous_transform(T_inv_web, centroid_free_end)

# Modify the nodes along the neutral axis of the panel to line up properly
# Must reference the points in the part reference frame, not the assembly
centroid_nodes_free, centroid_labels_free = move_closest_nodes_to_axis(part=p, target_point=centroid_free_end, axis_dof = 1, free_dof = 2, restricted_directions=[0, -1, 0])
centroid_nodes_fixed, centroid_labels_fixed = move_closest_nodes_to_axis(part=p, target_point=centroid_fixed_end, axis_dof = 1, free_dof = 2, restricted_directions=[0, -1, 0])

# ----------------------------------------------------------------------------------------------------------------------------------

# Change the local thickness of the elements to prevent local failure due to large input forces
# Set the local thickness of the free side geometry
set_local_section(
    part=p,
    seed_nodes=centroid_nodes_free,
    section_name='local-thickness',
    set_name='Local-Thickness-Free',
    restriction_type='bounds',
    restriction_params={
        'z_max': 0.5,
        'y_min': 0.001,
        'y_max': panel.mesh_longitudinal_web * 5
        },
    depth_of_search=2
    )

# Set the local thickness of the fixed side geometry
set_local_section(
    part=p,
    seed_nodes=centroid_nodes_fixed,
    section_name='local-thickness',
    set_name='Local-Thickness-Fixed',
    restriction_type='bounds',
    restriction_params={
        'z_min': 2.5,
        'y_min': 0.001,
        'y_max': panel.mesh_longitudinal_web * 5
        },
    depth_of_search=2
    )

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
assembly.Instance(dependent=ON, name='panel', part=model.parts['panel'])

# Position the flange and web properly
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_1))
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,1,0), angle=math.degrees(rot_2))
assembly.translate(instanceList=['panel'], vector=(web_displacement, 0.0, 0.0))

# ----------------------------------------------------------------------------------------------------------------------------------

# Link the y-axis displacement of the free-ends of the panel via Equations
for index, web_location in enumerate(web_locations):
    # Points along the axis we want to capture the points
    point_one = np.array([[panel.length / 2], [web_location], [0.0]])
    point_two = np.array([[-panel.length / 2], [web_location], [0.0]])

    # Labels of the nodes we have captured along the z-axis of the free ends of the panel
    _, labels_one = get_nodes_along_axis(assembly, reference_point=point_one, dof=3, instance_name='panel')
    _, labels_two = get_nodes_along_axis(assembly, reference_point=point_two, dof=3, instance_name='panel')

    labels = sorted([lab for lab in (labels_one + labels_two)])

    assembly.Set(
        name = 'Web{}-Main'.format(index),
        nodes = assembly.instances['panel'].nodes.sequenceFromLabels((labels[0],))
    )

    assembly.Set(
        name = 'Web{}-Follower'.format(index),
        nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels[1:])
    )

    equation_sets(model, 'Web{}'.format(index), 'Web{}-Follower'.format(index), 'Web{}-Main'.format(index), linked_dof= [2])

# # ----------------------------------------------------------------------------------------------------------------------------------

# Link the end of the panels together via Equations

assembly.Set(name = 'Load-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((centroid_labels_free[0],)))
assembly.Set(name = 'Load-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(centroid_labels_free[1:]))
equation_sets(model, 'Load', 'Load-Follower', 'Load-Main', linked_dof= [1])

# Link the end of the plate together in the x-axis direction to ensure it all moves as a group
# Points along the axis we want to capture the points
point_one = np.array([[panel.length / 2], [0.0], [0.0]])
point_two = np.array([[-panel.length / 2], [0.0], [0.0]])

# Labels of the nodes we have captured along the z-axis of the free ends of the panel
_, labels_one = get_nodes_along_axis(assembly, reference_point=point_one, dof=2, instance_name='panel')
_, labels_two = get_nodes_along_axis(assembly, reference_point=point_two, dof=2, instance_name='panel')

assembly.Set(name = 'Free-End-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((labels_one[0],)))
assembly.Set(name = 'Free-End-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels_one[1:]))

assembly.Set(name = 'Fixed-End-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((labels_two[0],)))
assembly.Set(name = 'Fixed-End-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels_two[1:]))

equation_sets(
    model=model,
    name='Free-End',
    set_one='Free-End-Follower',
    set_two='Free-End-Main',
    linked_dof=[1]
)

equation_sets(
    model=model,
    name='Fixed-End',
    set_one='Fixed-End-Follower',
    set_two='Fixed-End-Main',
    linked_dof=[1]
)

# ----------------------------------------------------------------------------------------------------------------------------------

# Boundary conditions

# ----------------------------------------------------------------------------------------------------------------------------------

# Rigid Body Motion Constraint
# Define a constraint point to limit the x-2 displacement of the panel to prevent RBM
constraint_point = np.array([[0.0], [0.0], [0.0]])
_, label = find_closest_node(assembly, reference_point=constraint_point, instance_name='panel')
middle_node = assembly.instances['panel'].nodes.sequenceFromLabels((label,))
constraint_set = assembly.Set(name='middle-bc', nodes=(middle_node,))

# Constrain in the x-2 direction, and allow all other motion
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Middle-BC', region=assembly.sets['middle-bc'], u2=0.0)

# Edge Constraint

# Plate-edge BC (zero motion in x3 direction)
boundary_regions = [
    # Y-Aligned BCs
    [float(panel.length)/2 - capture_offset, float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, capture_offset],
    [-float(panel.length)/2 - capture_offset, -float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, capture_offset],
        
    # X-Aligned BCs
    [-float(panel.length)/2, float(panel.length)/2, float(panel.width)/2 - capture_offset, float(panel.width)/2 + capture_offset, -capture_offset, capture_offset],
    [-float(panel.length)/2, float(panel.length)/2, -float(panel.width)/2 - capture_offset, -float(panel.width)/2 + capture_offset, -capture_offset, capture_offset]
    ]

# Capture all of the edges of the plate
labels = []
for index, region in enumerate(boundary_regions):
    _, new_labels = get_nodes(assembly, instance_name='panel', bounds=region)
    # _, new_labels = create_node_set(assembly, 'simply-supported-edge-{}'.format(index), 'plate', region)
    labels.extend(new_labels)

plate_edge_BC_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels)
plate_edge_set = assembly.Set(name='plate-edge', nodes=plate_edge_BC_nodes)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Edge-BC', region=plate_edge_set, u3=0.0)

# Fix the centroid on one side of the panel
fixed_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(centroid_labels_fixed)
fixed_centroid_BC = assembly.Set(name='Fixed-BC', nodes=fixed_nodes)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Fixed-BC', region=fixed_centroid_BC, u1=0.0)

# Teguh Boundary Conditions
# Capture the T-edges
# labels = []
# for index, region in enumerate(boundary_regions[2:]):
    # _, new_labels = get_nodes(assembly, instance_name='panel', bounds=region)
    # labels.extend(new_labels)
# 
# teguh_boundary_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels)
# plate_edge_set = assembly.Set(name='teguh-edge', nodes=teguh_boundary_nodes)
# model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Teguh-BC', region=plate_edge_set, u2=0.0)
# 
# Teguh Linked Sides
# Points on either side of the panel that we want to capture get_nodes_along_axis()
# point_one = np.array([[0.0], [panel.width / 2], [0.0]])
# point_two = np.array([[0.0], [-panel.width / 2], [0.0]])

# # Labels of the nodes we have captured along the x-axis of the free ends of the panel
# _, labels_one = get_nodes_along_axis(assembly, reference_point=point_one, dof=1, instance_name='panel')
# _, labels_two = get_nodes_along_axis(assembly, reference_point=point_two, dof=1, instance_name='panel')

# assembly.Set(name = 'Teguh-1-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((labels_one[0],)))
# assembly.Set(name = 'Teguh-1-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels_one[1:]))

# assembly.Set(name = 'Teguh-2-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((labels_two[0],)))
# assembly.Set(name = 'Teguh-2-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels_two[1:]))

# equation_sets(
#     model=model,
#     name='Teguh-1',
#     set_one='Teguh-1-Follower',
#     set_two='Teguh-1-Main',
#     linked_dof=[2]
# )

# equation_sets(
#     model=model,
#     name='Teguh-2',
#     set_one='Teguh-2-Follower',
#     set_two='Teguh-2-Main',
#     linked_dof=[2]
# )

# --------------------------------------------------------------------------------------------------------------------------------------------
# Load application
load_nodes = assembly.instances['panel'].nodes.sequenceFromLabels((centroid_labels_free[0],))
load_set = assembly.Set(name='load_set', nodes=load_nodes)
load_region = regionToolset.Region(nodes=load_nodes)
# Create Step Object

model.StaticRiksStep(
    name='Riks-Step',
    previous='Initial',
    nlgeom=ON,
    initialArcInc=0.01,
    maxArcInc=1e36,
    maxNumInc=30
)

model.ConcentratedForce(
    name="Load",
    createStepName="Riks-Step",
    region=load_set,
    distributionType=UNIFORM,
    cf1=float(panel.axial_force),
    cf2=0.0,
    cf3=0.0
)

# Create the imperfection and regenerate the assembly
L_w = panel.width
L_l = panel.length
part = model.parts['panel']
apply_geometric_imperfection(part, lambda x, y, z: (0.0, 0.0015*cos((pi/L_w)*x)*sin((pi/L_l)*z), 0.0))
assembly.regenerate()
# ----------------------------------------------------------------------------------------------------------------------------------
# Create Job

job = mdb.Job(
    atTime=None,
    contactPrint=OFF,
    description='',
    echoPrint=OFF,
    explicitPrecision=SINGLE,
    getMemoryFromAnalysis=True,
    historyPrint=OFF,
    memory=90,
    memoryUnits=PERCENTAGE,
    model=model_name,
    modelPrint=OFF,
    multiprocessingMode=DEFAULT,
    name=job_name,
    nodalOutputPrecision=SINGLE,
    numCpus=int(panel.numCpus),
    numGPUs=int(panel.numGpus),
    queue=None,
    resultsFormat=ODB,
    scratch='',
    type=ANALYSIS,
    userSubroutine='',
    waitHours=0,
    waitMinutes=0,
    numDomains=int(panel.numCpus)
)

job.writeInput()