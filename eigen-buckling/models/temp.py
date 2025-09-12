# Sam Bonnell - UBC Labratory for Structural Efficiency MASc Student
# 2025-08-11

# ----------------------------------------------------------------------------------------------------------------------------------
# Library Import
import numpy as np
import os
import json
import gzip
import random
import math
from datetime import datetime

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

# # Ensure that we can capture the utils\ folder at the start
import sys
import os

# # Add the parent directory (project root) to Python path
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# ----------------------------------------------------------------------------------------------------------------------------------

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n\nStart Time: {}\n\n".format(current_time))

# !!! Set correct working directory !!!
working_directory = r'C:\\Users\\sbonnell\\Desktop\\lase\\projects\\ultimate-strength\\eigen-buckling'
input_directory = r'data\\input.jsonl'
output_directory = r'data\\output.jsonl'
os.chdir(working_directory)

# !!! Set correct job name
job_name = 'parametric-panel'

# Configure coordinate output
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

# ----------------------------------------------------------------------------------------------------------------------------------
# Design Parameters
# Load the variables from the last line of the input jsonl
from utils.IO_utils import from_dict, ThicknessGroup
from utils.node_utilities import get_nodes_along_axis, move_closest_nodes_to_axis
from utils.mesh_utilities import mesh_from_faces
from utils.section_utilities import assign_section_sets, set_local_element_thickness
from utils.transformation_utilities import homogenous_transform
from utils.constraint_utilities import equation_sets

with open(input_directory) as f:
    last_line = [l for l in f if l.strip()][-1]
    data = json.loads(last_line)

panel = from_dict(data)

# Creation of a list used to create section assignments for each component of the panel
thicknesses = ThicknessGroup(
    panel=panel.t_panel,
    longitudinal_web=panel.t_longitudinal_web,
    longitudinal_flange=panel.t_longitudinal_flange,
)

ThicknessList = thicknesses.unique()

# ----------------------------------------------------------------------------------------------------------------------------------
# Start of Definition of Panel Model

# Create model object
model = mdb.Model(name='eigen')

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
p.Set(faces=plate_faces, name='PlateFace')

# Capture the flange faces
flange_faces = p.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax = half_width + capture_offset,
    yMin=panel.h_longitudinal_web - capture_offset, yMax=panel.h_longitudinal_web + capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p.Set(faces=flange_faces, name='FlangeFaces')

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

p.Set(faces=all_faces, name="WebFaces")

# ----------------------------------------------------------------------------------------------------------------------------------
# Material & Section Definitions
material = model.Material(name='steel')

# Elastic properties
E = 200e9  # Pa
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
for index in range(len(ThicknessList)):
    model.HomogeneousShellSection(
    idealization=NO_IDEALIZATION,
    integrationRule=SIMPSON,
    material='steel',
    name="t-{}".format(ThicknessList[index]),
    nodalThicknessField='',
    numIntPts=5,
    poissonDefinition=DEFAULT,
    preIntegrate=OFF,
    temperature=GRADIENT,
    thickness=float(ThicknessList[index]),
    thicknessField='',
    thicknessModulus=None, 
    thicknessType=UNIFORM, 
    useDensity=OFF)

# Create a new shell section that is N times the thickness of the web for local stiffness increases
thickness_multiplier = 5
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
# Define loading steps

model.BuckleStep(
    name='Buckle-Step',
    previous='Initial',
    numEigen=5,
    maxIterations=500
)

# model.StaticStep(
#    name='Buckle-Step',
#    previous='Initial',
#    nlgeom=OFF
# )

# ----------------------------------------------------------------------------------------------------------------------------------

face_seed_map = {
    'PlateFace': panel.mesh_plate,
    'WebFaces': panel.mesh_longitudinal_web,
    'FlangeFaces': panel.mesh_longitudinal_flange,
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

# Leave for future testing, if so needed
# forward_test = np.array([
#     [0.900],
#     [0.075],
#     [3.00],
#     [1]
# ])

# backwards_test = np.array([
#     [1.5],
#     [0.3],
#     [0.1],
#     [1]
# ])

# print("Forward Test")
# print(np.dot(T_web, forward_test))

# print("Backwards Test\n")
# print(np.dot(T_inv_web, backwards_test))

# ----------------------------------------------------------------------------------------------------------------------------------
# Section Assignment

# assign_section_sets(part = p, section_name = "t-{}".format(panel.t_panel), set_name = 'PlateFace')
# assign_section_sets(part = p, section_name = "t-{}".format(panel.t_longitudinal_web), set_name = 'WebFaces')
# assign_section_sets(part = p, section_name = "t-{}".format(panel.t_longitudinal_flange), set_name = 'FlangeFaces')

# ----------------------------------------------------------------------------------------------------------------------------------
# Create a mesh part!
p = model.parts['plate'].PartFromMesh(name='panel', copySets=TRUE)

assign_section_sets(part = p, section_name = "t-{}".format(panel.t_panel), set_name = 'PlateFace')
assign_section_sets(part = p, section_name = "t-{}".format(panel.t_longitudinal_web), set_name = 'WebFaces')
assign_section_sets(part = p, section_name = "t-{}".format(panel.t_longitudinal_flange), set_name = 'FlangeFaces')

# ----------------------------------------------------------------------------------------------------------------------------------
# Find the node closest to the centroid of the face
assembly.regenerate()
capture_offset = 0.001
centroid_offset = -0.002

A_panel = panel.width * panel.t_panel
A_web = panel.h_longitudinal_web * panel.t_longitudinal_web * panel.num_longitudinal
A_flange = panel.w_longitudinal_flange * panel.t_longitudinal_flange * panel.num_longitudinal

y_panel = panel.t_panel / 2
y_web = panel.t_panel + (panel.h_longitudinal_web) / 2
y_flange = (panel.t_panel / 2) + panel.h_longitudinal_web + (panel.t_longitudinal_flange / 2)

# If the TIE constraints defined between the edges and the surfaces are not set with thickness=ON, you need to consider the panels to each start at the half-thickness of the surface
centroid = (A_panel * y_panel + A_web * y_web + A_flange * y_flange) / (A_panel + A_web + A_flange) + centroid_offset

# The centroid is based on the half-thickness of the plates
A_panel = panel.width * panel.t_panel
A_web = (panel.h_longitudinal_web - (panel.t_panel + panel.t_longitudinal_flange) / 2) * panel.t_longitudinal_web * panel.num_longitudinal
A_flange = panel.w_longitudinal_flange * panel.t_longitudinal_flange * panel.num_longitudinal

# The plate is instantiated at (0, 0), therefore, the centroid is simply 0
y_panel = 0.0
y_web = panel.h_longitudinal_web / 2
y_flange = panel.h_longitudinal_web

centroid = (A_panel * y_panel + A_web * y_web + A_flange * y_flange) / (A_panel + A_web + A_flange) + centroid_offset

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
_, _ = move_closest_nodes_to_axis(part=p, target_point=centroid_free_end, axis_dof = 1, free_dof = 2)
_, _ = move_closest_nodes_to_axis(part=p, target_point=centroid_fixed_end, axis_dof = 1, free_dof = 2)

# ----------------------------------------------------------------------------------------------------------------------------------

# Change the local thickness of the elements to prevent local failure due to large input forces
set_local_element_thickness(part=p, target_point=centroid_free_end, axis_dof = 1, depth_of_search = 2, set_name='left-thickness-region')
set_local_element_thickness(part=p, target_point=centroid_fixed_end, axis_dof = 1, depth_of_search = 2, set_name='right_thickness_region')

# ----------------------------------------------------------------------------------------------------------------------------------

# # Reference the coordinates in the global geometry and use the inverse matrix to map them into the plate geometry
# for index, web_location in enumerate(web_locations):
#     # Location of each of the webs that intersect the plate
#     web_point = np.array([[0.0], [float(web_location)], [0.0]])

#     # Transform back into the plate reference frame
#     web_point = homogenous_transform(T_inv_plate, web_point)

#     # Align the plate mesh to the axis along the x-axis to this location
#     _, _ = move_closest_nodes_to_axis(model.parts['plate-mesh'], web_point, axis_dof = 2, free_dof = 1)

# # Linking flange, web, and plate together

# # Parent, Child node labels
# web_plate_set = set()
# web_flange_set = set()

# web_node_set = set()

# for index, web_location in enumerate(web_locations):
#     # Upper and lower point in the assembly reference frame that we want to capture information along. We look at the centre of the panel as we are interested for nodes along the x-axis
#     lower_point = np.array([[0.0],[web_location],[0.0]])
#     upper_point = np.array([[0.0],[web_location],[panel.h_longitudinal_web]])

#     # plate nodes. Reference the dof within the part, ie. 2
#     _, plate_nodes = get_nodes_along_axis(plate, reference_point = homogenous_transform(T_inv_plate, lower_point), dof = 2)

#     # web nodes. Reference the dof within the part, ie. 1
#     _, web_nodes_lower = get_nodes_along_axis(web, reference_point = homogenous_transform(T_inv_web, lower_point), dof = 3)
#     _, web_nodes_upper = get_nodes_along_axis(web, reference_point = homogenous_transform(T_inv_web, upper_point), dof = 3)
#     web_node_set.update(web_nodes_lower + web_nodes_upper)

#     # flange nodes. Reference the dof within the part, ie. 1
#     _, flange_nodes = get_nodes_along_axis(flange, reference_point = homogenous_transform(T_inv_web, upper_point), dof = 3)

#     # Determine parent and child sets/parts based on node count.
#     # The parent is the smaller set (fewer nodes = higher DOF along edge).
#     if len(web_nodes_lower) <= len(plate_nodes):
#         parent_set, child_set = web_nodes_lower, plate_nodes
#         parent_part_one = web
#         child_part_one = plate
#         transformation_one, transformation_two = T_web, T_inv_plate
#     else:
#         parent_set, child_set = plate_nodes, web_nodes_lower
#         parent_part_one = plate
#         child_part_one = web
#         transformation_one, transformation_two = T_plate, T_inv_web

#     # For each nodes within the parent set, we need to move the corresponding closest node in the child set to match it
#     for node_label in parent_set:
#         node = parent_part_one.nodes.sequenceFromLabels((node_label,))[0]
#         driving_point = homogenous_transform(transformation_two, homogenous_transform(transformation_one, node.coordinates))
#         _, label = move_closest_node_to_point(part = child_part_one, target_point = driving_point , free_dof = [3])
#         web_plate_set.add((node.label, label))

#     # web-flange intersection
#     # No transformations are required as they are built in the same reference frame
#     if len(web_nodes_lower) <= len(flange_nodes):
#         parent_set, child_set = web_nodes_upper, flange_nodes
#         parent_part_two = web
#         child_part_two = flange
#     else:
#         parent_set, child_set = flange_nodes, web_nodes_upper
#         parent_part_two = flange
#         child_part_two = web

#     # For each nodes within the parent set, we need to move the corresponding closest node in the child set to match it
#     moved_labels = set()
#     for node_label in parent_set:
#         node = parent_part_two.nodes.sequenceFromLabels((node_label,))[0]
#         driving_point = node.coordinates
#         _, label = move_closest_node_to_point(part = child_part_two, target_point = driving_point , free_dof = [3])
#         web_flange_set.add((node.label, label))

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
assembly.Instance(dependent=ON, name='panel', part=model.parts['panel'])

# Position the flange and web properly
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_1))
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,1,0), angle=math.degrees(rot_2))
assembly.translate(instanceList=['panel'], vector=(web_displacement, 0.0, 0.0))

# ----------------------------------------------------------------------------------------------------------------------------------

# Link the plate, web, and flange via Equations
# equation_constraint(model, assembly, parent_part_name=parent_part_one.name.replace('-mesh', ''), child_part_name=child_part_one.name.replace('-mesh', ''), nodes_to_link=web_plate_set, linked_dof=[1, 2, 3])
# equation_constraint(model, assembly, parent_part_name=parent_part_two.name.replace('-mesh', ''), child_part_name=child_part_two.name.replace('-mesh', ''), nodes_to_link=web_flange_set, linked_dof=[1, 2, 3])

# I have the nodes sets already, so I simply need to merge them

# Link the y-axis displacement of the free-ends of the panel via Equations
for index, web_location in enumerate(web_locations):
    # Points along the axis we want to capture the points
    point_one = np.array([[panel.length / 2], [web_location], [0.0]])
    point_two = np.array([[-panel.length / 2], [web_location], [0.0]])

    # Labels of the nodes we have captured along the z-axis of the free ends of the panel
    _, labels_one = get_nodes_along_axis(assembly, reference_point=point_one, dof=3, instance_name='panel')
    _, labels_two = get_nodes_along_axis(assembly, reference_point=point_two, dof=3, instance_name='panel')

    # We need to link these as a "ring", so that the displacement is consistent around the entire set of free-ends
    # Remove the nodes that occur in the upper and lower sets from the flange-web and plate-web linking
    # labels = sorted([lab for lab in (labels_one + labels_two) if lab not in web_node_set])
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


    # end_pairs = set((labels[i], labels[i + 1]) for i in range(len(labels) - 1))

    # equation_constraint(model, assembly, parent_part_name='panel', child_part_name='panel', nodes_to_link=end_pairs, linked_dof=[2])

# # ----------------------------------------------------------------------------------------------------------------------------------

# Link the end of the panels together via Equations

free_point = np.array([[-panel.length / 2], [0.0], [centroid]])

_, centroid_labels_free = get_nodes_along_axis(assembly, reference_point=free_point, dof=2, instance_name='panel')

assembly.Set(name = 'Load-Main', nodes = assembly.instances['panel'].nodes.sequenceFromLabels((centroid_labels_free[0],)))
assembly.Set(name = 'Load-Follower', nodes = assembly.instances['panel'].nodes.sequenceFromLabels(centroid_labels_free[1:]))

equation_sets(model, 'Load', 'Load-Main', 'Load-Follower', linked_dof= [1])


# # Link via constraints instead?
# # ----------------------------------------------------------------------------------------------------------------------------------

# # Boundary conditions

# # ----------------------------------------------------------------------------------------------------------------------------------

# # Rigid Body Motion Constraint
# # Define a constraint point to limit the x-2 displacement of the panel to prevent RBM
# constraint_point = np.array([[0.0], [0.0], [0.0]])
# _, label = find_closest_node(assembly, reference_point=constraint_point, instance_name='panel')
# middle_node = assembly.instances['panel'].nodes.sequenceFromLabels((label,))
# constraint_set = assembly.Set(name='middle-bc', nodes=(middle_node,))

# # Constrain in the x-2 direction, and allow all other motion
# model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Middle-BC', region=assembly.sets['middle-bc'], u2=0.0)

# # Edge Constraint

# # Plate-edge BC (zero motion in x3 direction)
# boundary_regions = [
#     # X-Aligned BCs
#     [float(panel.length)/2 - capture_offset, float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, capture_offset],
#     [-float(panel.length)/2 - capture_offset, -float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, capture_offset],
        
#     # Y-Aligned BCs
#     [-float(panel.length)/2, float(panel.length)/2, float(panel.width)/2 - capture_offset, float(panel.width)/2 + capture_offset, -capture_offset, capture_offset],
#     [-float(panel.length)/2, float(panel.length)/2, -float(panel.width)/2 - capture_offset, -float(panel.width)/2 + capture_offset, -capture_offset, capture_offset]
#     ]

# # Capture all of the edges of the plate
# labels = []
# for index, region in enumerate(boundary_regions):
#     _, new_labels = get_nodes(assembly, instance_name='panel', bounds=region)
#     # _, new_labels = create_node_set(assembly, 'simply-supported-edge-{}'.format(index), 'plate', region)
#     labels.extend(new_labels)

# plate_edge_BC_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels)
# plate_edge_set = assembly.Set(name='plate-edge', nodes=plate_edge_BC_nodes)
# model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Edge-BC', region=plate_edge_set, u3=0.0)

# # Fix the centroid on one side of the panel
# fixed_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(centroid_labels_fixed)
# fixed_centroid_BC = assembly.Set(name='Fixed-BC', nodes=fixed_nodes)
# model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Fixed-BC', region=fixed_centroid_BC, u1=0.0)

# # --------------------------------------------------------------------------------------------------------------------------------------------
# # Load application

# load_nodes = assembly.instances['panel'].nodes.sequenceFromLabels((centroid_labels_free[0],))
# load_set = assembly.Set(name='load_set', nodes=load_nodes)
# load_region = regionToolset.Region(nodes=load_nodes)
# # Create Step Object

# model.ConcentratedForce(
#     name="Load",
#     createStepName="Buckle-Step",
#     region=load_set,
#     distributionType=UNIFORM,
#     cf1=float(panel.axial_force),
#     cf2=0.0,
#     cf3=0.0
# )

# # ----------------------------------------------------------------------------------------------------------------------------------
# # Create Job

# job = mdb.Job(
#     atTime=None,
#     contactPrint=OFF,
#     description='',
#     echoPrint=OFF,
#     explicitPrecision=SINGLE,
#     getMemoryFromAnalysis=True,
#     historyPrint=OFF,
#     memory=90,
#     memoryUnits=PERCENTAGE,
#     model='Model-1',
#     modelPrint=OFF,
#     multiprocessingMode=DEFAULT,
#     name=job_name,
#     nodalOutputPrecision=SINGLE,
#     numCpus=4,
#     numGPUs=0,
#     queue=None,
#     resultsFormat=ODB,
#     scratch='',
#     type=ANALYSIS,
#     userSubroutine='',
#     waitHours=0,
#     waitMinutes=0,
#     numDomains=4
# )

# job.writeInput()

# job.submit(consistencyChecking=OFF)
# job.waitForCompletion()