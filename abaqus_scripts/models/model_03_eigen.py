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

p = model.Part(name='longitudinal', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShellExtrude(sketch=sketch, depth=panel.length)
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

p = model.Part(name = 'transverse', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShellExtrude(sketch=sketch, depth=panel.width)
del model.sketches['geometry']

# ----------------------------------------------------------------------------------------------------------------------------------
# Material & Section Definitions
material = model.Material(name='steel')

# Elastic properties
E = 205e9  # Pa
nu = 0.3
material.Elastic(table=((E, nu),))
rho = 7850
stp_T = 296.15
material.Density(table=((rho, float(stp_T)),))

# Engineering stress-strain data
eng_stress = [
    335.654807, 336.5632097, 337.4716124, 345.6472369, 354.7312642,
    368.3573051, 385.6169569, 403.33081, 420.1362604, 441.0295231,
    454.655564, 463.7395912, 471.4610144, 476.4572294, 479.1824375,
    479.1824375
]  # MPa

eng_strain = [
    0.001515152, 0.015151515, 0.024242424, 0.026136364, 0.028787879,
    0.032575758, 0.038636364, 0.046212121, 0.056818182, 0.072727273,
    0.090909091, 0.109848485, 0.134090909, 0.159469697, 0.187878788,
    0.212878788
]

# Convert to true stress and true plastic strain
plastic_data = []
for sigma_eng, eps_eng in zip(eng_stress, eng_strain):
    # Convert MPa to Pa and calculate true stress
    sigma_true = sigma_eng * 1e6 * (1.0 + eps_eng)
    
    # Calculate true strain
    eps_true = math.log(1.0 + eps_eng)
    
    # Calculate plastic strain
    eps_plastic = eps_true - sigma_true / E
    if eps_plastic >= 0.0:
        plastic_data.append((sigma_true, eps_plastic))

if len(plastic_data) > 0:
    first_stress = plastic_data[0][0]
    plastic_data[0] = (first_stress, 0.0)

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
# Define loading steps

model.BuckleStep(
    name='Buckle-Step',
    previous='Initial',
    numEigen=1,
    maxIterations=5000
)

# model.StaticStep(
#    name='Buckle-Step',
#    previous='Initial',
#    nlgeom=OFF
# )

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

# Create instances, orient, combine, mesh, and move forwards
assembly.Instance(name = 'longitudinal', part = model.parts['longitudinal'], dependent= OFF)
assembly.Instance(name = 'transverse', part = model.parts['transverse'], dependent= OFF)

assembly.rotate(instanceList=['transverse'], axisPoint= (0,0,panel.width / 2), axisDirection= (0,1,0), angle=math.degrees(math.pi / 2))
assembly.translate(instanceList=['transverse'], vector=(0.0, 0.0, (panel.length - panel.width) / 2))

assembly.InstanceFromBooleanMerge(
    name='panel-model', 
    instances=(assembly.instances['longitudinal'], 
               assembly.instances['transverse']),
    domain=GEOMETRY, 
    originalInstances=SUPPRESS,
    keepIntersections=ON
)

# Reset the working environment
instanceNames = assembly.instances.keys()
for instanceName in instanceNames:
    del assembly.instances[instanceName]

p = model.parts['panel-model']

# Capture the face sets of the new parts to allow meshing and section assignment
capture_offset = 1e-5

# Capture the plate faces
plate_faces = p.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax=half_width + capture_offset,
    yMin=-capture_offset, yMax=capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p.Set(faces=plate_faces, name='plate_face')

# Capture the longitudinal flange faces
flange_faces = p.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax = half_width + capture_offset,
    yMin=panel.h_longitudinal_web - capture_offset, yMax=panel.h_longitudinal_web + capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p.Set(faces=flange_faces, name='l_flange_faces')

# Capture the transverse flange faces
flange_faces = p.faces.getByBoundingBox(
    xMin=-half_width - capture_offset, xMax = half_width + capture_offset,
    yMin=panel.h_transverse_web - capture_offset, yMax=panel.h_transverse_web + capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)
p.Set(faces=flange_faces, name='t_flange_faces')

# Capture each of the longitudinal web faces
all_faces = p.faces.getByBoundingBox(
    xMin=1e6, xMax=-1e6,
    yMin=1e6, yMax=-1e6,
    zMin=1e6, zMax=-1e6
)

for web_location in l_web_locations:
    faces = p.faces.getByBoundingBox(
        xMin=web_location-1e-6, xMax=web_location+1e-6,
        yMin=0.0, yMax=panel.h_longitudinal_web+1e-6,
        zMin=0.0, zMax=panel.length+1e-6
    )
    all_faces = all_faces + faces
p.Set(faces=all_faces, name="l_web_faces")

# Capture each of the transverse web faces
transverse_faces = p.faces.getByBoundingBox(
    xMin=1e6, xMax=-1e6,
    yMin=1e6, yMax=-1e6,
    zMin=1e6, zMax=-1e6
)
points = []

for web_location in t_web_locations:
    # The web location is defined in the local reference frame of the transverse web, so we need to convert into the global frame to be able to reposition it.
    web_edge = np.array([
        [web_location], 
        [0.0], 
        [0.0]
    ], dtype=float)
    panel_point = homogenous_transform(T_t, web_edge)
    faces = p.faces.getByBoundingBox(
        xMin=-half_width - capture_offset, xMax=half_width + capture_offset,
        yMin=-capture_offset, yMax=panel.h_transverse_web + capture_offset,
        zMin=panel_point[2] - capture_offset, zMax=panel_point[2] + capture_offset,
    )
    transverse_faces = transverse_faces + faces
    points.append(panel_point)
p.Set(faces=transverse_faces, name="t_web_faces")

# ----------------------------------------------------------------------------------------------------------------------------------
# Assign sections
assign_section(container= p, section_name = "panel", method='sets', set_name = 'plate_face')
assign_section(container= p, section_name = "l_web", method='sets', set_name = 'l_web_faces')
assign_section(container= p, section_name = "l_flange", method='sets', set_name = 'l_flange_faces')
assign_section(container= p, section_name = "t_web", method='sets', set_name = 't_web_faces')
assign_section(container= p, section_name = "t_flange", method='sets', set_name = 't_flange_faces')
# Mesh Refinement Section
# Create two orphan mesh parts, align the mesh of each, and then InstanceFromBooleanMerge the result
face_seed_map = {
    'plate_face': panel.mesh_plate,
    'l_web_faces': panel.mesh_longitudinal_web,
    'l_flange_faces': panel.mesh_longitudinal_flange,
    't_web_faces': panel.mesh_transverse_web,
    't_flange_faces': panel.mesh_transverse_flange
}

# dp = p.DatumPlaneByPointNormal(point=(0.0, panel.h_longitudinal_web, 0.0), normal=(0.0,1.0,0.0))

dp = p.DatumPlaneByThreePoints(
    point1=(0.0, panel.h_longitudinal_web, 0.0),
    point2=(1.0, panel.h_longitudinal_web, 0.0),
    point3=(0.0, panel.h_longitudinal_web, 1.0)
)
p.PartitionFaceByDatumPlane(faces=transverse_faces, datumPlane=p.datums[dp.id])
# del p.features[dp.name]

p.seedPart(size=0.025, deviationFactor=0.1)
p.generateMesh()

# mesh_from_faces(model.parts['panel'], face_seed_map, technique=FREE)

model.parts['panel-model'].PartFromMesh(name='panel', copySets=TRUE)

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
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
load_set = assembly.Set(name='load_end', nodes=(load_nodes,))

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
# Global restriction prevents the movement of the global sides, but we can't restrict the motion of the edge of the panel globally
global_side_set_1, _ = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 + capture_offset, panel.length / 2 - capture_offset, -panel.width / 2 - capture_offset, l_web_locations[0] - capture_offset, -capture_offset, capture_offset])
constraint_set = assembly.Set(name='global_side_set_1', nodes=(global_side_set_1,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='global_side_set_1', region=assembly.sets['global_side_set_1'], u3 = 0.0, ur1 = 0.0, ur2 = 0.0, ur3 = 0.0)

global_side_set_2, _ = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 + capture_offset, panel.length / 2 - capture_offset, l_web_locations[-1] + capture_offset, panel.width / 2 + capture_offset, -capture_offset, capture_offset])
constraint_set = assembly.Set(name='global_side_set_2', nodes=(global_side_set_2,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='global_side_set_2', region=assembly.sets['global_side_set_2'], u3 = 0.0, ur1 = 0.0, ur2 = 0.0, ur3 = 0.0)

# Apply the local displacement BC to prevent U2 movement
side_nodes_1, _ = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 + capture_offset, panel.length / 2 - capture_offset, panel.width / 2 - capture_offset, panel.width / 2 + capture_offset, -capture_offset, capture_offset])
constraint_set = assembly.Set(name='side_nodes_1', nodes=(side_nodes_1,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='side_nodes_1', region=assembly.sets['side_nodes_1'], u2 = 0.0)

side_nodes_2, _ = get_nodes(assembly, instance_name='panel', bounds = [-panel.length / 2 + capture_offset, panel.length / 2 - capture_offset, -panel.width / 2 - capture_offset, -panel.width / 2 + capture_offset,  -capture_offset, capture_offset])
constraint_set = assembly.Set(name='side_nodes_2', nodes=(side_nodes_2,))
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='side_nodes_2', region=assembly.sets['side_nodes_2'], u2 = 0.0)

# ----------------------------------------------------------------------------------------------------------------------------------

# Load Application

# ----------------------------------------------------------------------------------------------------------------------------------
load_node = assembly.instances['panel'].nodes.sequenceFromLabels((load_node_labels[0],))
load_region = regionToolset.Region(nodes=load_node)

# Create Step Object
model.ConcentratedForce(
    name="Load",
    createStepName="Buckle-Step",
    region=load_region,
    distributionType=UNIFORM,
    cf1=panel.axial_force,
    cf2=0.0,
    cf3=0.0
)

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

job.submit(consistencyChecking=OFF)
job.waitForCompletion()