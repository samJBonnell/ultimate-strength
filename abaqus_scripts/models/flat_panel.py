# Sam Bonnell - UBC Labratory for Structural Efficiency MASc Student
# 2025-09-25

# ----------------------------------------------------------------------------------------------------------------------------------
# Library Import
import numpy as np
import os
import json
import math
from datetime import datetime
import re

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
import os
from os.path import join, exists

# Define project root (change this to match your setup)
project_root = 'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength'
sys.path.insert(0, project_root)

# Define paths
working_directory = join(project_root, 'abaqus_scripts', 'working')
input_directory = join(project_root, 'data', 'input.jsonl')
output_directory = join(project_root, 'data', 'output.jsonl')

# Create working directory if it doesn't exist
if not exists(working_directory):
    os.makedirs(working_directory)

# Change to working directory
os.chdir(working_directory)
# Configure coordinate output
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

# ----------------------------------------------------------------------------------------------------------------------------------
# Design Parameters
# Load the variables from the last line of the input jsonl
from us_lib.abaqus_io import from_dict, ThicknessGroup, ModelOutput, Element, Stress
from us_lib.data.reader import write_trial_ndjson

from abq_lib.node_utilities import get_nodes
from abq_lib.mesh_utilities import mesh_from_faces
from abq_lib.section_utilities import assign_section

with open(input_directory) as f:
    last_line = [l for l in f if l.strip()][-1]
    data = json.loads(last_line)

panel = from_dict(data)

# Creation of a list used to create section assignments for each component of the panel
thicknesses = ThicknessGroup(
    panel=panel.t_panel,
)

ThicknessList = thicknesses.unique()

model_name = str(panel.model_name)
job_name = str(panel.job_name)

# ----------------------------------------------------------------------------------------------------------------------------------
# Start of Definition of Panel Model

# Create model object
model = mdb.Model(name=model_name)

# ----------------------------------------------------------------------------------------------------------------------------------

half_width = float(panel.width) / 2

# base plate line
sketch = model.ConstrainedSketch(name='geometry', sheetSize=4.0)
sketch.Line(point1=(-half_width, 0.0), point2=(half_width, 0.0))

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

# ----------------------------------------------------------------------------------------------------------------------------------

# Assembly & Instances
model.rootAssembly.DatumCsysByDefault(CARTESIAN)
assembly = model.rootAssembly

# ----------------------------------------------------------------------------------------------------------------------------------
# Define loading steps

model.StaticStep(
   name='Step-1',
   previous='Initial',
   nlgeom=OFF,
   maxNumInc=50
)

# ----------------------------------------------------------------------------------------------------------------------------------

face_seed_map = {
    'PlateFace': panel.mesh_plate,
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
assign_section(container= p, section_name = "t-{}".format(panel.t_panel), method='sets', set_name = 'PlateFace')

# Create a mesh part!
p = model.parts['plate'].PartFromMesh(name='panel', copySets=TRUE)

# ----------------------------------------------------------------------------------------------------------------------------------
# Find the node closest to the centroid of the face
assembly.regenerate()
capture_offset = 0.001

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
assembly.Instance(dependent=ON, name='panel', part=model.parts['panel'])

# Position the flange and web properly
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_1))
assembly.rotate(instanceList=['panel'], axisPoint= (0,0,0), axisDirection=(0,1,0), angle=math.degrees(rot_2))
assembly.translate(instanceList=['panel'], vector=(web_displacement, 0.0, 0.0))

# ----------------------------------------------------------------------------------------------------------------------------------

# Boundary conditions

# ----------------------------------------------------------------------------------------------------------------------------------

# Edge Constraint
# Plate-edge BC (zero motion in x3 direction)
boundary_regions = [
    # X-Aligned BCs
    [float(panel.length)/2 - capture_offset, float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, capture_offset],
    [-float(panel.length)/2 - capture_offset, -float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, capture_offset],
        
    # Y-Aligned BCs
    [-float(panel.length)/2, float(panel.length)/2, float(panel.width)/2 - capture_offset, float(panel.width)/2 + capture_offset, -capture_offset, capture_offset],
    [-float(panel.length)/2, float(panel.length)/2, -float(panel.width)/2 - capture_offset, -float(panel.width)/2 + capture_offset, -capture_offset, capture_offset]
    ]

# Capture all of the edges of the plate
labels = []
for index, region in enumerate(boundary_regions):
    _, new_labels = get_nodes(assembly, instance_name='panel', bounds=region)
    labels.extend(new_labels)

plate_edge_BC_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(labels)
plate_edge_set = assembly.Set(name='plate-edge', nodes=plate_edge_BC_nodes)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Edge-BC', region=plate_edge_set, u1=0.0, u2=0.0, u3=0.0)
# --------------------------------------------------------------------------------------------------------------------------------------------
# Load application

# We want to capture a set of nodes in a region +- the width of the patch at patch_location
xMin = panel.pressure_location[0] - panel.pressure_patch_size[0] - capture_offset
xMax = panel.pressure_location[0] + panel.pressure_patch_size[0] + capture_offset

yMin = panel.pressure_location[1] - panel.pressure_patch_size[1] - capture_offset
yMax = panel.pressure_location[1] + panel.pressure_patch_size[1] + capture_offset

zMin = -capture_offset
zMax = capture_offset

bounds = [xMin, xMax, yMin, yMax, zMin, zMax]

load_nodes, load_labels = get_nodes(assembly, instance_name='panel', bounds=bounds)

load_nodes = assembly.instances['panel'].nodes.sequenceFromLabels(load_labels)
load_set = assembly.Set(name='load_set', nodes=load_nodes)
load_region = regionToolset.Region(nodes=load_nodes)

# Create Step Object
model.ConcentratedForce(
    name="Load",
    createStepName="Step-1",
    region=load_set,
    distributionType=UNIFORM,
    cf1=0.0,
    cf2=0.0,
    cf3=float(panel.pressure)
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

job.submit(consistencyChecking=OFF)
job.waitForCompletion()

# Extract the Von Mises stress for each element in the 
odb = odbAccess.openOdb(path="{}.odb".format(job_name), readOnly=True)
stressTensor = odb.steps['Step-1'].frames[-1].fieldOutputs['S'].getSubset(position=INTEGRATION_POINT)
vonMisesStress = stressTensor.getScalarField(invariant=MISES)

stress_field = []
element_counts = []
offset = 0

for index in range(1, len(vonMisesStress.bulkDataBlocks), 2):
    temp = vonMisesStress.bulkDataBlocks[index]
    element_labels = temp.elementLabels
    stress_data = temp.data
    element_counts.append(len(stress_data))
    
    for j in range(len(stress_data)):
        element = Element(int(element_labels[j] + offset))
        element.add_attribute("stress", Stress(float(stress_data[j][0])))
        stress_field.append(element)
    
    offset += element_labels[-1]

# Create and save ModelOutput
model_output = ModelOutput(
    job_name=job_name,
    steps=["Step-1"],
    element_counts=element_counts
)
model_output.add_step("Step-1", stress_field)

write_trial_ndjson(model_output, output_directory)
odb.close()