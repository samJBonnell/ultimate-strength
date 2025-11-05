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
# Part Definitions
# Dimensional definitions
l_web_spacing = float(panel.width) / (panel.num_longitudinal + 1)
t_web_spacing = float(panel.length) / (panel.num_transverse + 1)
half_width = float(panel.width) / 2
half_length = float(panel.length) / 2
l_web_locations = np.arange(-half_width + l_web_spacing, half_width, l_web_spacing)
t_web_locations = np.arange(-half_width + t_web_spacing, half_length, t_web_spacing)
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
sketch.Line(point1=(-half_length, 0.0), point2=(half_length, 0.0))

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

# Capture each of the web faces
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

p_l.Set(faces=all_faces, name="web_faces")