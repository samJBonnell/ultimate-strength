# Program Information
# The following program is a parametric definiton of a stiffened panel allowing for the iteration of panel geometries and thicknesses for
# a variety of optimization problems. This was created to complete two of four optimization tasks in late 2024 as a learning opportunity
# prior to starting fully into 

# Sam Bonnell - UBC Labratory for Structural Efficiency MASc Student
# 2024-06-19

# ----------------------------------------------------------------------------------------------------------------------------------
# Operation Mode
# If operation mode is set to True, the program will pull panel dimensions from "temp\optimizationFile.csv" where the last line of the
# file represents the most recent iteration of the optmization.
# The variables, across a row, in "temp\optimizationFile.csv" are as follows:
#   1. Number of transverse stiffeners:    2 < x < 4
#   2. Number of longitudinal stiffeners:  2 < x < 7
#   3. Plate thickness:                    0.005 < x < 0.05
#   4. Transvers Stiffener Height:         0.050 < x < 1.00
#   5. Transverse Stiffener thickness:     0.005 < x < 0.05      
#   6. Transverse Flange thickness:        0.005 < x < 0.05
#   7. Transverse Flange width:            0.010 < x < 0.75
#   8. Longitudinal Stiffener Height:      0.010 < x < 0.95
#   9. Longitudinal Stiffener thickness:   0.005 < x < 0.05
#   10. Longitudinal Flange thickness:     0.005 < x < 0.05
#   11. Longitudinal Flange width:         0.010 < x < 0.50

# Required to add a few new parameters into the model
# Location of patch loading,
# Size of the patch loading

# Need to select the correct subsection of the 

# ----------------------------------------------------------------------------------------------------------------------------------
# Library Import
import numpy as np
import os
import json
import gzip
import random

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
import odbAccess

# ----------------------------------------------------------------------------------------------------------------------------------
# !!! Set correct working directory !!!
setPath = r'Z:\\lase\\reduced_order_models'
input_directory = r'data\\hydrostatic\\input.jsonl'
output_directory = r'data\\hydrostatic\\output.jsonl'
os.chdir(setPath)

# Configure coordinate output
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

# ----------------------------------------------------------------------------------------------------------------------------------
# Dataclass Definitions

class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dict__)

def from_dict(d):
    if isinstance(d, dict):
        return Struct(**{k: from_dict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [from_dict(i) for i in d]
    else:
        return d
    
class ThicknessGroup(object):
    def __init__(self, panel, transverse_web, transverse_flange, longitudinal_web, longitudinal_flange):
        self.panel = panel
        self.transverse_web = transverse_web
        self.transverse_flange = transverse_flange
        self.longitudinal_web = longitudinal_web
        self.longitudinal_flange = longitudinal_flange

    def unique(self):
        """Return unique thicknesses in the order they first appear."""
        seen = set()
        ordered = []
        for t in [self.panel, self.transverse_web, self.transverse_flange,
                  self.longitudinal_web, self.longitudinal_flange]:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered

    def __repr__(self):
        return "ThicknessGroup({})".format(self.__dict__)
    
class ElementStress(object):
    def __init__(self, element_id, stress):
        self.element_id = element_id
        self.stress = stress

    def to_dict(self):
        return {
            "element_id": self.element_id,
            "stress": self.stress,
        }

class PanelOutput(object):
    def __init__(self, id, max_stress, assembly_mass, element_counts,
                 stress_field, job_name, step):
        self.id = id
        self.max_stress = max_stress
        self.assembly_mass = assembly_mass
        self.element_counts = element_counts
        self.stress_field = stress_field
        self.job_name = job_name
        self.step = step

    def to_dict(self):
        return {
            "id": self.id,
            "max_stress": self.max_stress,
            "assembly_mass": self.assembly_mass,
            "element_counts": self.element_counts,
            "stress_field": [s.to_dict() for s in self.stress_field],
            "job_name": self.job_name,
            "step": self.step,
        }

# ----------------------------------------------------------------------------------------------------------------------------------
# Function Definitions

def createSurface(assembly, surface_name, instance_name, catch_point):
    """Create a surface from a single face using a known point."""
    face = assembly.instances[instance_name].faces.findAt((catch_point,))
    assembly.Surface(name=surface_name, side2Faces=face)
    print("[createSurface] Created surface '{}' on '{}' using catch point {}.".format(surface_name, instance_name, catch_point))


def createSurfaceBounds(assembly, surface_name, instance_name, bounds):
    """Create a surface by selecting all faces within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    faces = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not faces:
        raise ValueError("[createSurfaceBounds] No faces found for surface '{}' in bounds {} on instance '{}'.".format(surface_name, bounds, instance_name))
    assembly.Surface(name=surface_name, side2Faces=faces)
    print("[createSurfaceBounds] Created surface '{}' on '{}' with {} face(s).".format(surface_name, instance_name, len(faces)))


def createEdge(assembly, set_name, instance_name, catch_point):
    """Create an edge set using a known point."""
    edges = assembly.instances[instance_name].edges.findAt((catch_point,))
    assembly.Set(name=set_name, edges=edges)
    print("[createEdge] Created edge set '{}' on '{}' using catch point {}.".format(set_name, instance_name, catch_point))


def createEdgeBounds(assembly, set_name, instance_name, bounds):
    """Create an edge set by selecting all edges within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    edges = assembly.instances[instance_name].edges.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not edges:
        raise ValueError("[createEdgeBounds] No edges found for set '{}' in bounds {} on instance '{}'.".format(set_name, bounds, instance_name))
    assembly.Set(name=set_name, edges=edges)
    print("[createEdgeBounds] Created edge set '{}' on '{}' with {} edge(s).".format(set_name, instance_name, len(edges)))
    return edges

def assignSection(model, part_name, section_name, catch_points):
    """Assign a section using specific face points."""
    faces = []
    for point in catch_points:
        face = model.parts[part_name].faces.findAt((point,))
        faces.append(face)
    if not faces:
        raise ValueError("[assignSection] No faces found for section assignment on part '{}'.".format(part_name))
    all_faces = faces[0]
    for f in faces[1:]:
        all_faces += f
    model.parts[part_name].Set(name='sectionAssignment', faces=all_faces)
    model.parts[part_name].SectionAssignment(
        region=model.parts[part_name].sets['sectionAssignment'],
        sectionName=section_name,
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )
    print("[assignSection] Assigned section '{}' to {} face(s) on part '{}'.".format(section_name, len(faces), part_name))

def assignSectionBounds(model, part_name, section_name, bounds):
    """Assign a section to all faces within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    faces = model.parts[part_name].faces.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not faces:
        raise ValueError("[assignSectionBounds] No faces found for section assignment in bounds {} on part '{}'.".format(bounds, part_name))
    model.parts[part_name].Set(name='sectionAssignment', faces=faces)
    model.parts[part_name].SectionAssignment(
        region=model.parts[part_name].sets['sectionAssignment'],
        sectionName=section_name,
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )
    print("[assignSectionBounds] Assigned section '{}' to {} face(s) on part '{}'.".format(section_name, len(faces), part_name))

def createSet(assembly, set_name, instance_name, bounds):
    """Create a node set using faces found within a bounding box (note: nodes from faces may be ambiguous)."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    faces = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not faces:
        raise ValueError("[createSet] No faces found for set '{}' in bounds {} on instance '{}'.".format(set_name, bounds, instance_name))
    assembly.Set(name=set_name, nodes=faces)  # This line assumes face-based node selection
    print("[createSet] Created face-based node set '{}' with {} face(s) on instance '{}'.".format(set_name, len(faces), instance_name))
    return faces

def writeDebug(object):
    f = open("temp\\debug.csv", "w")
    f.write(str(object) + "\n")
    f.close()

def mesh(part, mesh_size, elemShape=QUAD, technique=STRUCTURED, elemCode=S4R, elemLibrary=STANDARD, constraint=FINER):
    part.seedEdgeBySize(edges=part.edges[:], size=mesh_size, constraint=constraint)
    part.setMeshControls(
        regions=part.faces[:],
        technique=technique,
        elemShape=elemShape
    )
    elemType1 = ElemType(elemCode=elemCode, elemLibrary=elemLibrary)
    part.setElementType(regions=(part.faces[:],), elemTypes=(elemType1,))
    part.generateMesh()

def write_trial_ndjson(output, path="results.jsonl"):
    with open(path, "a") as f:
        json_line = json.dumps(clean_json(output))
        f.write(json_line + "\n")

def write_trial_ndjson_gz(output, path="results.jsonl.gz"):
    with gzip.open(path, "ab") as f:
        json_line = json.dumps(output.to_dict()) + "\n"
        f.write(json_line.encode("utf-8"))

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif hasattr(obj, 'to_dict'):
        return clean_json(obj.to_dict())
    else:
        return obj

# ----------------------------------------------------------------------------------------------------------------------------------
# Design Parameters

# Load the variables from the last line of the input jsonl
with open(input_directory) as f:
    last_line = [l for l in f if l.strip()][-1]
    data = json.loads(last_line)

# data = json.loads(json_line)
panel = from_dict(data)

# Creation of a list used to create section assignments for each component of the panel
thicknesses = ThicknessGroup(
    panel=panel.t_panel,
    transverse_web=panel.t_transverse_web,
    transverse_flange=panel.t_transverse_flange,
    longitudinal_web=panel.t_longitudinal_web,
    longitudinal_flange=panel.t_longitudinal_flange,
)
ThicknessList = thicknesses.unique()

# ----------------------------------------------------------------------------------------------------------------------------------
# Start of Definition of Panel Model

# Create model object
model = mdb.Model(name='Model-1')

# Part Definitions
# Plate
model.Part(
    name='plate',
    dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)

# Transverse Stiffener
model.Part(
    name='transStiff',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY)

# Transverse Flange
model.Part(
    name='transFlange',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY)

# Longitudinal Stiffener
model.Part(
    name='longStiff',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY)

# Longitudinal Flange
model.Part(
    name='longFlange',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY)

# ----------------------------------------------------------------------------------------------------------------------------------
# Definition of Panel Model follows...
# Part Geometry

# Plate sketch & part

# Create subset parts of the plate by partitioning the part once it is created
# We need to partition from maximum Z and regularized X (Y coordinate is orthogonal to the plane of creation)
'''
We require at a minimum, 4 rectangles to partition the face such that we can use rectangular meshes for the model

X ----------------------------- X
|      |                        |       
|      x - x ------------------ |       
|      | p |                    |       
|------x - x                    |       z < - - - 
|          |                    |                |
|          |                    |                |
X ----------------------------- X                v
                                                 x
                                                 
Sketch the four rectangles in a single sketch and then partition the face in a single call

Define the force application area as a set of four points from a centroid and width/length:
- panel.patch_centroid_z
- panel.patch_centroid_x
- panel.patch_length
- panel.patch_width

Define the centroid as a positive scalar less than the panel length
Four points are defined as follows:

boundary_one = (-panel.width / 2, panel.length)
boundary_two = (-panel.width / 2, 0)
boundary_three = (panel.width / 2, 0)
boundary_four = (panel.width / 2, panel.length)

point_one = (panel.patch_centroid_x - (panel.patch_width / 2), panal.patch_centroid_z + (panel.patch_length / 2))
point_two = (panel.patch_centroid_x - (panel.patch_width / 2), panal.patch_centroid_z - (panel.patch_length / 2))
point_three = (panel.patch_centroid_x + (panel.patch_width / 2), panal.patch_centroid_z + (panel.patch_length / 2))
point_four = (panel.patch_centroid_x + (panel.patch_width / 2), panal.patch_centroid_z - (panel.patch_length / 2))

Rectangle One -     point_four & boundary_one
Rectangle Two -     point_one & boundary_two
Rectangle Three -   point_two & boundary_three
Rectangle Four -    point_three & boundary_four
'''

# ConstrainedSketch Object on the same plane as the panel
sketch = model.ConstrainedSketch(name='BaseSketch', sheetSize=100.0)
sketch.rectangle(point1=(-panel.width/2, 0), point2=(panel.width/2, panel.length))

# Arbitrary testing points
patch_centroid_z = 3.0
patch_centroid_x = 1.2

patch_width = 0.5
patch_length = 0.6

p = model.Part(name='plate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShell(sketch=sketch)

partition_sketch = model.ConstrainedSketch(name='PartitionSketch', sheetSize=100.0)
# e.g., center rectangle
patch_x = patch_centroid_x
patch_y = patch_centroid_z
pw = patch_width
pl = patch_length

boundary_one = (-panel.width / 2, panel.length)
boundary_two = (-panel.width / 2, 0)
boundary_three = (panel.width / 2, 0)
boundary_four = (panel.width / 2, panel.length)

point_one = (patch_centroid_x - (patch_width / 2), patch_centroid_z + (patch_length / 2))
point_two = (patch_centroid_x - (patch_width / 2), patch_centroid_z - (patch_length / 2))
point_three = (patch_centroid_x + (patch_width / 2), patch_centroid_z - (patch_length / 2))
point_four = (patch_centroid_x + (patch_width / 2), patch_centroid_z + (patch_length / 2))

partition_sketch.rectangle(point1=point_four, point2=boundary_one)
partition_sketch.rectangle(point1=point_one, point2=boundary_two)
partition_sketch.rectangle(point1=point_two, point2=boundary_three)
partition_sketch.rectangle(point1=point_three, point2=boundary_four)

p.PartitionFaceBySketch(
    faces=p.faces[:],
    sketch=partition_sketch
)

del model.sketches['BaseSketch']

# ----------------------------------------------------------------------------------------------------------------------------------
# Transverse Stiffener sketch & part
stepSize = 0
stiffenerPoint = -(float(panel.length)/2)
model.ConstrainedSketch(name='profileSketch', sheetSize=200.0)
for index in range(panel.num_transverse):
    if index == 0:
        stepSize = float(panel.length)/(2*panel.num_transverse)
    else:
        stepSize = float(panel.length)/(panel.num_transverse)
    stiffenerPoint += stepSize
    model.sketches['profileSketch'].Line(point1=(stiffenerPoint, 0.0), point2=(stiffenerPoint, panel.h_transverse_web))
    model.sketches['profileSketch'].VerticalConstraint(entity=model.sketches['profileSketch'].geometry[2 + index], addUndoState=False)

model.parts['transStiff'].BaseShellExtrude(depth=panel.width, sketch=model.sketches['profileSketch'])
del model.sketches['profileSketch']

# ----------------------------------------------------------------------------------------------------------------------------------
# Transverse Flange sketch & part
halfWidth = float(panel.w_transverse_flange)/2

stepSize = 0
stiffenerPoint = -(float(panel.length)/2)
model.ConstrainedSketch(name='profileSketch', sheetSize=200.0)
for index in range(panel.num_transverse):
    if index == 0:
        stepSize = float(panel.length)/(2*panel.num_transverse)
    else:
        stepSize = float(panel.length)/(panel.num_transverse)
    stiffenerPoint += stepSize
    model.sketches['profileSketch'].Line(point1=(stiffenerPoint - halfWidth, panel.h_transverse_web), point2=(stiffenerPoint + halfWidth, panel.h_transverse_web))
    model.sketches['profileSketch'].HorizontalConstraint(entity=model.sketches['profileSketch'].geometry[2 + index], addUndoState=False)

model.parts['transFlange'].BaseShellExtrude(depth=panel.width, sketch=model.sketches['profileSketch'])
del model.sketches['profileSketch']

# ----------------------------------------------------------------------------------------------------------------------------------
# Longitudinal Stiffener sketch & part
stepSize = 0
stiffenerPoint = -(float(panel.width)/2)
model.ConstrainedSketch(name='profileSketch', sheetSize=200.0)
for index in range(panel.num_longitudinal):
    stepSize = float(panel.width)/(panel.num_longitudinal + 1)
    stiffenerPoint += stepSize
    model.sketches['profileSketch'].Line(point1=(stiffenerPoint, 0.0), point2=(stiffenerPoint, panel.h_longitudinal_web))
    model.sketches['profileSketch'].VerticalConstraint(entity=model.sketches['profileSketch'].geometry[2 + index], addUndoState=False)

model.parts['longStiff'].BaseShellExtrude(depth=panel.length, sketch=model.sketches['profileSketch'])
del model.sketches['profileSketch']

# ----------------------------------------------------------------------------------------------------------------------------------
# Longitudinal Flange sketch & part
halfWidth = float(panel.w_longitudinal_flange)/2

stepSize = 0
stiffenerPoint = -(float(panel.width)/2)
model.ConstrainedSketch(name='profileSketch', sheetSize=200.0)
for index in range(panel.num_longitudinal):
    stepSize = float(panel.width)/(panel.num_longitudinal + 1)
    stiffenerPoint += stepSize
    model.sketches['profileSketch'].Line(point1=(stiffenerPoint - halfWidth, panel.h_longitudinal_web), point2=(stiffenerPoint + halfWidth, panel.h_longitudinal_web))
    model.sketches['profileSketch'].HorizontalConstraint(entity=model.sketches['profileSketch'].geometry[2 + index], addUndoState=False)

model.parts['longFlange'].BaseShellExtrude(depth=panel.length, sketch=model.sketches['profileSketch'])
del model.sketches['profileSketch']

# ----------------------------------------------------------------------------------------------------------------------------------
# Material & Section Definitions
model.Material(name='steel')
model.materials['steel'].Elastic(table=((200000000000.0,0.3),))
model.materials['steel'].Density(table=((7850,float(296.15)),))

# Section Defintions
for index in range(len(ThicknessList)):
    model.HomogeneousShellSection(
    idealization=NO_IDEALIZATION,
    integrationRule=SIMPSON,
    material='steel',
    name='t-' + str(ThicknessList[index]),
    nodalThicknessField='',
    numIntPts=5,
    poissonDefinition=DEFAULT,
    preIntegrate=OFF,
    temperature=GRADIENT,
    thickness=0.01,
    thicknessField='',
    thicknessModulus=None, 
    thicknessType=UNIFORM, 
    useDensity=OFF)

# ----------------------------------------------------------------------------------------------------------------------------------

# Assembly & Instances
model.rootAssembly.DatumCsysByDefault(CARTESIAN)
assembly = model.rootAssembly

# Create Step Object
model.StaticStep(name='Step-1', previous='Initial')

# Instance Creation
assembly.Instance(dependent=ON, name='plate-1', part=model.parts['plate'])
assembly.Instance(dependent=ON, name='transStiff-1', part=model.parts['transStiff'])
assembly.Instance(dependent=ON, name='transFlange-1', part=model.parts['transFlange'])
assembly.Instance(dependent=ON, name='longStiff-1', part=model.parts['longStiff'])
assembly.Instance(dependent=ON, name='longFlange-1', part=model.parts['longFlange'])

# Instance Transformations
assembly.rotate(instanceList=('plate-1',), axisPoint=(0,0,0), axisDirection=(1,0,0), angle=90)
assembly.rotate(instanceList=['transStiff-1', 'transFlange-1'], axisPoint=(0.0,0.0,float(panel.width)/2), axisDirection=(0.0,1.0,0.0), angle=-90)
assembly.translate(instanceList=['plate-1','longStiff-1','longFlange-1'], vector=(0.0,0.0,-float(panel.length)/2))
assembly.translate(instanceList=['transStiff-1', 'transFlange-1'], vector=(0.0,0.0,-float(panel.width)/2))

assembly.InstanceFromBooleanCut(name='longStiff', instanceToBeCut=assembly.instances['longStiff-1'], cuttingInstances=[assembly.instances['transStiff-1']])
assembly.resumeFeatures(featureNames=['transStiff-1'])

assembly.InstanceFromBooleanCut(name='longFlange', instanceToBeCut=assembly.instances['longFlange-1'], cuttingInstances=[assembly.instances['transStiff-1']])
assembly.resumeFeatures(featureNames=['transStiff-1'])

# Section Assignment
boundingLength = max(panel.length, panel.width, panel.h_transverse_web, panel.h_longitudinal_web)

# Needs to be completed after the boolean operations to ensure that the section assignment is applied to the newly created sections
assignSectionBounds(model, 'plate', 't-' + str(panel.t_panel), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])
assignSectionBounds(model, 'transStiff', 't-' + str(panel.t_transverse_web), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])
assignSectionBounds(model, 'transFlange', 't-' + str(panel.t_transverse_flange), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])
assignSectionBounds(model, 'longStiff', 't-' + str(panel.t_longitudinal_web), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])
assignSectionBounds(model, 'longFlange', 't-' + str(panel.t_longitudinal_flange), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])

# ----------------------------------------------------------------------------------------------------------------------------------
# Creation of surface and edge indexing to allow constraint creation parametrically!

surfaceList = ['plate-1', 'longFlange-2', 'transFlange-1']
for index in range(panel.num_transverse):
    surfaceList.append('transStiff-1')

# Plate, Longitudinal Flange, Longitudinal Stiffeners @ each Transverse Stiffeners
edgeList = ['longStiff-2', 'longStiff-2']
for index in range(panel.num_transverse):
    edgeList.append('longStiff-2')

for index in range(panel.num_transverse):
    edgeList.append('longFlange-2')

edgeList.append('transStiff-1')
edgeList.append('transStiff-1')

# Bounding boxes used to identify edges for constraints
tOffset = 0.01
bounds = [
    [-float(panel.width)/2, float(panel.width)/2, -tOffset, tOffset, -float(panel.length)/2, float(panel.length)/2],
    [-float(panel.width)/2, float(panel.width)/2, float(panel.h_longitudinal_web) - tOffset, float(panel.h_longitudinal_web) + tOffset, -float(panel.length)/2, float(panel.length)/2],
    [-float(panel.width)/2, float(panel.width)/2, float(panel.h_transverse_web) - tOffset, float(panel.h_transverse_web) + tOffset, -float(panel.length)/2, float(panel.length)/2]
    ]

stepSize = 0
stiffenerPoint = -(float(panel.length)/2)
for index in range(panel.num_transverse):
    if index == 0:
        stepSize = float(panel.length)/(2*panel.num_transverse)
    else:
        stepSize = float(panel.length)/(panel.num_transverse)
    stiffenerPoint += stepSize
    bounds.append([-float(panel.width)/2, float(panel.width)/2, -tOffset, panel.h_transverse_web + tOffset, stiffenerPoint - tOffset, stiffenerPoint + tOffset])

# Creating searchable indexes for each of the parts
surfaceIndex = [0, 1, 2]
for index in range(panel.num_transverse):
    surfaceIndex.append(3 + index)

edgeIndex = [0, 1]
for index in range(2):
    for edges in range(panel.num_transverse):
        edgeIndex.append(3 + edges)

edgeIndex.append(0)
edgeIndex.append(2)

constraintIndex = edgeIndex

# Creating sets of constraint edges and surfaces based on the parts and indexing completed above
for i in range(len(surfaceList)):
    createSurfaceBounds(assembly, 'surf-' + str(i), surfaceList[i], bounds[surfaceIndex[i]])

for i in range(len(edgeList)):
    createEdgeBounds(assembly, 'edge-' + str(i), edgeList[i], bounds[edgeIndex[i]])

for i in range(len(constraintIndex)):
    model.Tie(
    adjust=ON,
    main=assembly.surfaces['surf-' + str(constraintIndex[i])],
    name='Constraint' + str(i),
    positionToleranceMethod=COMPUTED,
    secondary=assembly.sets['edge-' + str(i)],
    thickness=ON,
    tieRotations=ON)

# --------------------------------------------------------------------------------------------------------------------------------------------
# Boundary Conditions

tOffset = 0.01
boundaryRegions = [
    # X-Aligned BCs
    [float(panel.width)/2 - tOffset, float(panel.width)/2 + tOffset, -tOffset, panel.h_transverse_web + tOffset, -float(panel.length)/2, float(panel.length)/2],
    [-float(panel.width)/2 - tOffset, -float(panel.width)/2 + tOffset, -tOffset, panel.h_transverse_web + tOffset, -float(panel.length)/2, float(panel.length)/2],
        
    # Z-Aligned BCs
    [-float(panel.width)/2, float(panel.width)/2, -tOffset, panel.h_transverse_web + tOffset, float(panel.length)/2 - tOffset, float(panel.length)/2 + tOffset],
    [-float(panel.width)/2, float(panel.width)/2, -tOffset, panel.h_transverse_web + tOffset, -float(panel.length)/2 - tOffset, -float(panel.length)/2 + tOffset]
    ]

boundaryList = [
    ['transFlange-1', 'transStiff-1', 'plate-1'],
    ['longFlange-2', 'longStiff-2', 'plate-1']
    ]

e1 = createEdgeBounds(assembly, 'TempSet-1', boundaryList[0][0], boundaryRegions[0])
e2 = createEdgeBounds(assembly, 'TempSet-2', boundaryList[0][1], boundaryRegions[0])
e3 = createEdgeBounds(assembly, 'TempSet-3', boundaryList[0][2], boundaryRegions[0])
e4 = createEdgeBounds(assembly, 'TempSet-4', boundaryList[0][0], boundaryRegions[1])
e5 = createEdgeBounds(assembly, 'TempSet-5', boundaryList[0][1], boundaryRegions[1])
e6 = createEdgeBounds(assembly, 'TempSet-6', boundaryList[0][2], boundaryRegions[1])

tempSet = e1 + e2 + e3 + e4 + e5 + e6
assembly.Set(edges=tempSet, name='BCs-1')

e1 = createEdgeBounds(assembly, 'TempSet-1', boundaryList[1][0], boundaryRegions[2])
e2 = createEdgeBounds(assembly, 'TempSet-2', boundaryList[1][1], boundaryRegions[2])
e3 = createEdgeBounds(assembly, 'TempSet-3', boundaryList[1][2], boundaryRegions[2])
e4 = createEdgeBounds(assembly, 'TempSet-4', boundaryList[1][0], boundaryRegions[3])
e5 = createEdgeBounds(assembly, 'TempSet-5', boundaryList[1][1], boundaryRegions[3])
e6 = createEdgeBounds(assembly, 'TempSet-6', boundaryList[1][2], boundaryRegions[3])

tempSet = e1 + e2 + e3 + e4 + e5 + e6
assembly.Set(edges=tempSet, name='BCs-2')

# Create displacement boundary conditions and apply a pressure to the whole assembly
model.DisplacementBC(amplitude=UNSET, createStepName='Step-1',distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-1', region=assembly.sets['BCs-1'], u1=0.0,u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)
model.DisplacementBC(amplitude=UNSET, createStepName='Step-1',distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='BC-2', region=assembly.sets['BCs-2'], u1=0.0, ur1=0.0)

# ----------------------------------------------------------------------------------------------------------------------------------
# Define the loading

load_area = model.parts['plate-1'].faces.findAt((patch_centroid_x, 0.0, patch_centroid_z),)
model.Pressure(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM, field='', magnitude=float(panel.pressure_magnitude), name='Load-1', region=load_area)

# ----------------------------------------------------------------------------------------------------------------------------------
# Meshing the Part
mesh(model.parts['plate'], panel.mesh_plate)
mesh(model.parts['transStiff'], panel.mesh_transverse_web)
mesh(model.parts['transFlange'], panel.mesh_transverse_flange)
mesh(model.parts['longStiff'], panel.mesh_longitudinal_web)
mesh(model.parts['longFlange'], panel.mesh_longitudinal_flange)

# ----------------------------------------------------------------------------------------------------------------------------------
# Create Job

mdb.Job(atTime=None,
        contactPrint=OFF,
        description='',
        echoPrint=OFF,
        explicitPrecision=SINGLE,
        getMemoryFromAnalysis=True,
        historyPrint=OFF,
        memory=90,
        memoryUnits=PERCENTAGE,
        model='Model-1',
        modelPrint=OFF,
        multiprocessingMode=DEFAULT, 
        name='Job-' + str(0),
        nodalOutputPrecision=SINGLE,
        numCpus=1,
        numGPUs=0,
        queue=None, 
        resultsFormat=ODB,
        scratch='',
        type=ANALYSIS,
        userSubroutine='',
        waitHours=0,
        waitMinutes=0)

# # Submit job and wait for completion

# mdb.jobs['Job-' + str(0)].submit(consistencyChecking=OFF)
# mdb.jobs['Job-' + str(0)].waitForCompletion()

# # # ----------------------------------------------------------------------------------------------------------------------------------

# trial_id = 0
# odb_path = "Job-0.odb"
# odb = odbAccess.openOdb(path=odb_path, readOnly=True)

# stressTensor = odb.steps['Step-1'].frames[-1].fieldOutputs['S'].getSubset(position=CENTROID)
# vonMisesStress = stressTensor.getScalarField(invariant=MISES)

# stress_field = []
# element_counts = []
# max_stress = 0.0
# offset = 0

# # Top surface blocks only (odd indices)
# for index in range(1, len(vonMisesStress.bulkDataBlocks), 2):
#     temp = vonMisesStress.bulkDataBlocks[index]
#     element_labels = temp.elementLabels
#     stress_data = temp.data

#     element_counts.append(len(stress_data))
#     temp_max = max([val[0] for val in stress_data])
#     max_stress = max(max_stress, temp_max)

#     for j in range(len(stress_data)):
#         stress_field.append(
#             ElementStress(
#                 element_id=int(element_labels[j] + offset),
#                 stress=float(stress_data[j][0])
#             )
#         )

#     offset += element_labels[-1]

# assembly_mass = assembly.getMassProperties()['mass']

# # --- Compose Output and Save ---:

# panel_output = PanelOutput(
#     id=str(panel.id),
#     max_stress=max_stress,
#     assembly_mass=assembly_mass,
#     element_counts=element_counts,
#     stress_field=stress_field,
#     job_name="Job-0",
#     step="Step-1"
# )

# write_trial_ndjson(panel_output, path=output_directory)
# odb.close()