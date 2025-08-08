# Program Information
# The following program is a parametric definiton of a stiffened panel allowing for the iteration of panel geometries and thicknesses for
# a variety of optimization problems. This was created to complete two of four optimization tasks in late 2024 as a learning opportunity
# prior to starting fully into 

# Sam Bonnell - UBC Labratory for Structural Efficiency MASc Student
# 2024-06-19

# ----------------------------------------------------------------------------------------------------------------------------------

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
# !!! Set correct working directory !!!
setPath = r'Z:\\lase\\yuecheng'
input_directory = r'data\\input.jsonl'
output_directory = r'data\\output.jsonl'
os.chdir(setPath)

# !!! Set correct job name
job_name = 'buckling_eigen_panel'

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
    def __init__(self, panel, longitudinal_web, longitudinal_flange):
        self.panel = panel
        self.longitudinal_web = longitudinal_web
        self.longitudinal_flange = longitudinal_flange

    def unique(self):
        """Return unique thicknesses in the order they first appear."""
        seen = set()
        ordered = []
        for t in [self.panel, self.longitudinal_web, self.longitudinal_flange]:
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

    @staticmethod
    def from_dict(d):
        return ElementStress(d["element_id"], d["stress"])

    def to_dict(self):
        return {"element_id": self.element_id, "stress": self.stress}

class ElementDisplacement(object):
    def __init__(self, element_id, x, y, z):
        self.element_id = element_id
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_dict(d):
        return ElementDisplacement(d["element_id"], d["x"], d["y"], d["z"])

    def to_dict(self):
        return {
            "element_id": self.element_id,
            "x": self.x,
            "y": self.y,
            "z": self.z
        }
    
class PanelOutput(object):
    def __init__(self, id, element_counts, stress_field, displacement_field, job_name, steps):
        self.id = id
        self.element_counts = element_counts                  # Dict[str, int]
        self.stress_field = stress_field                      # Dict[str, List[ElementStress]]
        self.displacement_field = displacement_field          # Dict[str, List[ElementDisplacement]]
        self.job_name = job_name
        self.steps = steps                                    # List of step names (str)

    @staticmethod
    def from_dict(d):
        stress_field = {}
        for step, stresses in d.get("stress_field", {}).items():
            stress_field[step] = [ElementStress.from_dict(s) for s in stresses]

        displacement_field = {}
        for step, disps in d.get("displacement_field", {}).items():
            displacement_field[step] = [ElementDisplacement.from_dict(s) for s in disps]

        return PanelOutput(
            id=d["id"],
            element_counts=d["element_counts"],
            stress_field=stress_field,
            displacement_field=displacement_field,
            job_name=d["job_name"],
            steps=d["steps"]
        )

    def to_dict(self):
        return {
            "id": self.id,
            "element_counts": self.element_counts,
            "stress_field": {
                step: [s.to_dict() for s in stresses]
                for step, stresses in self.stress_field.items()
            },
            "displacement_field": {
                step: [d.to_dict() for d in disps]
                for step, disps in self.displacement_field.items()
            },
            "job_name": self.job_name,
            "steps": self.steps
        }


# ----------------------------------------------------------------------------------------------------------------------------------
# Function Definitions

def create_surface_point(assembly, surface_name, instance_name, catch_point):
    """Create a surface from a single face using a known point."""
    face = assembly.instances[instance_name].faces.findAt((catch_point,))
    assembly.Surface(name=surface_name, side2Faces=face)
    print("[create_surface_point] Created surface '{}' on '{}' using catch point {}.".format(surface_name, instance_name, catch_point))

def create_surface_bounds(assembly, surface_name, instance_name, bounds):
    """Create a surface by selecting all faces within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    faces = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not faces:
        raise ValueError("[create_surface_bounds] No faces found for surface '{}' in bounds {} on instance '{}'.".format(surface_name, bounds, instance_name))
    assembly.Surface(name=surface_name, side2Faces=faces)
    print("[create_surface_bounds] Created surface '{}' on '{}' with {} face(s).".format(surface_name, instance_name, len(faces)))

def create_edge_set_point(assembly, set_name, instance_name, catch_point):
    """Create an edge set using a known point."""
    edges = assembly.instances[instance_name].edges.findAt((catch_point,))
    assembly.Set(name=set_name, edges=edges)
    print("[create_edge_set_point] Created edge set '{}' on '{}' using catch point {}.".format(set_name, instance_name, catch_point))

def create_edge_set_bounds(assembly, set_name, instance_name, bounds):
    """Create an edge set by selecting all edges within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    edges = assembly.instances[instance_name].edges.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not edges:
        raise ValueError("[create_edge_set_bounds] No edges found for set '{}' in bounds {} on instance '{}'.".format(set_name, bounds, instance_name))
    assembly.Set(name=set_name, edges=edges)
    print("[create_edge_set_bounds] Created edge set '{}' on '{}' with {} edge(s).".format(set_name, instance_name, len(edges)))
    return edges

def create_node_set_bounds(assembly, set_name, instance_name, bounds):
    """Create a node set by selecting all edges within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    nodes = assembly.instances[instance_name].nodes.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not nodes:
        raise ValueError("[create_node_set_bounds] No nodes found for set '{}' in bounds {} on instance '{}'.".format(set_name, bounds, instance_name))
    assembly.Set(name=set_name, nodes=nodes)
    print("[create_node_set_bounds] Created node set '{}' on '{}' with {} node(s).".format(set_name, instance_name, len(nodes)))
    return nodes

def assign_section_point(model, part_name, section_name, catch_points):
    """Assign a section using specific face points."""
    faces = []
    for point in catch_points:
        face = model.parts[part_name].faces.findAt((point,))
        faces.append(face)
    if not faces:
        raise ValueError("[assign_section_point] No faces found for section assignment on part '{}'.".format(part_name))
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
    print("[assign_section_point] Assigned section '{}' to {} face(s) on part '{}'.".format(section_name, len(faces), part_name))

def assign_section_bounds(model, part_name, section_name, bounds):
    """Assign a section to all faces within a bounding box."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    faces = model.parts[part_name].faces.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not faces:
        raise ValueError("[assign_section_bounds] No faces found for section assignment in bounds {} on part '{}'.".format(bounds, part_name))
    model.parts[part_name].Set(name='sectionAssignment', faces=faces)
    model.parts[part_name].SectionAssignment(
        region=model.parts[part_name].sets['sectionAssignment'],
        sectionName=section_name,
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )
    print("[assign_section_bounds] Assigned section '{}' to {} face(s) on part '{}'.".format(section_name, len(faces), part_name))

def create_node_set(assembly, set_name, instance_name, bounds):
    """Create a node set using faces found within a bounding box (note: nodes from faces may be ambiguous)."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    faces = assembly.instances[instance_name].faces.getByBoundingBox(
        xMin=x_min, xMax=x_max,
        yMin=y_min, yMax=y_max,
        zMin=z_min, zMax=z_max
    )
    if not faces:
        raise ValueError("[create_node_set] No faces found for set '{}' in bounds {} on instance '{}'.".format(set_name, bounds, instance_name))
    assembly.Set(name=set_name, nodes=faces)  # This line assumes face-based node selection
    print("[create_node_set] Created face-based node set '{}' with {} face(s) on instance '{}'.".format(set_name, len(faces), instance_name))
    return faces

def write_debug_file(object, file):
    f = open(file, "w")
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

def find_closest_node(nodes, target_point, min_dist=1e20):
    for node in nodes:
        x, y, z = node.coordinates
        dx = x - target_point[0]
        dy = y - target_point[1]
        dz = z - target_point[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < min_dist:
            min_dist = dist
            closest_node = node

    return closest_node

def get_nodes_along_axis(nodes, reference_point, dof, max_bound, capture_offset):    
    lower = [reference_point[0] - capture_offset,
             reference_point[1] - capture_offset,
             reference_point[2] - capture_offset]
    upper = [reference_point[0] + capture_offset,
             reference_point[1] + capture_offset,
             reference_point[2] + capture_offset]
    
    idx = dof - 1
    lower[idx] = -max_bound
    upper[idx] =  max_bound
    
    return nodes.getByBoundingBox(lower[0], lower[1], lower[2],
                                  upper[0], upper[1], upper[2])

def move_closest_nodes_to_neutral_axis(part, target_point, axis_dof = 1, free_dof = 2):
    """Move the set of all nodes closest to the axis defined by the """
    reference_point = find_closest_node(part.nodes, target_point)

    # Capture all of the points on the part that lie along the line of action of the dof
    capture_offset = 0.001
    max_bound = 1e5 # A large number that will never be reached by the bounds of the part

    # Capture nodes along the neutral axis defined by the dof and the reference_point
    nodes = get_nodes_along_axis(part.nodes, reference_point.coordinates, axis_dof, max_bound, capture_offset)

    for node in nodes:
        #print("Node: '{}'- Position: '{}'".format(node.label, node.coordinates))
        # Find the coordinates of the point and presribe the neutral axis location
        temp_coords = node.coordinates

        coordinates = list(node.coordinates)
        coordinates[free_dof - 1] = target_point[free_dof - 1]

        # Move mesh to match this neutral axis location
        part.editNode(nodes=(node,), coordinates=(tuple(coordinates),))
        print("[move_closest_nodes_to_neutral_axis] Moved node '{}' from location {} to '{}'.".format(node.label, temp_coords, node.coordinates))

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
model = mdb.Model(name='Model-1')

# Part Definitions
# Plate
model.Part(
    name='plate',
    dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)

# Longitudinal Stiffener
model.Part(
    name='web',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY)

# Longitudinal Flange
model.Part(
    name='flange',
    dimensionality=THREE_D,
    type=DEFORMABLE_BODY)

# ----------------------------------------------------------------------------------------------------------------------------------
# Definition of Panel Model follows...
# Part Geometry

# Plate sketch & part
sketch = model.ConstrainedSketch(name='profileSketch', sheetSize=100.0)
sketch.rectangle(point1=(-panel.width/2, 0), point2=(panel.width/2, panel.length))

p = model.Part(name='plate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShell(sketch=sketch)
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

model.parts['web'].BaseShellExtrude(depth=panel.length, sketch=model.sketches['profileSketch'])
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

model.parts['flange'].BaseShellExtrude(depth=panel.length, sketch=model.sketches['profileSketch'])
del model.sketches['profileSketch']

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
    name='t-' + str(ThicknessList[index]),
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

model.BuckleStep(
    name='Buckle-Step',
    previous='Initial',
    numEigen=5,
    maxIterations=500
)

# ----------------------------------------------------------------------------------------------------------------------------------
# Meshing the Part
mesh(model.parts['plate'], panel.mesh_plate)
mesh(model.parts['web'], panel.mesh_longitudinal_web)
mesh(model.parts['flange'], panel.mesh_longitudinal_flange)

model.parts['plate'].PartFromMesh(name='plate-mesh', copySets=FALSE)
model.parts['web'].PartFromMesh(name='web-mesh', copySets=FALSE)
model.parts['flange'].PartFromMesh(name='flange-mesh', copySets=FALSE)

# Instance Transformations
# Position the plate in the correct orientation

rot_1 = math.pi / 2
rot_2 = math.pi / 2
displacement = -float(panel.length) / 2

# Create a homogenous transformation matrix to tranfer the part coordinate system into the assembly coordinate system and vise-versa
'''
The following matrices are used to reference the local and global reference frames found between the assembly and the flange and web parts.
All reference to the coordinates of a feature in either of these reference frames must be connected via the np.dot() function to ensure correspondance
of all points
'''
T = np.array([
    [math.cos(rot_1)*math.cos(rot_2), -math.cos(rot_2)*math.sin(rot_1), math.sin(rot_2), displacement],
    [math.sin(rot_1), math.cos(rot_1), 0, 0],
    [-math.cos(rot_1)*math.sin(rot_2), math.sin(rot_1)*math.sin(rot_2), math.cos(rot_2), 0],
    [0, 0, 0, 1]
])

T_inv = np.array([
    [math.cos(rot_1)*math.cos(rot_2), math.sin(rot_1), -math.cos(rot_1)*math.sin(rot_2), -displacement*math.cos(rot_1)*math.cos(rot_2)],
    [-math.cos(rot_2)*math.sin(rot_1), math.cos(rot_1), math.sin(rot_1)*math.sin(rot_2), displacement*math.cos(rot_2)*math.sin(rot_1)],
    [math.sin(rot_2), 0, math.cos(rot_2), -displacement*math.sin(rot_2)],
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
# print(np.dot(T, forward_test))

# print("Backwards Test\n")
# print(np.dot(T_inv, backwards_test))

# ----------------------------------------------------------------------------------------------------------------------------------
# Section Assignment
boundingLength = max(panel.length, panel.width, panel.h_longitudinal_web)

# Needs to be completed after the boolean operations to ensure that the section assignment is applied to the newly created sections
assign_section_bounds(model, 'plate', 't-' + str(panel.t_panel), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])
assign_section_bounds(model, 'web', 't-' + str(panel.t_longitudinal_web), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])
assign_section_bounds(model, 'flange', 't-' + str(panel.t_longitudinal_flange), [-boundingLength, boundingLength, -boundingLength, boundingLength, -boundingLength, boundingLength])

# ----------------------------------------------------------------------------------------------------------------------------------
# Find the node closest to the centroid of the face
assembly.regenerate()
t_offset = 0.01

A_panel = panel.width * panel.t_panel
A_web = panel.h_longitudinal_web * panel.t_longitudinal_web * panel.num_longitudinal
A_flange = panel.w_longitudinal_flange * panel.t_longitudinal_flange * panel.num_longitudinal

y_panel = panel.t_panel / 2
y_web = panel.t_panel + (panel.h_longitudinal_web) / 2
y_flange = (panel.t_panel / 2) + panel.h_longitudinal_web + (panel.t_longitudinal_flange / 2)

# If the TIE constraints defined between the edges and the surfaces are not set with thickness=ON, you need to consider the panels to each start at the half-thickness of the surface
centroid = (A_panel * y_panel + A_web * y_web + A_flange * y_flange) / (A_panel + A_web + A_flange)

# Apply load to the left most edge of the panel
web_step = panel.width / (panel.num_longitudinal + 1)
current_step = web_step

# Capture the nodal points of the part, not the assembly instance
instance = model.parts['web-mesh']
nodes = instance.nodes

# Define left and right target points at web locations in the assembly reference frame
target_point_left = np.array([
    [-float(panel.length / 2)],
    [-(float(panel.width) / 2) + web_step],
    [centroid]
])
target_point_right = np.array([
    [float(panel.length / 2)],
    [-(float(panel.width) / 2) + web_step],
    [centroid]
])

bottom_row = np.array([[1.0]])

# Homogenous numpy arrays
target_point_left = np.vstack([target_point_left, bottom_row])
target_point_right = np.vstack([target_point_right, bottom_row])

# Transform the points into the web reference frame
target_point_left = np.dot(T_inv, target_point_left)
target_point_right = np.dot(T_inv, target_point_right)

target_point_left = tuple(target_point_left.flatten()[:3])
target_point_right = tuple(target_point_right.flatten()[:3])

# Must reference the points in the part reference frame, not the assembly
move_closest_nodes_to_neutral_axis(model.parts['web-mesh'], target_point_left, axis_dof = 1, free_dof = 2)
move_closest_nodes_to_neutral_axis(model.parts['web-mesh'], target_point_right, axis_dof = 1, free_dof = 2)













# Find closest nodes to target points
closest_node_left = find_closest_node(nodes, target_point_left)
closest_node_right = find_closest_node(nodes, target_point_right)

boundary_regions = [
    [
        target_point_left[0] - t_offset, target_point_left[0] + t_offset,
        -float(panel.width)/2, float(panel.width)/2,
        closest_node_left.coordinates[2] - t_offset, closest_node_left.coordinates[2] + t_offset
    ],

    [
    target_point_right[0] - t_offset, target_point_right[0] + t_offset,
    -float(panel.width)/2, float(panel.width)/2,
    closest_node_right.coordinates[2] - t_offset, closest_node_right.coordinates[2] + t_offset
    ]
]

# Modify the mesh to allow for the correct definition of the centroidal nodes
plate = model.parts['plate-mesh']
web = model.parts['web-mesh']
flange = model.parts['flange-mesh']

web.SetFromNodeLabels(name='test-set', nodeLabels=(closest_node_left.label,))
web.editNode(
    nodes=web.sets['test-set'].nodes,
    coordinates=(target_point_left,)
)

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
assembly.Instance(dependent=ON, name='plate-1', part=model.parts['plate-mesh'])
assembly.Instance(dependent=ON, name='web-1', part=model.parts['web-mesh'])
assembly.Instance(dependent=ON, name='flange-1', part=model.parts['flange-mesh'])

assembly.rotate(instanceList=('plate-1',), axisPoint=(0,0,0), axisDirection=(0,0,1), angle=90)
assembly.translate(instanceList=['plate-1'], vector=(float(panel.length)/2,0.0, 0.0))

# Position the flange and web properly
assembly.rotate(instanceList=['web-1','flange-1'], axisPoint= (0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_1))
assembly.rotate(instanceList=['web-1','flange-1'], axisPoint= (0,0,0), axisDirection=(0,1,0), angle=math.degrees(rot_2))
assembly.translate(instanceList=['web-1','flange-1'], vector=(displacement, 0.0, 0.0))

'''

The below sections are a work-in-progress



'''




























# ----------------------------------------------------------------------------------------------------------------------------------
# Creation of surface and edge indexing to allow constraint creation parametrically!
from itertools import chain

# === Constants ===
t_offset = 0.01
half_width = panel.width / 2
half_length = panel.length / 2
h_long = panel.h_longitudinal_web

# === Surface Names ===
surface_list = ['plate-1', 'flange-1']

# === Edge Names ===
edge_list = ['web-1'] * 2

# === Bounding Boxes ===
bounds = [
    [-half_length, half_length, -half_width, half_width, -t_offset, t_offset],                            # plate
    [-half_length, half_length, -half_width, half_width, h_long - t_offset, h_long + t_offset],           # flange
]

# === Index Maps ===
surface_index = [0, 1]
edge_index = [0, 1]
constraint_index = edge_index

# === Create surface sets ===
for i, surface_name in enumerate(surface_list):
    create_surface_bounds(assembly, "surf-{}".format(i), surface_name, bounds[surface_index[i]])

# === Create edge sets ===
for i, edge_name in enumerate(edge_list):
    create_edge_set_bounds(assembly, "edge-{}".format(i), edge_name, bounds[edge_index[i]])

# === Create tie constraints ===
for i in range(len(constraint_index)):
    model.Tie(
        name="Constraint{}".format(i),
        main=assembly.surfaces["surf-{}".format(constraint_index[i])],
        secondary=assembly.sets["edge-{}".format(i)],
        adjust=ON,
        positionToleranceMethod=COMPUTED,
        tieRotations=ON,
        thickness=ON
    )

# --------------------------------------------------------------------------------------------------------------------------------------------
# Boundary Conditions

t_offset = 0.01
boundary_regions = [
    # X-Aligned BCs
    [float(panel.length)/2 - t_offset, float(panel.length)/2 + t_offset, -float(panel.width)/2, float(panel.width)/2, -t_offset, panel.h_longitudinal_web + t_offset],
    [-float(panel.length)/2 - t_offset, -float(panel.length)/2 + t_offset, -float(panel.width)/2, float(panel.width)/2, -t_offset, panel.h_longitudinal_web + t_offset],
        
    # Y-Aligned BCs
    [-float(panel.length)/2, float(panel.length)/2, float(panel.width)/2 - t_offset, float(panel.width)/2 + t_offset, -t_offset, panel.h_longitudinal_web + t_offset],
    [-float(panel.length)/2, float(panel.length)/2, -float(panel.width)/2 - t_offset, -float(panel.width)/2 + t_offset, -t_offset, panel.h_longitudinal_web + t_offset]
    ]

boundary_list = [
    ['plate-1', 'web-1', 'flange-1']
    ]

# Rigid Boundary Conditions
e1 = create_edge_set_bounds(assembly, 'TempSet-1', boundary_list[0][0], boundary_regions[0])
e2 = create_edge_set_bounds(assembly, 'TempSet-2', boundary_list[0][1], boundary_regions[0])
e3 = create_edge_set_bounds(assembly, 'TempSet-3', boundary_list[0][2], boundary_regions[0])

tempSet = e1
#tempSet = e1 + e2 + e3
assembly.Set(edges=tempSet, name='End')

# Sliding Boundary Conditions
e1 = create_edge_set_bounds(assembly, 'TempSet-4', boundary_list[0][0], boundary_regions[2])
e2 = create_edge_set_bounds(assembly, 'TempSet-5', boundary_list[0][0], boundary_regions[3])

tempSet = e1 + e2
assembly.Set(edges=tempSet, name='Sides')

# Sliding Boundary Conditions
e1 = create_edge_set_bounds(assembly, 'TempSet-6', boundary_list[0][0], boundary_regions[1])
e2 = create_edge_set_bounds(assembly, 'TempSet-7', boundary_list[0][1], boundary_regions[1])
e3 = create_edge_set_bounds(assembly, 'TempSet-8', boundary_list[0][2], boundary_regions[1])

tempSet = e1
assembly.Set(edges=tempSet, name='Free-Sides')

# ----------------------------------------------------------------------------------------------------------------------------------
# Create displacement boundary conditions and apply a pressure to the whole assembly

assembly.regenerate()

A_panel = panel.width * panel.t_panel
A_web = panel.h_longitudinal_web * panel.t_longitudinal_web * panel.num_longitudinal
A_flange = panel.w_longitudinal_flange * panel.t_longitudinal_flange * panel.num_longitudinal

y_panel = panel.t_panel / 2
y_web = panel.t_panel + (panel.h_longitudinal_web) / 2
y_flange = (panel.t_panel / 2) + panel.h_longitudinal_web + (panel.t_longitudinal_flange / 2)

# If the TIE constraints defined between the edges and the surfaces are not set with thickness=ON, you need to consider the panels to each start at the half-thickness of the surface
centroid = (A_panel * y_panel + A_web * y_web + A_flange * y_flange) / (A_panel + A_web + A_flange)

# Apply load to the left most edge of the panel
web_step = panel.width / (panel.num_longitudinal + 1)
current_step = web_step

instance = assembly.instances['web-1']
nodes = instance.nodes

# Define left and right target points at web locations
target_point_left = (-float(panel.length / 2), -(float(panel.width) / 2) + web_step, centroid)
target_point_right = (float(panel.length / 2), -(float(panel.width) / 2) + web_step, centroid)

# Find closest nodes to target points
closest_node_left = find_closest_node(nodes, target_point_left)
closest_node_right = find_closest_node(nodes, target_point_right)

boundary_regions = [
    [
        target_point_left[0] - t_offset, target_point_left[0] + t_offset,
        -float(panel.width)/2, float(panel.width)/2,
        closest_node_left.coordinates[2] - t_offset, closest_node_left.coordinates[2] + t_offset
    ],

    [
    target_point_right[0] - t_offset, target_point_right[0] + t_offset,
    -float(panel.width)/2, float(panel.width)/2,
    closest_node_right.coordinates[2] - t_offset, closest_node_right.coordinates[2] + t_offset
    ]
]

# Create node sets using bounding boxes
create_node_set_bounds(assembly, 'Set-1', 'web-1', boundary_regions[0])
create_node_set_bounds(assembly, 'Set-2', 'web-1', boundary_regions[1])

node_labels_left = [node.label for node in assembly.sets['Set-1'].nodes]
node_labels_right = [node.label for node in assembly.sets['Set-2'].nodes]

target_nodes_left = instance.nodes.sequenceFromLabels(labels=tuple(node_labels_left))
target_nodes_right = instance.nodes.sequenceFromLabels(labels=tuple(node_labels_right))

face_region_left = regionToolset.Region(nodes=target_nodes_left)
face_region_right = regionToolset.Region(nodes=target_nodes_right)

if panel.num_longitudinal > 1:
    ref_point_left = assembly.ReferencePoint(point=closest_node_left.coordinates)
    ref_point_obj_left = assembly.referencePoints[ref_point_left.id]
    ref_point_region_left = assembly.Set(referencePoints=(ref_point_obj_left,), name="Load-Point-Set-Left")

    ref_point_right = assembly.ReferencePoint(point=closest_node_right.coordinates)
    ref_point_obj_right = assembly.referencePoints[ref_point_right.id]
    ref_point_region_right = assembly.Set(referencePoints=(ref_point_obj_right,), name="Load-Point-Set-Right")

    # Left coupling
    model.Coupling(
        name='LoadCoupling',
        controlPoint=ref_point_region_left,
        surface=face_region_left,
        influenceRadius=WHOLE_SURFACE,
        couplingType=DISTRIBUTING,
        localCsys=None,
        u1=ON, u2=ON, u3=ON,
        ur1=OFF, ur2=OFF, ur3=OFF
    )

    # Right coupling
    model.Coupling(
        name='LoadCoupling_Two',
        controlPoint=ref_point_region_right,
        surface=face_region_right,
        influenceRadius=WHOLE_SURFACE,
        couplingType=DISTRIBUTING,
        localCsys=None,
        u1=ON, u2=ON, u3=ON,
        ur1=OFF, ur2=OFF, ur3=OFF
    )

# Side boundary conditions
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Fixed-BC', region=face_region_right, u1=0.0, u2=0.0, u3=0.0)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Fixed-Side-BC', region=assembly.sets['End'], u2=0.0, u3=0.0)

model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Side-BC', region=assembly.sets['Sides'], u3=0.0)

model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Free-BC', region=face_region_left, u2=0.0, u3=0.0)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Free-Side-BC', region=assembly.sets['Free-Sides'], u3=0.0)

# Load application
model.ConcentratedForce(
    name="Load",
    createStepName="Buckle-Step",
    region=face_region_left,
    distributionType=UNIFORM,
    cf1=float(panel.axial_force),
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
    model='Model-1',
    modelPrint=OFF,
    multiprocessingMode=DEFAULT,
    name=job_name,
    nodalOutputPrecision=SINGLE,
    numCpus=4,
    numGPUs=0,
    queue=None,
    resultsFormat=ODB,
    scratch='',
    type=ANALYSIS,
    userSubroutine='',
    waitHours=0,
    waitMinutes=0,
    numDomains=4
)

job.writeInput()

job.submit(consistencyChecking=OFF)
job.waitForCompletion()