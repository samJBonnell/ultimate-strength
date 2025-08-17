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
working_directory = r'Z:\\lase\\ultimate_strength'
input_directory = r'data\\input.jsonl'
output_directory = r'data\\output.jsonl'
os.chdir(working_directory)

# !!! Set correct job name
job_name = 'buckling_riks_panel'

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

def assign_section_bounds(part, section_name, bounds, target_type="faces"):
    """
    Assign a section to all faces or elements within a bounding box.

    Parameters
    ----------
    part : Part
        The Abaqus part object.
    section_name : str
        Name of the section to assign.
    bounds : tuple
        Bounding box in the form (x_min, x_max, y_min, y_max, z_min, z_max).
    target_type : str, optional
        Either 'faces' or 'elements'. Default is 'faces'.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Select the target collection
    if target_type.lower() == "faces":
        targets = part.faces.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
    elif target_type.lower() == "elements":
        targets = part.elements.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
    else:
        raise ValueError("[assign_section_bounds] target_type must be 'faces' or 'elements'.")

    if not targets:
        raise ValueError("[assign_section_bounds] No {} found for section assignment in bounds {} on part '{}'.".format(
            target_type, bounds, part.name))

    # Create set and assign section
    part.Set(name='sectionAssignment', **{target_type: targets})
    part.SectionAssignment(
        region=part.sets['sectionAssignment'],
        sectionName=section_name,
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )

    print("[assign_section_bounds] Assigned section '{}' to {} {}(s) on part '{}'.".format(
        section_name, len(targets), target_type[:-1], part.name))

def create_node_set(container, set_name, instance_name=None, bounds=None):
    """
    Create a node set using a bounding box and return the nodes and their labels.
    
    Works for:
    - Assembly (requires instance_name)
    - Part (instance_name is ignored)
    
    Parameters
    ----------
    container : Assembly or Part
        The object on which to create the set.
    set_name : str
        Name of the node set to create.
    instance_name : str, optional
        Required if `container` is an Assembly.
    bounds : tuple
        (x_min, x_max, y_min, y_max, z_min, z_max)
    
    Returns
    -------
    tuple
        (nodes, labels) where:
            nodes  = NodeArray object from Abaqus
            labels = list of node labels (ints)
    """
    if bounds is None:
        raise ValueError("[create_node_set] Bounds must be specified.")

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Assembly case
    if hasattr(container, "instances") and instance_name is not None:
        target_nodes = container.instances[instance_name].nodes.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
        if not target_nodes:
            raise ValueError("[create_node_set] No nodes found for set '{}' in bounds {} on instance '{}'.".format(
                set_name, bounds, instance_name))
        container.Set(name=set_name, nodes=target_nodes)
        print("[create_node_set] Created node set '{}' with {} node(s) in assembly on instance '{}'.".format(
            set_name, len(target_nodes), instance_name))

    # Part case
    elif hasattr(container, "nodes"):
        target_nodes = container.nodes.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
        if not target_nodes:
            raise ValueError("[create_node_set] No nodes found for set '{}' in bounds {} in part.".format(
                set_name, bounds))
        container.Set(name=set_name, nodes=target_nodes)
        print("[create_node_set] Created node set '{}' with {} node(s) in part.".format(
            set_name, len(target_nodes)))

    else:
        raise TypeError("[create_node_set] 'container' must be an Assembly or Part object.")

    # Extract labels from the nodes
    labels = [node.label for node in target_nodes]

    return target_nodes, labels

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
    """
    Given a list of nodes and a point in R3, the function will return the closest
    node based on Euclidean distance.

    Parameters
    ----------
    nodes : Abaqus Node list
    target_point : tuple containing location information (x, y, z)

    Returns
    -------
    Closest Abaqus Node item to target_point
    """
    for node in nodes:
        x, y, z = node.coordinates
        dx = x - target_point[0]
        dy = y - target_point[1]
        dz = z - target_point[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < min_dist:
            min_dist = dist
            closest_node = node

    return closest_node, closest_node.label

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

def move_closest_nodes_to_axis(part, target_point, axis_dof = 1, free_dof = 2):
    """Move the closest nodes along the axis_dof direction to target_point along the free_dof direction"""
    reference_point, _ = find_closest_node(part.nodes, target_point)

    # Capture all of the points on the part that lie along the line of action of the dof
    capture_offset = 0.001
    max_bound = 1e5 # A large number that will never be reached by the bounds of the part

    # Capture nodes along the neutral axis defined by the dof and the reference_point
    nodes = get_nodes_along_axis(part.nodes, reference_point.coordinates, axis_dof, max_bound, capture_offset)

    for node in nodes:
        # Find the coordinates of the point and presribe the neutral axis location
        temp_coords = node.coordinates

        coordinates = list(node.coordinates)
        coordinates[free_dof - 1] = target_point[free_dof - 1]

        # Move mesh to match this neutral axis location
        part.editNode(nodes=(node,), coordinates=(tuple(coordinates),))
        print("[move_closest_nodes_to_axis] Moved node '{}' from location {} to '{}'.".format(node.label, temp_coords, node.coordinates))

    labels = [node.label for node in nodes]
    return nodes, labels

def set_local_element_thickness(part, target_point, axis_dof, section_name='local-thickness', depth_of_search=1, set_name='temp'):
    """Assign a section to elements connected along an axis, expanding out by edge-sharing neighbours."""
    reference_point, _ = find_closest_node(part.nodes, target_point)

    capture_offset = 0.001
    max_bound = 1e5

    nodes = get_nodes_along_axis(part.nodes, reference_point.coordinates, axis_dof, max_bound, capture_offset)

    def edges_of_element(elem):
        node_labels = [node.label for node in elem.getNodes()]
        edges = []
        n = len(node_labels)
        for i in range(n):
            n1 = node_labels[i]
            n2 = node_labels[(i + 1) % n]
            edges.append(tuple(sorted((n1, n2))))
        return edges

    connected_labels = set()
    for node in nodes:
        connected_labels.update(e.label for e in node.getElements())

    edge_map = {}
    for elem in part.elements:
        for edge in edges_of_element(elem):
            edge_map.setdefault(edge, set()).add(elem.label)

    selected_labels = set(connected_labels)
    frontier = set(connected_labels)

    for _ in range(depth_of_search):
        next_frontier = set()
        for elem_label in frontier:
            elem = part.elements.getFromLabel(elem_label)
            for edge in edges_of_element(elem):
                for neigh in edge_map.get(edge, ()):
                    if neigh not in selected_labels:
                        selected_labels.add(neigh)
                        next_frontier.add(neigh)
        frontier = next_frontier

    selected_elements = part.elements.sequenceFromLabels(sorted(selected_labels))

    elem_set = part.Set(name=set_name, elements=selected_elements)
    part.SectionAssignment(
        region=elem_set,
        sectionName=section_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION
    )

    print("Assigned section '{}' to {} elements.".format(section_name, len(selected_elements)))

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

web_spacing = float(panel.width) / (panel.num_longitudinal + 1)
half_width = float(panel.width) / 2
web_locations = np.arange(-half_width + web_spacing, half_width, web_spacing)

# Plate sketch & part
sketch = model.ConstrainedSketch(name='geometry', sheetSize=100.0)
sketch.rectangle(point1=(-panel.width/2, 0), point2=(panel.width/2, panel.length))

p = model.Part(name='plate', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p.BaseShell(sketch=sketch)
del model.sketches['geometry']

# ----------------------------------------------------------------------------------------------------------------------------------
# Longitudinal Stiffener sketch & part

model.ConstrainedSketch(name='geometry', sheetSize=200.0)
for index, web_location in enumerate(web_locations):
    model.sketches['geometry'].Line(point1=(float(web_location), 0.0), point2=(float(web_location), panel.h_longitudinal_web))
    model.sketches['geometry'].VerticalConstraint(entity=model.sketches['geometry'].geometry[2 + index], addUndoState=False)

model.parts['web'].BaseShellExtrude(depth=panel.length, sketch=model.sketches['geometry'])
del model.sketches['geometry']

# ----------------------------------------------------------------------------------------------------------------------------------
# Longitudinal Flange sketch & part
half_flange_width = float(panel.w_longitudinal_flange)/2
model.ConstrainedSketch(name='geometry', sheetSize=200.0)

for index, web_location in enumerate(web_locations):
    model.sketches['geometry'].Line(point1=(float(web_location) - half_flange_width, panel.h_longitudinal_web), point2=(float(web_location) + half_flange_width, panel.h_longitudinal_web))
    model.sketches['geometry'].HorizontalConstraint(entity=model.sketches['geometry'].geometry[2 + index], addUndoState=False)

model.parts['flange'].BaseShellExtrude(depth=panel.length, sketch=model.sketches['geometry'])
del model.sketches['geometry']

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
# Meshing the Part
mesh(model.parts['plate'], panel.mesh_plate)
mesh(model.parts['web'], panel.mesh_longitudinal_web)
mesh(model.parts['flange'], panel.mesh_longitudinal_flange)

model.parts['plate'].PartFromMesh(name='plate-mesh', copySets=FALSE)
model.parts['web'].PartFromMesh(name='web-mesh', copySets=FALSE)
model.parts['flange'].PartFromMesh(name='flange-mesh', copySets=FALSE)

plate = model.parts['plate-mesh']
web = model.parts['web-mesh']
flange = model.parts['flange-mesh']

# Instance Transformations
# Position the plate in the correct orientation

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

T_plate = np.array([
    [math.cos(rot_plate), -math.sin(rot_plate), 0, plate_displacement],
    [math.sin(rot_plate), math.cos(rot_plate),  0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T_inv_plate = np.array([
    [math.cos(rot_plate), math.sin(rot_plate), 0, -plate_displacement * math.cos(rot_plate)],
    [-math.sin(rot_plate), math.cos(rot_plate), 0, plate_displacement * math.sin(rot_plate)],
    [0, 0, 1, 0],
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
max_domain = max(panel.length, panel.width, panel.h_longitudinal_web)

assign_section_bounds(part = plate, section_name = "t-{}".format(panel.t_panel), bounds = [-max_domain, max_domain, -max_domain, max_domain, -max_domain, max_domain], target_type = 'elements')
assign_section_bounds(part = web, section_name = "t-{}".format(panel.t_longitudinal_web), bounds = [-max_domain, max_domain, -max_domain, max_domain, -max_domain, max_domain], target_type = 'elements')
assign_section_bounds(part = flange, section_name = "t-{}".format(panel.t_longitudinal_flange), bounds = [-max_domain, max_domain, -max_domain, max_domain, -max_domain, max_domain], target_type = 'elements')

# ----------------------------------------------------------------------------------------------------------------------------------
# Find the node closest to the centroid of the face
assembly.regenerate()
capture_offset = 0.001

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
target_point_left = np.dot(T_inv_web, target_point_left)
target_point_right = np.dot(T_inv_web, target_point_right)

target_point_left = tuple(target_point_left.flatten()[:3])
target_point_right = tuple(target_point_right.flatten()[:3])

# Modify the nodes along the neutral axis of the panel to line up properly
# Must reference the points in the part reference frame, not the assembly
_, centroid_node_labels_left = move_closest_nodes_to_axis(part=web, target_point=target_point_left, axis_dof = 1, free_dof = 2)
_, centroid_node_labels_right = move_closest_nodes_to_axis(part=web, target_point=target_point_right, axis_dof = 1, free_dof = 2)

# Change the local thickness of the elements to prevent local failure due to large input forces
set_local_element_thickness(part=web, target_point=target_point_left, axis_dof = 1, depth_of_search = 2, set_name='left-thickness-region')
set_local_element_thickness(part=web, target_point=target_point_right, axis_dof = 1, depth_of_search = 2, set_name='right_thickness_region')

# Reference the coordinates in the global geometry and use the inverse matrix to map them into the plate geometry
model.ConstrainedSketch(name='plate-mesh-edit', sheetSize=200.0)
for index, web_location in enumerate(web_locations):
    web_point = np.array([[0.0], [float(web_location)], [0.0]]) # Location of each of the webs that intersect the plate
    # Transform back into the plate reference frame
    web_point = np.vstack([web_point, bottom_row])
    web_point = np.dot(T_inv_plate, web_point)
    web_point = tuple(web_point.flatten()[:3])

    # Align the plate mesh to the axis along the x-axis to this location
    _, _ = move_closest_nodes_to_axis(model.parts['plate-mesh'], web_point, axis_dof = 2, free_dof = 1)

# ----------------------------------------------------------------------------------------------------------------------------------
# Instance Creation
assembly.Instance(dependent=ON, name='plate', part=model.parts['plate-mesh'])
assembly.Instance(dependent=ON, name='web', part=model.parts['web-mesh'])
assembly.Instance(dependent=ON, name='flange', part=model.parts['flange-mesh'])

assembly.rotate(instanceList=('plate',), axisPoint=(0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_plate))
assembly.translate(instanceList=['plate'], vector=(plate_displacement,0.0, 0.0))

# Position the flange and web properly
assembly.rotate(instanceList=['web','flange'], axisPoint= (0,0,0), axisDirection=(0,0,1), angle=math.degrees(rot_1))
assembly.rotate(instanceList=['web','flange'], axisPoint= (0,0,0), axisDirection=(0,1,0), angle=math.degrees(rot_2))
assembly.translate(instanceList=['web','flange'], vector=(web_displacement, 0.0, 0.0))

# ----------------------------------------------------------------------------------------------------------------------------------
# Collect the sets of nodes that fall along each point within web_locations and 

plate_labels = []
web_labels = []
for index, web_location in enumerate(web_locations):
    bounds = [-max_domain, max_domain, web_location - capture_offset, web_location + capture_offset, -capture_offset, capture_offset]

    # Find the plate nodes that lie under the contact area of the web
    _, new_labels = create_node_set(assembly, 'temp-plate-node-set-{}'.format(index), instance_name = 'plate', bounds = bounds)
    plate_labels.extend(new_labels)

    # Find the web nodes that lie above the contact area on the plate
    _, new_labels = create_node_set(assembly, 'temp-web-node-set-{}'.format(index), instance_name = 'web', bounds = bounds)
    web_labels.extend(new_labels)

    model.Tie(
        name="Constraint{}".format(index),
        main=assembly.sets['temp-plate-node-set-{}'.format(index)],
        secondary=assembly.sets['temp-web-node-set-{}'.format(index)],
        adjust=ON,
        positionToleranceMethod=COMPUTED,
        tieRotations=ON,
        thickness=ON
    )

web_labels = []
flange_labels = []
for index, web_location in enumerate(web_locations):
    bounds = [-max_domain, max_domain, web_location - capture_offset, web_location + capture_offset, panel.h_longitudinal_web - capture_offset, panel.h_longitudinal_web + capture_offset]
    print(index)

    # Find the plate nodes that lie under the contact area of the web
    _, new_labels = create_node_set(assembly, 'temp-web-upper-node-set-{}'.format(index), instance_name = 'web', bounds = bounds)
    web_labels.extend(new_labels)

    # Find the web nodes that lie above the contact area on the plate
    _, new_labels = create_node_set(assembly, 'temp-flange-node-set-{}'.format(index), instance_name = 'flange', bounds = bounds)
    flange_labels.extend(new_labels)

    model.Tie(
        name="Constraint{}".format(index + len(web_locations)),
        main=assembly.sets['temp-web-upper-node-set-{}'.format(index)],
        secondary=assembly.sets['temp-flange-node-set-{}'.format(index)],
        adjust=ON,
        positionToleranceMethod=COMPUTED,
        tieRotations=ON,
        thickness=ON
    )

# ----------------------------------------------------------------------------------------------------------------------------------
# Rigid Body Motion Constraint
# Define a constraint point to limit the x-2 displacement of the panel to prevent RBM
constraint_point = np.array([[0.0], [0.0], [0.0]])
# Transform into the plate reference frame
constraint_point = np.vstack([constraint_point, bottom_row])
constraint_point = np.dot(T_inv_plate, constraint_point)
constraint_point = tuple(constraint_point.flatten()[:3])

_, label = find_closest_node(plate.nodes, constraint_point)
middle_node = assembly.instances['plate'].nodes.sequenceFromLabels((label,))
constraint_set = assembly.Set(name='middle-bc', nodes=(middle_node,))

# Constrain in the x-2 direction, and allow all other motion
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Middle-BC', region=assembly.sets['middle-bc'], u2=0.0)

# Plate-edge BC (zero motion in x3 direction)
boundary_regions = [
    # X-Aligned BCs
    [float(panel.length)/2 - capture_offset, float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, panel.h_longitudinal_web + capture_offset],
    [-float(panel.length)/2 - capture_offset, -float(panel.length)/2 + capture_offset, -float(panel.width)/2, float(panel.width)/2, -capture_offset, panel.h_longitudinal_web + capture_offset],
        
    # Y-Aligned BCs
    [-float(panel.length)/2, float(panel.length)/2, float(panel.width)/2 - capture_offset, float(panel.width)/2 + capture_offset, -capture_offset, panel.h_longitudinal_web + capture_offset],
    [-float(panel.length)/2, float(panel.length)/2, -float(panel.width)/2 - capture_offset, -float(panel.width)/2 + capture_offset, -capture_offset, panel.h_longitudinal_web + capture_offset]
    ]

# Capture all of the edges of the plate
labels = []
for index, region in enumerate(boundary_regions):
    _, new_labels = create_node_set(assembly, 'simply-supported-edge-{}'.format(index), 'plate', region)
    labels.extend(new_labels)

plate_edge_BC_nodes = assembly.instances['plate'].nodes.sequenceFromLabels(labels)
plate_edge_set = assembly.Set(name='plate-edge', nodes=plate_edge_BC_nodes)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Edge-BC', region=plate_edge_set, u3=0.0)

# Link together all of the rings of nodes for a given stiffener
created_sets = set()
dof = 2

for index, web_location in enumerate(web_locations):
    labels = []  # Reset labels for each ring

    bounds_1 = [
        -(panel.length / 2) - capture_offset,
        -(panel.length / 2) + capture_offset,
        web_location - capture_offset,
        web_location + capture_offset,
        capture_offset,
        panel.h_longitudinal_web - capture_offset,
    ]
    bounds_2 = [
        (panel.length / 2) - capture_offset,
        (panel.length / 2) + capture_offset,
        web_location - capture_offset,
        web_location + capture_offset,
        capture_offset,
        panel.h_longitudinal_web - capture_offset,
    ]

    # Create node sets & get labels for each side of the ring
    _, new_label_1 = create_node_set(assembly, 'periodic-{}-1'.format(index), 'web', bounds=bounds_1)
    labels.extend(new_label_1)
    _, new_label_2 = create_node_set(assembly, 'periodic-{}-2'.format(index), 'web', bounds=bounds_2)
    labels.extend(new_label_2)

    labels.sort()

    # Sliding window of size 2 over labels
    window_size = 2
    for j in range(len(labels) - window_size + 1):
        lbl_1, lbl_2 = labels[j : j + window_size]

        # Create node sets if not already created
        if lbl_1 not in created_sets:
            created_sets.add(lbl_1)
            assembly.Set(
                name='periodic-node-{}-{}'.format(index, lbl_1),
                nodes=assembly.instances['web'].nodes.sequenceFromLabels((lbl_1,)),
            )
        if lbl_2 not in created_sets:
            created_sets.add(lbl_2)
            assembly.Set(
                name='periodic-node-{}-{}'.format(index, lbl_2),
                nodes=assembly.instances['web'].nodes.sequenceFromLabels((lbl_2,)),
            )

        # Create the equation linking dof=2 of both nodes
        model.Equation(
            name='Periodic-Equation-{}-{}'.format(index, j),
            terms=(
                (-1.0, 'periodic-node-{}-{}'.format(index, lbl_1), dof),
                (1.0, 'periodic-node-{}-{}'.format(index, lbl_2), dof),
            ),
        )

# Define the boundary conditions on the neutral axis of the fixed end of the panel
# We have, from before: centroid_node_labels_left, centroid_node_labels_right (fixed)
# Create boundary set for fixed side:

fixed_centroid_BC_nodes = assembly.instances['web'].nodes.sequenceFromLabels(centroid_node_labels_right)
fixed_centroid_BC = assembly.Set(name='Fixed-BC', nodes=fixed_centroid_BC_nodes)
model.DisplacementBC(amplitude=UNSET, createStepName='Initial', distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='Fixed-BC', region=fixed_centroid_BC, u1=0.0)

# Tie the end of the panel together as well
created_sets = set()
window_size = 2
# Constrain the x-displacement of the free end of the panel via Equations
dof = 1

for index in range(len(centroid_node_labels_left) - window_size + 1):
    lbl_1, lbl_2 = centroid_node_labels_left[index : index + window_size]
    # Create node sets if not already created
    if lbl_1 not in created_sets:
        created_sets.add(lbl_1)
        assembly.Set(
            name='free-node-{}'.format(lbl_1),
            nodes=assembly.instances['web'].nodes.sequenceFromLabels((lbl_1,)),
        )
    if lbl_2 not in created_sets:
        created_sets.add(lbl_2)
        assembly.Set(
            name='free-node-{}'.format(lbl_2),
            nodes=assembly.instances['web'].nodes.sequenceFromLabels((lbl_2,)),
        )
    # Create the equation linking dof=2 of both nodes
    model.Equation(
        name='Free-End-Equation-{}'.format(index),
        terms=(
            (-1.0, 'free-node-{}'.format(lbl_1), dof),
            (1.0, 'free-node-{}'.format(lbl_2), dof),
        ),
    )

# --------------------------------------------------------------------------------------------------------------------------------------------
# Load conditions!
# Load application

load_nodes = assembly.instances['web'].nodes.sequenceFromLabels((centroid_node_labels_left[0],))
load_set = assembly.Set(name='load_set', nodes=load_nodes)
load_region = regionToolset.Region(nodes=load_nodes)
# Create Step Object
model.StaticRiksStep(
    name='Riks-Step',
    previous='Initial',
    nlgeom=ON,
    initialArcInc=0.001,
    maxArcInc=0.5,
    maxNumInc=500,
    nodeOn=ON,
    region=load_region,
    dof=1,
    maximumDisplacement=0.5
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