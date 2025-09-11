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

# ----------------------------------------------------------------------------------------------------------------------------------

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("\n\nStart Time: {}\n\n".format(current_time))

# !!! Set correct working directory !!!
working_directory = r'C:\\Users\\sbonnell\\Desktop\\lase\\projects\\ultimate-strength\\eigen-buckling'
input_directory = r'data\\input.jsonl'
output_directory = r'data\\output.jsonl'
os.chdir(working_directory)

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
    
def assign_section_sets(part, section_name, set_name):
    # Assign section to each face set
    faces = part.sets[set_name].faces
    region = regionToolset.Region(faces=faces)
    part.SectionAssignment(
        region=region,
        sectionName=section_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION
    )

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

def mesh(part, edge_seed_map, elemShape=QUAD, technique=STRUCTURED, elemCode=S4R, elemLibrary=STANDARD, constraint=FINER):

    for set_name, mesh_size in edge_seed_map.items():
        edges = part.sets[set_name].edges
        part.seedEdgeBySize(edges=edges, size=mesh_size, constraint=constraint)

    part.setMeshControls(
        regions=part.faces[:],
        technique=technique,
        elemShape=elemShape
    )
    elemType1 = ElemType(elemCode=elemCode, elemLibrary=elemLibrary)
    part.setElementType(regions=(part.faces[:],), elemTypes=(elemType1,))
    part.generateMesh()

def mesh_from_faces(part, face_seed_map, elemShape=QUAD, technique=STRUCTURED,
                    elemCode=S4R, elemLibrary=STANDARD, constraint=FINER):
    """
    Mesh a part by seeding all edges of given face sets.

    Parameters
    ----------
    part : Part
        The Abaqus Part object to mesh.
    face_seed_map : dict
        Dictionary mapping face set names to desired mesh sizes, e.g.
        {'PlateFace': 0.01, 'WebFaces': 0.005, 'FlangeFaces': 0.005}
    elemShape : symbolic constant
        Element shape (QUAD or TRI).
    technique : symbolic constant
        Meshing technique (STRUCTURED, FREE, etc.).
    elemCode : symbolic constant
        Element type (S4R, S4, etc.).
    elemLibrary : symbolic constant
        Element library (STANDARD, EXPLICIT).
    constraint : symbolic constant
        Edge seeding constraint (FINER, MEDIUM, COARSER).
    """

    part = model.parts['plate']

    # Seed edges based on faces
    for face_set_name, mesh_size in face_seed_map.items():
        edges_to_seed = []

        # Collect edges from all faces
        for f in part.sets[face_set_name].faces:
            for e in f.getEdges():   # e may be an int or Edge object
                # Convert integer IDs to Edge objects if needed
                if isinstance(e, int):
                    e = part.edges[e]
                edges_to_seed.append(e)

        if edges_to_seed:
            # Remove duplicates using Edge objects directly
            edges_to_seed = list({id(e): e for e in edges_to_seed}.values())

            # Apply seeding
            part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, constraint=constraint)

    # Apply mesh controls to all faces
    part.setMeshControls(regions=part.faces[:], technique=technique, elemShape=elemShape)

    # Assign element type
    elem_type = ElemType(elemCode=elemCode, elemLibrary=elemLibrary)
    part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type,))

    # Generate mesh
    part.generateMesh()



def write_trial_ndjson(output, path="results.jsonl"):
    with open(path, "a") as f:
        json_line = json.dumps(clean_json(output))
        f.write(json_line + "\n")

def write_trial_ndjson_gz(output, path="results.jsonl.gz"):
    with gzip.open(path, "ab") as f:
        json_line = json.dumps(output.to_dict()) + "\n"
        f.write(json_line.encode("utf-8"))

def find_closest_node(container, reference_point, instance_name=None, min_dist=1e20):
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

    # Extract nodes
    if hasattr(container, "instances") and instance_name is not None:
        nodes = container.instances[instance_name].nodes
    elif hasattr(container, "nodes"):
        nodes = container.nodes
    else:
        raise TypeError("[find_closest_node] container must be an Assembly or Part object")

    for node in nodes:
        x, y, z = node.coordinates
        dx = x - reference_point[0]
        dy = y - reference_point[1]
        dz = z - reference_point[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < min_dist:
            min_dist = dist
            closest_node = node

    return closest_node, closest_node.label

def get_nodes_along_axis(container, reference_point, dof, instance_name=None, capture_offset=0.001, max_bound=1e5):
    """
    Return nodes along a specified axis within a capture offset around a reference point.
    
    Works for:
    - Assembly (requires instance_name)
    - Part (instance_name is ignored)
    
    Parameters
    ----------
    container : Assembly or Part
        Object to search nodes in.
    reference_point : tuple or list of 3 floats
        (x, y, z) coordinates around which to search.
    dof : int
        Axis along which the nodes can vary (1=x, 2=y, 3=z)
    capture_offset : float
        Small tolerance around the other two coordinates.
    max_bound : float
        Maximum distance to search along the dof axis.
    instance_name : str, optional
        Required if container is an Assembly.
    
    Returns
    -------
    tuple
        (nodes, labels) where:
            nodes  = NodeArray from Abaqus
            labels = list of node labels
    """
    if len(reference_point) != 3:
        raise ValueError("reference_point must be a 3-element tuple or list")

    # Create bounding box
    lower = [reference_point[i] - capture_offset for i in range(3)]
    upper = [reference_point[i] + capture_offset for i in range(3)]

    lower = [float(coord) for coord in lower]
    upper = [float(coord) for coord in upper]

    # Expand the dof axis to max bounds
    idx = dof - 1
    lower[idx] = -max_bound
    upper[idx] = max_bound

    # Extract nodes
    if hasattr(container, "instances") and instance_name is not None:
        nodes = container.instances[instance_name].nodes.getByBoundingBox(
            xMin=lower[0], xMax=upper[0],
            yMin=lower[1], yMax=upper[1],
            zMin=lower[2], zMax=upper[2]
        )
        if not nodes:
            raise ValueError("[get_nodes_along_axis] No nodes found in bounds {} on instance '{}'.".format((lower, upper), instance_name))
    elif hasattr(container, "nodes"):
        nodes = container.nodes.getByBoundingBox(
            xMin=lower[0], xMax=upper[0],
            yMin=lower[1], yMax=upper[1],
            zMin=lower[2], zMax=upper[2]
        )
        if not nodes:
            raise ValueError("[get_nodes_along_axis] No nodes found in bounds {} in part.".format((lower, upper)))
    else:
        raise TypeError("[get_nodes_along_axis] container must be an Assembly or Part object")

    labels = [node.label for node in nodes]
    return nodes, labels


def move_closest_nodes_to_axis(part, target_point, axis_dof = 1, free_dof = 2):
    """Move the closest nodes along the axis_dof direction to target_point along the free_dof direction"""
    reference_point, _ = find_closest_node(part, target_point)

    # Capture all of the points on the part that lie along the line of action of the dof
    capture_offset = 0.001
    max_bound = 1e5 # A large number that will never be reached by the bounds of the part

    # Capture nodes along the neutral axis defined by the dof and the reference_point
    nodes, _ = get_nodes_along_axis(part, reference_point.coordinates, axis_dof, max_bound, capture_offset)

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

def move_closest_node_to_point(part, target_point, free_dof=[2], exclusion_label=-1, tol=1e-8):
    """
    Given a point in R3, move the closest node along all listed
    degrees of freedom within free_dof to target_point.

    Parameters
    ----------
    part : Abaqus Part
    target_point : tuple containing location information (x, y, z)
    free_dof : list of int
        Axes along which the node point is to align itself with target_point (1=x, 2=y, 3=z).
    exclusion_label : int
        Node label to exclude from moving.
    tol : float
        Numerical tolerance for checking if already aligned.

    Returns
    -------
    node : Abaqus Node object
    label : int
        Node label
    """

    # Find the closest node to the target_point
    node, _ = find_closest_node(part, target_point)
    label = node.label

    if label == exclusion_label:
        return node, label

    temp_coords = list(node.coordinates)
    coordinates = list(node.coordinates)

    # Check if already aligned along the free DOFs
    already_aligned = all(
        abs(coordinates[dof - 1] - target_point[dof - 1]) < tol for dof in free_dof)
    if already_aligned:
        return node, label

    # Otherwise, move node along free DOFs
    for dof in free_dof:
        coordinates[dof - 1] = target_point[dof - 1]

    part.editNode(nodes=(node,), coordinates=(tuple(coordinates),))
    print("[move_closest_node_to_point] Moved node '{}' from {} to {}.".format(
        node.label, tuple(temp_coords), tuple(coordinates)
    ))

    return node, label

def set_local_element_thickness(part, target_point, axis_dof, section_name='local-thickness', depth_of_search=1, set_name='temp'):
    """Assign a section to elements connected along an axis, expanding out by edge-sharing neighbours."""
    reference_point, _ = find_closest_node(part, target_point)

    capture_offset = 0.001
    max_bound = 1e5

    nodes, _ = get_nodes_along_axis(part, reference_point.coordinates, axis_dof, max_bound, capture_offset)

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

def homogenous_transform(transformation, point=None):
    """
    Given a point in R3 and a 4x4 homogeneous transformation matrix,
    return the transformed point.

    Parameters
    ----------
    transformation : (4,4) numpy.ndarray
        Homogeneous transformation matrix.
    point : point to be transformed
        Point is length-3 or shape (3,1).

    Returns
    -------
    transformed_point : tuple
        Transformed (x, y, z) coordinates.
    """
    if point is None:
        point = []

    bottom_row = np.array([[1.0]])

    homogenous_point = np.asarray(point).reshape(3, 1)
    homogenous_point = np.vstack([homogenous_point, bottom_row])
    transformed_point = tuple(np.dot(transformation, homogenous_point).flatten()[:3])

    return transformed_point

def get_nodes(container, instance_name=None, bounds=None):
    """
    Return a set of nodes and their labels using a bounding box.
    
    Works for:
    - Assembly (requires instance_name)
    - Part (instance_name is ignored)
    
    Parameters
    ----------
    container : Assembly or Part
        The object on which to create the set.
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
            raise ValueError("[create_node_set] No nodes found in bounds {} on instance '{}'.".format(bounds, instance_name))

    # Part case
    elif hasattr(container, "nodes"):
        target_nodes = container.nodes.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
        if not target_nodes:
            raise ValueError("[create_node_set] No nodes found in bounds {} in part.".format(bounds))

    else:
        raise TypeError("[create_node_set] 'container' must be an Assembly or Part object.")

    # Extract labels from the nodes
    labels = [node.label for node in target_nodes]
    return target_nodes, labels

def equation_constraint(model, assembly, parent_part_name, child_part_name, nodes_to_link, linked_dof=[1, 2, 3, 4, 5, 6]):
    # nodes_to_link are sets of node labels that correspond to the parent_part_name and child_part_name parts
    for pair in nodes_to_link:
        label_one, label_two = pair[0], pair[1]
        # Create a set for each of these nodes
        assembly.Set(
            name='equation-set-{}-{}-{}-{}-1'.format(parent_part_name, child_part_name, label_one, label_two),
            nodes=assembly.instances[parent_part_name].nodes.sequenceFromLabels((label_one,))
        )
        assembly.Set(
            name='equation-set-{}-{}-{}-{}-2'.format(parent_part_name, child_part_name, label_one, label_two),
            nodes=assembly.instances[child_part_name].nodes.sequenceFromLabels((label_two,))
        )

        for dof in linked_dof:
            model.Equation(
                name='Equation-{}-{}-{}-{}-{}'.format(parent_part_name, child_part_name, label_one, label_two, dof),
                terms=(
                    (-1.0, 'equation-set-{}-{}-{}-{}-1'.format(parent_part_name, child_part_name, label_one, label_two), dof),
                    ( 1.0, 'equation-set-{}-{}-{}-{}-2'.format(parent_part_name, child_part_name, label_one, label_two), dof),
                ),
            )

def equation_sets(model, name, set_one, set_two, linked_dof=[1, 2, 3, 4, 5, 6]):
    """
    Link two sets using equations for each of the requested degrees of freedom
    
    Parameters
    ----------
    model : Abaqus model
    name : str
        Name of the equation as it will appear in the Abaqus tree
    set_one : name of the set of follower nodes
    set_two : name of the set of the main nodes
        Must contain a single 'driving' node for the rest of the set
    linked_dof : list[] containing the requested degrees of freedom to be linked
        x = 1, y = 2, z = 3, rev_x = 4, rev_y = 5, rev_z = 6
    
    Returns
    -------
    
    """

    for dof in linked_dof:
        model.Equation(
            name = '{}-{}'.format(name, dof),
            terms = (
                (1.0, set_one, dof),
                (-1.0, set_two, dof)
            )
        )

    print("[equation_sets] Linked '{}' with {}.".format(set_one, set_two))

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