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

    # Seed edges based on faces
    for face_set_name, mesh_size in face_seed_map.items():
        edges_to_seed = []
        for f in part.sets[face_set_name].faces:
            edges_to_seed.extend(f.getEdges())

        # Remove duplicate edges
        edges_to_seed = list(set(edges_to_seed))

        # Apply seeding
        part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, constraint=constraint)

    # Apply mesh controls
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
model = mdb.Model(name='Parametric-Panel')

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
web_faces = []
for web_location in web_locations:
    faces = p.faces.getByBoundingBox(
        xMin=web_location-1e-6, xMax=web_location+1e-6,
        yMin=0.0, yMax=panel.h_longitudinal_web+1e-6,
        zMin=0.0, zMax=panel.length+1e-6
    )
    web_faces.extend(faces)

web_faces = tuple(web_faces)
p.Set(faces=web_faces, name='WebFaces')

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

face_seed_map = {
    'PlateFace': panel.mesh_plate,
    'WebFaces': panel.mesh_web,
    'FlangeFaces': panel.mesh_flange,
}

mesh_from_faces(model.parts['plate'], face_seed_map)