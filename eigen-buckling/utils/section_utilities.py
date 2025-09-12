# ABAQUS Prefactory Information
from abaqus import *
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True, reportDeprecated=False)

# Import module information from ABAQUS
import regionToolset

# from utils.node_utilities import *

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
    print("[assign_section_bounds] Assigned section '{}' to {} faces.".format(section_name, len(faces)))

def set_local_element_thickness(part, target_point, axis_dof, section_name='local-thickness', depth_of_search=1, set_name='temp'):
    """Assign a section to elements connected along an axis, expanding out by edge-sharing neighbours."""
    from utils.node_utilities import find_closest_node, get_nodes_along_axis
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