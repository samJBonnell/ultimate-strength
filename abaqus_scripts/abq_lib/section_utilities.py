# ABAQUS Prefactory Information
from abaqus import *
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True, reportDeprecated=False)

# Import module information from ABAQUS
import regionToolset

# from utils.node_utilities import *

def assign_section(container, section_name, part_name=None, method="points", 
                   catch_points=None, bounds=None, set_name=None, target_type="faces"):
    """
    Unified function to assign a section to faces or elements using different methods.
    
    Parameters
    ----------
    container : Model or Part
        The Abaqus model object (for points method) or part object (for other methods).
    section_name : str
        Name of the section to assign.
    part_name : str, optional
        Name of the part (required only for points method when model is passed).
    method : str
        Assignment method: "points", "bounds", or "sets".
    catch_points : list, optional
        List of coordinate points for face finding (required for points method).
    bounds : tuple, optional
        Bounding box (x_min, x_max, y_min, y_max, z_min, z_max) for bounds method.
    set_name : str, optional
        Name of existing set (required for sets method).
    target_type : str, optional
        Either 'faces' or 'elements' (for bounds method). Default is 'faces'.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If required parameters are missing or invalid, or if no targets are found.
    """
    
    # Input validation
    valid_methods = ["points", "bounds", "sets"]
    if method not in valid_methods:
        raise ValueError("[assign_section] method must be one of {}".format(valid_methods))
    
    # Determine the part object based on method and input
    if method == "points":
        if hasattr(container, 'parts'):  # It's a model
            if not part_name:
                raise ValueError("[assign_section] part_name is required when using points method with model")
            part = container.parts[part_name]
        else:  # It's already a part
            part = container
            part_name = getattr(part, 'name', 'UnknownPart')
    else:
        if hasattr(container, 'parts'):  # It's a model, extract part
            if not part_name:
                raise ValueError("[assign_section] part_name is required when passing model for {} method".format(method))
            part = container.parts[part_name]
        else:  # It's already a part
            part = container
            part_name = getattr(part, 'name', 'UnknownPart')
    
    # Execute based on method
    if method == "points":
        if not catch_points:
            raise ValueError("[assign_section] catch_points is required for points method")
        targets, target_count = _assign_by_points(part, catch_points)
        target_desc = "face(s)"
        
    elif method == "bounds":
        if not bounds:
            raise ValueError("[assign_section] bounds is required for bounds method")
        targets, target_count, target_desc = _assign_by_bounds(part, bounds, target_type)
        
    elif method == "sets":
        if not set_name:
            raise ValueError("[assign_section] set_name is required for sets method")
        targets, target_count = _assign_by_sets(part, set_name)
        target_desc = "face(s)"
    
    # Create section assignment
    part.SectionAssignment(
        region=targets,
        sectionName=section_name,
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION
    )
    
    print("[assign_section] Assigned section '{}' to {} {} on part '{}' using {} method.".format(
        section_name, target_count, target_desc, part_name, method))


def _assign_by_points(part, catch_points):
    """Helper function for points-based assignment."""
    faces = []
    for point in catch_points:
        face = part.faces.findAt((point,))
        faces.append(face)
    
    if not faces:
        raise ValueError("[assign_section] No faces found for section assignment on part '{}'.".format(part.name))
    
    # Combine all faces
    all_faces = faces[0]
    for f in faces[1:]:
        all_faces += f
    
    # Create set and return region
    part.Set(name='sectionAssignment', faces=all_faces)
    return part.sets['sectionAssignment'], len(faces)


def _assign_by_bounds(part, bounds, target_type):
    """Helper function for bounds-based assignment."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Select the target collection
    if target_type.lower() == "faces":
        targets = part.faces.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
        target_desc = "face(s)"
    elif target_type.lower() == "elements":
        targets = part.elements.getByBoundingBox(
            xMin=x_min, xMax=x_max,
            yMin=y_min, yMax=y_max,
            zMin=z_min, zMax=z_max
        )
        target_desc = "element(s)"
    else:
        raise ValueError("[assign_section] target_type must be 'faces' or 'elements'")
    
    if not targets:
        raise ValueError("[assign_section] No {} found in bounds {} on part '{}'.".format(
            target_type, bounds, part.name))
    
    # Create set and return region
    part.Set(name='sectionAssignment', **{target_type: targets})
    return part.sets['sectionAssignment'], len(targets), target_desc


def _assign_by_sets(part, set_name):
    """Helper function for set-based assignment."""
    if set_name not in part.sets:
        raise ValueError("[assign_section] Set '{}' not found in part '{}'.".format(set_name, part.name))
    
    faces = part.sets[set_name].faces
    target_count = len(faces)
    print(faces)
    region = regionToolset.Region(faces=faces)
    return region, target_count


def set_local_section(part, seed_nodes, section_name='local-thickness', depth_of_search=1, set_name='temp', restriction_type=None, restriction_params=None):
    """
    Assign a section to elements connected to seed nodes, expanding out by edge-sharing neighbours.
    
    Parameters
    ----------
    part : Part
        The Abaqus part object
    seed_nodes : list or sequence
        Initial nodes to start the element selection from
    section_name : str
        Name of the section to assign
    depth_of_search : int
        Number of layers of edge-connected neighbors to include
    set_name : str
        Name for the element set created
    restriction_type : str, optional
        Type of restriction: 'bounds', 'distance', 'axis_aligned', or 'element_set'
    restriction_params : dict, optional
        Parameters for restriction:
        - 'bounds': {'x_min': val, 'x_max': val, 'y_min': val, 'y_max': val, 'z_min': val, 'z_max': val}
        - 'distance': {'max_distance': val, 'center_point': (x,y,z)}
        - 'axis_aligned': {'axis': 0/1/2, 'tolerance': val, 'reference_coord': val}
        - 'element_set': {'allowed_elements': set_of_labels}
    """
    
    def is_element_allowed(elem_label, restriction_type, restriction_params):
        """Check if element is allowed based on restriction criteria."""
        if restriction_type is None or restriction_params is None:
            return True
            
        elem = part.elements.getFromLabel(elem_label)
        
        if restriction_type == 'bounds':
            # Get element centroid
            nodes = elem.getNodes()
            centroid = [sum([node.coordinates[i] for node in nodes]) / len(nodes) for i in range(3)]
            bounds = restriction_params
            return (bounds.get('x_min', -1e10) <= centroid[0] <= bounds.get('x_max', 1e10) and
                    bounds.get('y_min', -1e10) <= centroid[1] <= bounds.get('y_max', 1e10) and
                    bounds.get('z_min', -1e10) <= centroid[2] <= bounds.get('z_max', 1e10))
        
        elif restriction_type == 'distance':
            # Check distance from center point
            nodes = elem.getNodes()
            centroid = [sum([node.coordinates[i] for node in nodes]) / len(nodes) for i in range(3)]
            center = restriction_params['center_point']
            max_dist = restriction_params['max_distance']
            
            dist = sum((centroid[i] - center[i])**2 for i in range(3))**0.5
            return dist <= max_dist
        
        elif restriction_type == 'axis_aligned':
            # Restrict traversal perpendicular to specified axis
            axis = restriction_params['axis']  # 0=x, 1=y, 2=z
            tolerance = restriction_params.get('tolerance', 0.1)
            reference_coord = restriction_params['reference_coord']
            
            nodes = elem.getNodes()
            centroid = [sum([node.coordinates[i] for node in nodes]) / len(nodes) for i in range(3)]
            
            return abs(centroid[axis] - reference_coord) <= tolerance
        
        elif restriction_type == 'element_set':
            # Only allow elements in specified set
            allowed = restriction_params['allowed_elements']
            return elem_label in allowed
        
        return True

    # Ensure list
    try:
        seed_nodes = list(seed_nodes)
    except TypeError:
        seed_nodes = [seed_nodes]

    seed_nodes = [n for n in seed_nodes if n is not None]

    # Initialize with elements touching the seed nodes
    visited = set()
    frontier = set()
    for node in seed_nodes:
        for e in node.getElements():
            if is_element_allowed(e.label, restriction_type, restriction_params):
                visited.add(e.label)
                frontier.add(e.label)

    # Expand outward by node connectivity
    for _ in range(depth_of_search):
        next_frontier = set()
        for elem_label in frontier:
            elem = part.elements.getFromLabel(elem_label)
            for n in elem.getNodes():
                for neigh in n.getElements():
                    if neigh.label not in visited and is_element_allowed(neigh.label, restriction_type, restriction_params):
                        visited.add(neigh.label)
                        next_frontier.add(neigh.label)
        frontier = next_frontier

    # Final selection
    selected_elements = part.elements.sequenceFromLabels(sorted(visited))
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


# Usage examples:
"""
# First, collect nodes using your utility functions elsewhere:
from utils.node_utilities import find_closest_node, get_nodes_along_axis

reference_point, _ = find_closest_node(part, target_point, restricted_directions=restricted_directions)
nodes_along_axis, _ = get_nodes_along_axis(part, reference_point.coordinates, axis_dof, max_bound, capture_offset)

# Then use this function with the collected nodes:
set_local_section(part, nodes_along_axis, 'my-section', depth_of_search=2)

# Or with restrictions:
set_local_section(part, nodes_along_axis, 'my-section', 
                  restriction_type='bounds',
                  restriction_params={'x_min': 0, 'x_max': 10})

# Or with single node:
single_node = [reference_point]
set_local_section(part, single_node, 'my-section')

# Or with any collection of nodes from different sources:
mixed_nodes = nodes_from_set1 + nodes_from_set2 + [specific_node]
set_local_section(part, mixed_nodes, 'my-section')
"""