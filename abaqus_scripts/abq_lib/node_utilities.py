def find_closest_node(container, reference_point, instance_name=None, 
                     restricted_directions=None, search_radius=None):
    """
    Function to find the closest node with optional restrictions.
    Automatically chooses the optimal algorithm based on parameters.
    
    Parameters
    ----------
    container : Abaqus Assembly or Part object
        Container with nodes to search
    reference_point : tuple
        Target location (x, y, z)  
    instance_name : str or None
        Instance name if searching within an Assembly
    restricted_directions : list of int or None
        Search restrictions for each axis [x, y, z]:
        -1: Don't search in negative direction
         0: No restrictions (search both directions)
         1: Don't search in positive direction
        If None, no restrictions applied (fastest path)
    search_radius : float or None
        Pre-filter nodes within this radius for large datasets
    
    Returns
    -------
    tuple
        (closest_node, node_label)
    """
    
    # Extract nodes
    if hasattr(container, "instances") and instance_name is not None:
        nodes = container.instances[instance_name].nodes
    elif hasattr(container, "nodes"):
        nodes = container.nodes  
    else:
        raise TypeError("Container must be an Assembly or Part object")
    
    # Check if restrictions are actually restrictive
    has_restrictions = (restricted_directions is not None and 
                        any(r != 0 for r in restricted_directions))
    
    # Choose optimal algorithm path
    if not has_restrictions and search_radius is None:
        # Fastest path: pure unrestricted search
        return _find_closest_unrestricted(nodes, reference_point)
    elif search_radius is not None:
        # Spatial filtering path for large datasets
        return _find_closest_with_spatial_filter(nodes, reference_point, 
                                                restricted_directions, search_radius)
    else:
        # Restricted search path
        return _find_closest_restricted(nodes, reference_point, restricted_directions)

def _find_closest_unrestricted(nodes, reference_point):
    """Optimized unrestricted search - fastest possible."""
    closest_node = None
    min_dist_squared = float('inf')
    ref_x, ref_y, ref_z = reference_point
    
    for node in nodes:
        x, y, z = node.coordinates
        dx, dy, dz = x - ref_x, y - ref_y, z - ref_z
        dist_squared = dx*dx + dy*dy + dz*dz
        
        if dist_squared < min_dist_squared:
            min_dist_squared = dist_squared
            closest_node = node
    
    if closest_node is None:
        raise ValueError("No nodes found")
    
    return closest_node, closest_node.label

def _find_closest_restricted(nodes, reference_point, restricted_directions):
    """Restricted search with directional constraints."""
    closest_node = None
    min_dist_squared = float('inf')
    ref_x, ref_y, ref_z = reference_point
    
    for node in nodes:
        x, y, z = node.coordinates
        dx, dy, dz = x - ref_x, y - ref_y, z - ref_z
        
        reject = False
        if restricted_directions[0] == -1 and dx < 0:
            reject = True
        elif restricted_directions[0] == 1 and dx > 0:
            reject = True
        elif restricted_directions[1] == -1 and dy < 0:
            reject = True
        elif restricted_directions[1] == 1 and dy > 0:
            reject = True
        elif restricted_directions[2] == -1 and dz < 0:
            reject = True
        elif restricted_directions[2] == 1 and dz > 0:
            reject = True
        
        if reject:
            continue
            
        dist_squared = dx*dx + dy*dy + dz*dz
        if dist_squared < min_dist_squared:
            min_dist_squared = dist_squared
            closest_node = node
    
    if closest_node is None:
        raise ValueError("No nodes found within restricted domain")
    
    return closest_node, closest_node.label

def _find_closest_with_spatial_filter(nodes, reference_point, restricted_directions, search_radius):
    """Search with spatial pre-filtering for large datasets."""
    ref_x, ref_y, ref_z = reference_point
    radius_squared = search_radius * search_radius
    
    # Pre-filter by radius
    candidate_nodes = []
    for node in nodes:
        x, y, z = node.coordinates
        dx, dy, dz = x - ref_x, y - ref_y, z - ref_z
        if dx*dx + dy*dy + dz*dz <= radius_squared:
            candidate_nodes.append(node)
    
    if not candidate_nodes:
        raise ValueError("No nodes found within search radius")
    
    # Apply restrictions if any
    has_restrictions = (restricted_directions is not None and 
                       any(r != 0 for r in restricted_directions))
    
    if has_restrictions:
        return _find_closest_restricted(candidate_nodes, reference_point, restricted_directions)
    else:
        return _find_closest_unrestricted(candidate_nodes, reference_point)

# Convenience functions for common use cases
def find_closest_node_above(container, reference_point, instance_name=None):
    """Find closest node in positive Z direction only."""
    return find_closest_node(container, reference_point, instance_name,
                           restricted_directions=[0, 0, -1])

def find_closest_node_below(container, reference_point, instance_name=None):
    """Find closest node in negative Z direction only."""
    return find_closest_node(container, reference_point, instance_name,
                           restricted_directions=[0, 0, 1])

def find_closest_node_right(container, reference_point, instance_name=None):
    """Find closest node in positive X direction only."""
    return find_closest_node(container, reference_point, instance_name,
                           restricted_directions=[-1, 0, 0])

def find_closest_node_left(container, reference_point, instance_name=None):
    """Find closest node in negative X direction only.""" 
    return find_closest_node(container, reference_point, instance_name,
                           restricted_directions=[1, 0, 0])

def move_closest_nodes_to_axis(part, target_point, axis_dof = 1, free_dof = 2, restricted_directions=None, verbose = False):
    """Move the closest nodes along the axis_dof direction to target_point along the free_dof direction"""
    # Pass the restricted_directions directly - if they have not been called, they will be None, otherwise, we have a restricted domain
    reference_point, _ = find_closest_node(part, target_point, restricted_directions=restricted_directions)

    # Capture all of the points on the part that lie along the line of action of the dof
    capture_offset = 0.001
    max_bound = 1e5 # A large number that will never be reached by the bounds of the part

    # Capture nodes along the axis defined by the dof and the reference_point
    nodes, _ = get_nodes_along_axis(part, reference_point.coordinates, axis_dof, max_bound, capture_offset)

    for node in nodes:
        # Find the coordinates of the point and presribe the neutral axis location
        temp_coords = node.coordinates

        coordinates = list(node.coordinates)
        coordinates[free_dof - 1] = target_point[free_dof - 1]

        # Move mesh to match this neutral axis location
        part.editNode(nodes=(node,), coordinates=(tuple(coordinates),))
        if verbose is True:
            print("[move_closest_nodes_to_axis] Moved node '{}' from location {} to '{}'.".format(node.label, temp_coords, node.coordinates))

    labels = [node.label for node in nodes]
    return nodes, labels

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

def apply_geometric_imperfection(part, field = None):
    """
    Move each node within container as a function of the field function. Ex. field = [x, 0, 0] will move each node proportional to its 'x' position in the 'x' direction
    Use this function to presribe a global or local imperfection of the container

    Parameters
    ----------
    part : Part
        The part on which to prescribe the displacement.
    field : lambda function
        The displacement field in R3
    
    Returns
    -------
    nodes : Abaqus Node Sequence
    labels : List[int]
        Node labels
    """

    if field is None:
        raise ValueError("[apply_geometric_imperfection] No displacement field defined.")
    
    if not hasattr(part, "nodes"):
        raise ValueError("[apply_geometric_imperfection] `part` is not type Part.")

    nodes = part.nodes
    new_coordinates = []
    
    node_labels = []
    for node in nodes:
        x, y, z = node.coordinates
        dx, dy, dz = field(x, y, z)
        new_coordinates.append((x + dx, y + dy, z + dz))
        node_labels.append(node.label)
    
    part.editNode(nodes=nodes, coordinates=new_coordinates)
    
    return part.nodes, node_labels