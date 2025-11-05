# ABAQUS Prefactory Information
from abaqus import *
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True, reportDeprecated=False)

# Import module information from ABAQUS
from mesh import *

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
    method : str
        seedEdgeBySize : default
        seedEdgeByBias : "bias"
    """

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
    part.setMeshControls(regions=part.faces, technique=technique, elemShape=elemShape)

    # Assign element type
    elem_type = ElemType(elemCode=elemCode, elemLibrary=elemLibrary)
    part.setElementType(regions=(part.faces,), elemTypes=(elem_type,))

    # Generate mesh
    part.generateMesh()