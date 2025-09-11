# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Thu Sep 11 10:17:19 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=201.953903198242, 
    height=243.600006103516)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 870, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'FaceArray' object has no attribute 'sequenceFromMask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 869, in <module>
#*     web_faces = p.faces.sequenceFromMask(mask=[f.getMask() for f in 
#* web_faces])
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'Face' object has no attribute 'getMask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 869, in <module>
#*     web_faces = p.faces.getSequenceFromMask(mask=[f.getMask() for f in 
#* web_faces])
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'Face' object has no attribute 'mask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 871, in <module>
#*     web_faces = p.faces.getSequenceFromMask(mask=[f.mask for f in web_faces])
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#* SyntaxError: ('invalid syntax', 
#* ('C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
#* 870, 13, 'web_faces = \n'))
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'Face' object has no attribute 'mask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 870, in <module>
#*     web_faces = p.faces.getSequenceFromMask(mask=[f.mask for f in web_faces])
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* TypeError: mask[0]; found Face, expecting string
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 870, in <module>
#*     web_faces = p.faces.getSequenceFromMask(mask=web_faces_masks)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#: [mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.9, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.9, 0.083333, 2.0), (1.0, 0.0, 0.0))]
#* TypeError: mask[0]; found Face, expecting string
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 871, in <module>
#*     web_faces = p.faces.getSequenceFromMask(mask=web_faces_masks)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#: ("('[#20 ]',),",)
#: ("('[#4000 ]',),",)
#: ("('[#100 ]',),",)
#: ("('[#800 ]',),",)
#* TypeError: mask[0]; found Face, expecting string
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 871, in <module>
#*     web_faces = p.faces.getSequenceFromMask(mask=web_faces_masks)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#: Warning: Index 46 in the sequence is out of range
#: Warning: Index 72 in the sequence is out of range
#: Warning: Index 107 in the sequence is out of range
#* AttributeError: 'Struct' object has no attribute 'mesh_web'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 964, in <module>
#*     'WebFaces': panel.mesh_web,
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#: Warning: Index 46 in the sequence is out of range
#: Warning: Index 72 in the sequence is out of range
#: Warning: Index 107 in the sequence is out of range
#* AttributeError: 'Struct' object has no attribute 'mesh_flange'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 965, in <module>
#*     'FlangeFaces': panel.mesh_flange,
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#: Warning: Index 46 in the sequence is out of range
#: Warning: Index 72 in the sequence is out of range
#: Warning: Index 107 in the sequence is out of range
#* TypeError: edges[0]; found int, expecting Vertex
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 968, in <module>
#*     mesh_from_faces(model.parts['plate'], face_seed_map)
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 411, in mesh_from_faces
#*     part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, 
#* constraint=constraint)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#: Warning: Index 46 in the sequence is out of range
#: Warning: Index 72 in the sequence is out of range
#: Warning: Index 107 in the sequence is out of range
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'Face' object has no attribute 'getMask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 868, in <module>
#*     mask = faces.getMask()
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* TypeError: 'Face' object is not iterable
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 869, in <module>
#*     web_faces.extend(faces)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
a = mdb.models['Parametric-Panel'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* TypeError: edges[0]; found int, expecting Vertex
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 971, in <module>
#*     mesh_from_faces(model.parts['plate'], face_seed_map)
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 411, in mesh_from_faces
#*     part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, 
#* constraint=constraint)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'EdgeArray' object has no attribute 'sequenceFromMask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 975, in <module>
#*     mesh_from_faces(model.parts['plate'], face_seed_map)
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 413, in mesh_from_faces
#*     edges_to_seed = part.edges.sequenceFromMask(mask=[e.mask for e in 
#* edges_to_seed])
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* AttributeError: 'int' object has no attribute 'mask'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 975, in <module>
#*     mesh_from_faces(model.parts['plate'], face_seed_map)
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 413, in mesh_from_faces
#*     edges_to_seed = part.edges.getSequenceFromMask(mask=[e.mask for e in 
#* edges_to_seed])
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "eigen" has been created.
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
del mdb.models['Parametric-Panel']
del mdb.models['Model-1']
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p = mdb.models['eigen'].parts['plate']
p.seedPart(size=0.15, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['eigen'].parts['plate']
p.generateMesh()
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "eigen" has been created.
#: ({'allInternalSets': 'Repository object', 'allInternalSurfaces': 'Repository object', 'allSets': 'Repository object', 'allSurfaces': 'Repository object', 'beamSectionOrientations': 'BeamOrientationArray object', 'cells': 'CellArray object', 'compositeLayups': 'Repository object', 'datum': 'Repository object', 'datums': 'Repository object', 'edges': 'EdgeArray object', 'elemEdges': 'Repository object', 'elemFaces': 'Repository object', 'elementEdges': 'MeshEdgeArray object', 'elementFaces': 'MeshFaceArray object', 'elements': 'MeshElementArray object', 'engineeringFeatures': 'EngineeringFeatures object', 'faces': 'FaceArray object', 'features': 'Repository object', 'featuresById': 'Repository object', 'geometryPrecision': 1, 'geometryRefinement': COARSE, 'geometryValidity': 1, 'ignoredEdges': 'IgnoredEdgeArray object', 'ignoredVertices': 'IgnoredVertexArray object', 'ips': 'IPArray object', 'isOutOfDate': 0, 'materialOrientations': 'MaterialOrientationArray object', 'modelName': 'eigen', 'name': 'plate', 'nodes': 'MeshNodeArray object', 'rebarOrientations': 'RebarOrientationArray object', 'referencePoints': 'Repository object', 'reinforcements': 'Repository object', 'retainedNodes': 'unknown', 'sectionAssignments': 'SectionAssignmentArray object', 'sets': 'Repository object', 'skins': 'Repository object', 'space': THREE_D, 'stringers': 'Repository object', 'surfaces': 'Repository object', 'timeStamp': 5647.0, 'twist': OFF, 'type': DEFORMABLE_BODY, 'vertices': 'VertexArray object'})
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: Start Time: 1757614412
#: The model "eigen" has been created.
#: ({'allInternalSets': 'Repository object', 'allInternalSurfaces': 'Repository object', 'allSets': 'Repository object', 'allSurfaces': 'Repository object', 'beamSectionOrientations': 'BeamOrientationArray object', 'cells': 'CellArray object', 'compositeLayups': 'Repository object', 'datum': 'Repository object', 'datums': 'Repository object', 'edges': 'EdgeArray object', 'elemEdges': 'Repository object', 'elemFaces': 'Repository object', 'elementEdges': 'MeshEdgeArray object', 'elementFaces': 'MeshFaceArray object', 'elements': 'MeshElementArray object', 'engineeringFeatures': 'EngineeringFeatures object', 'faces': 'FaceArray object', 'features': 'Repository object', 'featuresById': 'Repository object', 'geometryPrecision': 1, 'geometryRefinement': COARSE, 'geometryValidity': 1, 'ignoredEdges': 'IgnoredEdgeArray object', 'ignoredVertices': 'IgnoredVertexArray object', 'ips': 'IPArray object', 'isOutOfDate': 0, 'materialOrientations': 'MaterialOrientationArray object', 'modelName': 'eigen', 'name': 'plate', 'nodes': 'MeshNodeArray object', 'rebarOrientations': 'RebarOrientationArray object', 'referencePoints': 'Repository object', 'reinforcements': 'Repository object', 'retainedNodes': 'unknown', 'sectionAssignments': 'SectionAssignmentArray object', 'sets': 'Repository object', 'skins': 'Repository object', 'space': THREE_D, 'stringers': 'Repository object', 'surfaces': 'Repository object', 'timeStamp': 5861.0, 'twist': OFF, 'type': DEFORMABLE_BODY, 'vertices': 'VertexArray object'})
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 1757614480
#: The model "eigen" has been created.
#: ({'allInternalSets': 'Repository object', 'allInternalSurfaces': 'Repository object', 'allSets': 'Repository object', 'allSurfaces': 'Repository object', 'beamSectionOrientations': 'BeamOrientationArray object', 'cells': 'CellArray object', 'compositeLayups': 'Repository object', 'datum': 'Repository object', 'datums': 'Repository object', 'edges': 'EdgeArray object', 'elemEdges': 'Repository object', 'elemFaces': 'Repository object', 'elementEdges': 'MeshEdgeArray object', 'elementFaces': 'MeshFaceArray object', 'elements': 'MeshElementArray object', 'engineeringFeatures': 'EngineeringFeatures object', 'faces': 'FaceArray object', 'features': 'Repository object', 'featuresById': 'Repository object', 'geometryPrecision': 1, 'geometryRefinement': COARSE, 'geometryValidity': 1, 'ignoredEdges': 'IgnoredEdgeArray object', 'ignoredVertices': 'IgnoredVertexArray object', 'ips': 'IPArray object', 'isOutOfDate': 0, 'materialOrientations': 'MaterialOrientationArray object', 'modelName': 'eigen', 'name': 'plate', 'nodes': 'MeshNodeArray object', 'rebarOrientations': 'RebarOrientationArray object', 'referencePoints': 'Repository object', 'reinforcements': 'Repository object', 'retainedNodes': 'unknown', 'sectionAssignments': 'SectionAssignmentArray object', 'sets': 'Repository object', 'skins': 'Repository object', 'space': THREE_D, 'stringers': 'Repository object', 'surfaces': 'Repository object', 'timeStamp': 6054.0, 'twist': OFF, 'type': DEFORMABLE_BODY, 'vertices': 'VertexArray object'})
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:15:54
#: The model "eigen" has been created.
#: ({'allInternalSets': 'Repository object', 'allInternalSurfaces': 'Repository object', 'allSets': 'Repository object', 'allSurfaces': 'Repository object', 'beamSectionOrientations': 'BeamOrientationArray object', 'cells': 'CellArray object', 'compositeLayups': 'Repository object', 'datum': 'Repository object', 'datums': 'Repository object', 'edges': 'EdgeArray object', 'elemEdges': 'Repository object', 'elemFaces': 'Repository object', 'elementEdges': 'MeshEdgeArray object', 'elementFaces': 'MeshFaceArray object', 'elements': 'MeshElementArray object', 'engineeringFeatures': 'EngineeringFeatures object', 'faces': 'FaceArray object', 'features': 'Repository object', 'featuresById': 'Repository object', 'geometryPrecision': 1, 'geometryRefinement': COARSE, 'geometryValidity': 1, 'ignoredEdges': 'IgnoredEdgeArray object', 'ignoredVertices': 'IgnoredVertexArray object', 'ips': 'IPArray object', 'isOutOfDate': 0, 'materialOrientations': 'MaterialOrientationArray object', 'modelName': 'eigen', 'name': 'plate', 'nodes': 'MeshNodeArray object', 'rebarOrientations': 'RebarOrientationArray object', 'referencePoints': 'Repository object', 'reinforcements': 'Repository object', 'retainedNodes': 'unknown', 'sectionAssignments': 'SectionAssignmentArray object', 'sets': 'Repository object', 'skins': 'Repository object', 'space': THREE_D, 'stringers': 'Repository object', 'surfaces': 'Repository object', 'timeStamp': 6247.0, 'twist': OFF, 'type': DEFORMABLE_BODY, 'vertices': 'VertexArray object'})
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:18:39
#: The model "eigen" has been created.
cliCommand("""face_seed_map = {
    'PlateFace': panel.mesh_plate,
    'WebFaces': panel.mesh_longitudinal_web,
    'FlangeFaces': panel.mesh_longitudinal_flange,
}""")
a = mdb.models['eigen'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Buckle-Step')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
cliCommand("""mesh_from_faces(model.parts['plate'], face_seed_map)""")
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
""")
cliCommand("""    # Iterate over each face in the set""")
cliCommand("""    for f in part.sets[face_set_name].faces:""")
#*     for f in part.sets[face_set_name].faces:
#*     ^
#* IndentationError: unexpected indent
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
    # Iterate over each face in the set
    for f in part.sets[face_set_name].faces:
        # f could be a Face or FaceArray; getEdges always returns EdgeArray
        for e in f.getEdges():
            edges_to_seed.append(e)
""")
cliCommand("""# Apply mesh controls to all faces""")
cliCommand("""part.setMeshControls(regions=part.faces[:], technique=STRUCTURED, elemShape=QUAD)""")
cliCommand("""# Assign element type""")
cliCommand("""elem_type = ElemType(elemCode=S4R, elemLibrary=STANDARD)""")
cliCommand("""part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type,))""")
cliCommand("""# Generate mesh""")
cliCommand("""part.generateMesh()""")
cliCommand("""edges_to_seed""")
#: [0, 1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 5, 11, 12, 13, 7, 11, 14, 15]
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
    # Iterate over each face in the set
    for f in part.sets[face_set_name].faces:
        # f could be a Face or FaceArray; getEdges always returns EdgeArray
        for e in f.getEdges():
            edges_to_seed.append(e)
""")
cliCommand("""# Apply mesh controls to all faces""")
cliCommand("""part.setMeshControls(regions=part.faces[:], technique=STRUCTURED, elemShape=QUAD)""")
cliCommand("""# Assign element type""")
cliCommand("""elem_type = ElemType(elemCode=S4, elemLibrary=STANDARD)""")
cliCommand("""part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type,))""")
cliCommand("""# Generate mesh""")
cliCommand("""part.generateMesh()""")
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, 
    adaptiveMeshConstraints=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
cliCommand("""face_seed_map.items()""")
#: [('FlangeFaces', 0.025), ('WebFaces', 0.025), ('PlateFace', 0.02)]
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:24:34
#: The model "eigen" has been created.
#* TypeError: edges[0]; found int, expecting Vertex
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 980, in <module>
#*     mesh_from_faces(model.parts['plate'], face_seed_map)
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 418, in mesh_from_faces
#*     part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, 
#* constraint=constraint)
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
""")
cliCommand("""    # Iterate over each face in the set""")
cliCommand("""    for f in part.sets[face_set_name].faces:""")
#*     for f in part.sets[face_set_name].faces:
#*     ^
#* IndentationError: unexpected indent
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
    # Iterate over each face in the set
    for f in part.sets[face_set_name].faces:
        # f could be a Face or FaceArray; getEdges always returns EdgeArray
        for e in f.getEdges():
            edges_to_seed.append(e)
    if edges_to_seed:
        # edges_to_seed = part.edges.sequenceFromMask(mask=[e.mask for e in edges_to_seed])
        part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, constraint=FREE)
""")
#* TypeError: edges[0]; found int, expecting Vertex
cliCommand("""edges_to_seed""")
#: [18, 19, 20, 21, 18, 22, 23, 24, 27, 28, 29, 30, 27, 31, 32, 33, 36, 37, 38, 39, 36, 40, 41, 42, 45, 46, 47, 48, 45, 49, 50, 51]
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
""")
cliCommand("""    # Collect edges from all faces""")
cliCommand("""    for f in part.sets[face_set_name].faces:""")
#*     for f in part.sets[face_set_name].faces:
#*     ^
#* IndentationError: unexpected indent
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
    # Collect edges from all faces
    for f in part.sets[face_set_name].faces:
        edges_to_seed.extend(f.getEdges())
    if edges_to_seed:
        # Remove duplicates using edge labels
        edge_labels = list(set(e.label for e in edges_to_seed))
        edges_to_seed_array = part.edges.sequenceFromLabels(edge_labels)
        # Apply seeding
        part.seedEdgeBySize(edges=edges_to_seed_array, size=mesh_size, constraint=FREE)
""")
#* AttributeError: 'int' object has no attribute 'label'
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
    # Collect edges from all faces
    for f in part.sets[face_set_name].faces:
        for e in f.getEdges():   # e may be an int or Edge object
            # Convert integer IDs to Edge objects if needed
            if isinstance(e, int):
                e = part.edges[e]  # Abaqus 2.7 allows indexing by ID
            edges_to_seed.append(e)
    if edges_to_seed:
        # Remove duplicates using Edge objects directly
        edges_to_seed = list({id(e): e for e in edges_to_seed}.values())
        # Apply seeding
        part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, constraint=FREE)
""")
cliCommand("""# Apply mesh controls to all faces""")
cliCommand("""part.setMeshControls(regions=part.faces[:], technique=STRUCTURED, elemShape=QUAD)""")
cliCommand("""# Assign element type""")
cliCommand("""elem_type = ElemType(elemCode=S4, elemLibrary=STANDARD)""")
cliCommand("""part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type,))""")
cliCommand("""# Generate mesh""")
cliCommand("""part.generateMesh()""")
a = mdb.models['eigen'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.99016, 
    farPlane=11.0487, width=1.21073, height=1.42536, viewOffsetX=-0.2008, 
    viewOffsetY=-0.469566)
cliCommand("""face_seed_map = {
    'PlateFace': 0.5,
    'WebFaces': panel.mesh_longitudinal_web,
    'FlangeFaces': panel.mesh_longitudinal_flange,
}""")
cliCommand("""mesh_from_faces(model.parts['plate'], face_seed_map)""")
#* TypeError: edges[0]; found int, expecting Vertex
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 418, in mesh_from_faces
#*     part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, 
#* constraint=constraint)
cliCommand("""face_seed_map = {
    'PlateFace': 0.5,
    'WebFaces': panel.mesh_longitudinal_web,
    'FlangeFaces': panel.mesh_longitudinal_flange,
}""")
cliCommand("""part = model.parts['plate']""")
cliCommand("""# Seed edges based on faces""")
cliCommand("""for face_set_name, mesh_size in face_seed_map.items():
    edges_to_seed = []
    # Collect edges from all faces
    for f in part.sets[face_set_name].faces:
        for e in f.getEdges():   # e may be an int or Edge object
            # Convert integer IDs to Edge objects if needed
            if isinstance(e, int):
                e = part.edges[e]  # Abaqus 2.7 allows indexing by ID
            edges_to_seed.append(e)
    if edges_to_seed:
        # Remove duplicates using Edge objects directly
        edges_to_seed = list({id(e): e for e in edges_to_seed}.values())
        # Apply seeding
        part.seedEdgeBySize(edges=edges_to_seed, size=mesh_size, constraint=FREE)
""")
cliCommand("""# Apply mesh controls to all faces""")
cliCommand("""part.setMeshControls(regions=part.faces[:], technique=STRUCTURED, elemShape=QUAD)""")
cliCommand("""# Assign element type""")
cliCommand("""elem_type = ElemType(elemCode=S4, elemLibrary=STANDARD)""")
cliCommand("""part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type,))""")
cliCommand("""# Generate mesh""")
cliCommand("""part.generateMesh()""")
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.7103, 
    farPlane=11.3285, width=2.51438, height=2.96012, viewOffsetX=-0.132894, 
    viewOffsetY=0.273522)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:35:22
#: The model "eigen" has been created.
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.8331, 
    farPlane=11.2057, width=1.94158, height=2.28578, viewOffsetX=-0.0988335, 
    viewOffsetY=-0.318654)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:35:44
#: The model "eigen" has been created.
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.75397, 
    farPlane=11.2849, width=2.31054, height=2.72014, viewOffsetX=-0.0594282, 
    viewOffsetY=-0.0930701)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:44:40
#: The model "eigen" has been created.
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON, mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:45:27
#: The model "eigen" has been created.
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:46:33
#: The model "eigen" has been created.
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:50:51
#: The model "eigen" has been created.
#* KeyError: Model-1
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 1070, in <module>
#*     p = mdb.models['Model-1'].Part(name='panel', dimensionality=THREE_D, 
#* type=DEFORMABLE_BODY)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:51:08
#: The model "eigen" has been created.
#* Invalid sketch
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 1071, in <module>
#*     p.BaseShell(sketch=old_part)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:51:50
#: The model "eigen" has been created.
#* Invalid sketch
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 1071, in <module>
#*     p.BaseShell(sketch=old_part)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 11:54:51
#: The model "eigen" has been created.
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['eigen'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.65644, 
    farPlane=11.3824, width=3.11552, height=3.64967, viewOffsetX=0.149835, 
    viewOffsetY=-0.0394177)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 12:20:36
#: The model "eigen" has been created.
#: [move_closest_nodes_to_axis] Moved node '2' from location (0.899999976158142, 0.0, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '3' from location (1.5, 0.0, 0.0) to '(1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '6' from location (0.300000011920929, 0.0, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '7' from location (-0.899999976158142, 0.0, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '10' from location (-1.5, 0.0, 0.0) to '(-1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '12' from location (-0.300000011920929, 0.0, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '186' from location (0.920000016689301, 0.0, 0.0) to '(0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '187' from location (0.939999997615814, 0.0, 0.0) to '(0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '188' from location (0.959999978542328, 0.0, 0.0) to '(0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '189' from location (0.980000019073486, 0.0, 0.0) to '(0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '190' from location (1.0, 0.0, 0.0) to '(1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '191' from location (1.01999998092651, 0.0, 0.0) to '(1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '192' from location (1.03999996185303, 0.0, 0.0) to '(1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '193' from location (1.05999994277954, 0.0, 0.0) to '(1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '194' from location (1.08000004291534, 0.0, 0.0) to '(1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '195' from location (1.10000002384186, 0.0, 0.0) to '(1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '196' from location (1.12000000476837, 0.0, 0.0) to '(1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '197' from location (1.13999998569489, 0.0, 0.0) to '(1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '198' from location (1.1599999666214, 0.0, 0.0) to '(1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '199' from location (1.17999994754791, 0.0, 0.0) to '(1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '200' from location (1.20000004768372, 0.0, 0.0) to '(1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '201' from location (1.22000002861023, 0.0, 0.0) to '(1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '202' from location (1.24000000953674, 0.0, 0.0) to '(1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '203' from location (1.25999999046326, 0.0, 0.0) to '(1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '204' from location (1.27999997138977, 0.0, 0.0) to '(1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '205' from location (1.29999995231628, 0.0, 0.0) to '(1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '206' from location (1.32000005245209, 0.0, 0.0) to '(1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '207' from location (1.3400000333786, 0.0, 0.0) to '(1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '208' from location (1.36000001430511, 0.0, 0.0) to '(1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '209' from location (1.37999999523163, 0.0, 0.0) to '(1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '210' from location (1.39999997615814, 0.0, 0.0) to '(1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '211' from location (1.41999995708466, 0.0, 0.0) to '(1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '212' from location (1.44000005722046, 0.0, 0.0) to '(1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '213' from location (1.46000003814697, 0.0, 0.0) to '(1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '214' from location (1.48000001907349, 0.0, 0.0) to '(1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '571' from location (0.319999992847443, 0.0, 0.0) to '(0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '572' from location (0.340000003576279, 0.0, 0.0) to '(0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '573' from location (0.360000014305115, 0.0, 0.0) to '(0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '574' from location (0.379999995231628, 0.0, 0.0) to '(0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '575' from location (0.400000005960464, 0.0, 0.0) to '(0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '576' from location (0.419999986886978, 0.0, 0.0) to '(0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '577' from location (0.439999997615814, 0.0, 0.0) to '(0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '578' from location (0.46000000834465, 0.0, 0.0) to '(0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '579' from location (0.479999989271164, 0.0, 0.0) to '(0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '580' from location (0.5, 0.0, 0.0) to '(0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '581' from location (0.519999980926514, 0.0, 0.0) to '(0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '582' from location (0.540000021457672, 0.0, 0.0) to '(0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '583' from location (0.560000002384186, 0.0, 0.0) to '(0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '584' from location (0.579999983310699, 0.0, 0.0) to '(0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '585' from location (0.600000023841858, 0.0, 0.0) to '(0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '586' from location (0.620000004768372, 0.0, 0.0) to '(0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '587' from location (0.639999985694885, 0.0, 0.0) to '(0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '588' from location (0.660000026226044, 0.0, 0.0) to '(0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '589' from location (0.680000007152557, 0.0, 0.0) to '(0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '590' from location (0.699999988079071, 0.0, 0.0) to '(0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '591' from location (0.720000028610229, 0.0, 0.0) to '(0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '592' from location (0.740000009536743, 0.0, 0.0) to '(0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '593' from location (0.759999990463257, 0.0, 0.0) to '(0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '594' from location (0.779999971389771, 0.0, 0.0) to '(0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '595' from location (0.800000011920929, 0.0, 0.0) to '(0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '596' from location (0.819999992847443, 0.0, 0.0) to '(0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '597' from location (0.839999973773956, 0.0, 0.0) to '(0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '598' from location (0.860000014305115, 0.0, 0.0) to '(0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '599' from location (0.879999995231628, 0.0, 0.0) to '(0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '927' from location (-1.48000001907349, 0.0, 0.0) to '(-1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '928' from location (-1.46000003814697, 0.0, 0.0) to '(-1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '929' from location (-1.44000005722046, 0.0, 0.0) to '(-1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '930' from location (-1.41999995708466, 0.0, 0.0) to '(-1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '931' from location (-1.39999997615814, 0.0, 0.0) to '(-1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '932' from location (-1.37999999523163, 0.0, 0.0) to '(-1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '933' from location (-1.36000001430511, 0.0, 0.0) to '(-1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '934' from location (-1.3400000333786, 0.0, 0.0) to '(-1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '935' from location (-1.32000005245209, 0.0, 0.0) to '(-1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '936' from location (-1.29999995231628, 0.0, 0.0) to '(-1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '937' from location (-1.27999997138977, 0.0, 0.0) to '(-1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '938' from location (-1.25999999046326, 0.0, 0.0) to '(-1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '939' from location (-1.24000000953674, 0.0, 0.0) to '(-1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '940' from location (-1.22000002861023, 0.0, 0.0) to '(-1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '941' from location (-1.20000004768372, 0.0, 0.0) to '(-1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '942' from location (-1.17999994754791, 0.0, 0.0) to '(-1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '943' from location (-1.1599999666214, 0.0, 0.0) to '(-1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '944' from location (-1.13999998569489, 0.0, 0.0) to '(-1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '945' from location (-1.12000000476837, 0.0, 0.0) to '(-1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '946' from location (-1.10000002384186, 0.0, 0.0) to '(-1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '947' from location (-1.08000004291534, 0.0, 0.0) to '(-1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '948' from location (-1.05999994277954, 0.0, 0.0) to '(-1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '949' from location (-1.03999996185303, 0.0, 0.0) to '(-1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '950' from location (-1.01999998092651, 0.0, 0.0) to '(-1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '951' from location (-1.0, 0.0, 0.0) to '(-1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '952' from location (-0.980000019073486, 0.0, 0.0) to '(-0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '953' from location (-0.959999978542328, 0.0, 0.0) to '(-0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '954' from location (-0.939999997615814, 0.0, 0.0) to '(-0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '955' from location (-0.920000016689301, 0.0, 0.0) to '(-0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1105' from location (-0.280000001192093, 0.0, 0.0) to '(-0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1106' from location (-0.259999990463257, 0.0, 0.0) to '(-0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1107' from location (-0.239999994635582, 0.0, 0.0) to '(-0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1108' from location (-0.219999998807907, 0.0, 0.0) to '(-0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1109' from location (-0.200000002980232, 0.0, 0.0) to '(-0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1110' from location (-0.180000007152557, 0.0, 0.0) to '(-0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1111' from location (-0.159999996423721, 0.0, 0.0) to '(-0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1112' from location (-0.140000000596046, 0.0, 0.0) to '(-0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1113' from location (-0.119999997317791, 0.0, 0.0) to '(-0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1114' from location (-0.100000001490116, 0.0, 0.0) to '(-0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1115' from location (-0.0799999982118607, 0.0, 0.0) to '(-0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1116' from location (-0.0599999986588955, 0.0, 0.0) to '(-0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1117' from location (-0.0399999991059303, 0.0, 0.0) to '(-0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1118' from location (-0.0199999995529652, 0.0, 0.0) to '(-0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1119' from location (2.22044604925031e-16, 0.0, 0.0) to '(2.22044604925031e-16, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1120' from location (0.0199999995529652, 0.0, 0.0) to '(0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1121' from location (0.0399999991059303, 0.0, 0.0) to '(0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1122' from location (0.0599999986588955, 0.0, 0.0) to '(0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1123' from location (0.0799999982118607, 0.0, 0.0) to '(0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1124' from location (0.100000001490116, 0.0, 0.0) to '(0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1125' from location (0.119999997317791, 0.0, 0.0) to '(0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1126' from location (0.140000000596046, 0.0, 0.0) to '(0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1127' from location (0.159999996423721, 0.0, 0.0) to '(0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1128' from location (0.180000007152557, 0.0, 0.0) to '(0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1129' from location (0.200000002980232, 0.0, 0.0) to '(0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1130' from location (0.219999998807907, 0.0, 0.0) to '(0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1131' from location (0.239999994635582, 0.0, 0.0) to '(0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1132' from location (0.259999990463257, 0.0, 0.0) to '(0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1133' from location (0.280000001192093, 0.0, 0.0) to '(0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1192' from location (-0.879999995231628, 0.0, 0.0) to '(-0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1193' from location (-0.860000014305115, 0.0, 0.0) to '(-0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1194' from location (-0.839999973773956, 0.0, 0.0) to '(-0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1195' from location (-0.819999992847443, 0.0, 0.0) to '(-0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1196' from location (-0.800000011920929, 0.0, 0.0) to '(-0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1197' from location (-0.779999971389771, 0.0, 0.0) to '(-0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1198' from location (-0.759999990463257, 0.0, 0.0) to '(-0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1199' from location (-0.740000009536743, 0.0, 0.0) to '(-0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1200' from location (-0.720000028610229, 0.0, 0.0) to '(-0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1201' from location (-0.699999988079071, 0.0, 0.0) to '(-0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1202' from location (-0.680000007152557, 0.0, 0.0) to '(-0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1203' from location (-0.660000026226044, 0.0, 0.0) to '(-0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1204' from location (-0.639999985694885, 0.0, 0.0) to '(-0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1205' from location (-0.620000004768372, 0.0, 0.0) to '(-0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1206' from location (-0.600000023841858, 0.0, 0.0) to '(-0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1207' from location (-0.579999983310699, 0.0, 0.0) to '(-0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1208' from location (-0.560000002384186, 0.0, 0.0) to '(-0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1209' from location (-0.540000021457672, 0.0, 0.0) to '(-0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1210' from location (-0.519999980926514, 0.0, 0.0) to '(-0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1211' from location (-0.5, 0.0, 0.0) to '(-0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1212' from location (-0.479999989271164, 0.0, 0.0) to '(-0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1213' from location (-0.46000000834465, 0.0, 0.0) to '(-0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1214' from location (-0.439999997615814, 0.0, 0.0) to '(-0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1215' from location (-0.419999986886978, 0.0, 0.0) to '(-0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1216' from location (-0.400000005960464, 0.0, 0.0) to '(-0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1217' from location (-0.379999995231628, 0.0, 0.0) to '(-0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1218' from location (-0.360000014305115, 0.0, 0.0) to '(-0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1219' from location (-0.340000003576279, 0.0, 0.0) to '(-0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1220' from location (-0.319999992847443, 0.0, 0.0) to '(-0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1' from location (0.899999976158142, 0.0, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '4' from location (1.5, 0.0, 3.0) to '(1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '5' from location (0.300000011920929, 0.0, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '8' from location (-0.899999976158142, 0.0, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '9' from location (-1.5, 0.0, 3.0) to '(-1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '11' from location (-0.300000011920929, 0.0, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '364' from location (1.48000001907349, 0.0, 3.0) to '(1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '365' from location (1.46000003814697, 0.0, 3.0) to '(1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '366' from location (1.44000005722046, 0.0, 3.0) to '(1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '367' from location (1.41999995708466, 0.0, 3.0) to '(1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '368' from location (1.39999997615814, 0.0, 3.0) to '(1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '369' from location (1.37999999523163, 0.0, 3.0) to '(1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '370' from location (1.36000001430511, 0.0, 3.0) to '(1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '371' from location (1.3400000333786, 0.0, 3.0) to '(1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '372' from location (1.32000005245209, 0.0, 3.0) to '(1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '373' from location (1.29999995231628, 0.0, 3.0) to '(1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '374' from location (1.27999997138977, 0.0, 3.0) to '(1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '375' from location (1.25999999046326, 0.0, 3.0) to '(1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '376' from location (1.24000000953674, 0.0, 3.0) to '(1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '377' from location (1.22000002861023, 0.0, 3.0) to '(1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '378' from location (1.20000004768372, 0.0, 3.0) to '(1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '379' from location (1.17999994754791, 0.0, 3.0) to '(1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '380' from location (1.1599999666214, 0.0, 3.0) to '(1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '381' from location (1.13999998569489, 0.0, 3.0) to '(1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '382' from location (1.12000000476837, 0.0, 3.0) to '(1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '383' from location (1.10000002384186, 0.0, 3.0) to '(1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '384' from location (1.08000004291534, 0.0, 3.0) to '(1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '385' from location (1.05999994277954, 0.0, 3.0) to '(1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '386' from location (1.03999996185303, 0.0, 3.0) to '(1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '387' from location (1.01999998092651, 0.0, 3.0) to '(1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '388' from location (1.0, 0.0, 3.0) to '(1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '389' from location (0.980000019073486, 0.0, 3.0) to '(0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '390' from location (0.959999978542328, 0.0, 3.0) to '(0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '391' from location (0.939999997615814, 0.0, 3.0) to '(0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '392' from location (0.920000016689301, 0.0, 3.0) to '(0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '393' from location (0.879999995231628, 0.0, 3.0) to '(0.879999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '394' from location (0.860000014305115, 0.0, 3.0) to '(0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '395' from location (0.839999973773956, 0.0, 3.0) to '(0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '396' from location (0.819999992847443, 0.0, 3.0) to '(0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '397' from location (0.800000011920929, 0.0, 3.0) to '(0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '398' from location (0.779999971389771, 0.0, 3.0) to '(0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '399' from location (0.759999990463257, 0.0, 3.0) to '(0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '400' from location (0.740000009536743, 0.0, 3.0) to '(0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '401' from location (0.720000028610229, 0.0, 3.0) to '(0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '402' from location (0.699999988079071, 0.0, 3.0) to '(0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '403' from location (0.680000007152557, 0.0, 3.0) to '(0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '404' from location (0.660000026226044, 0.0, 3.0) to '(0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '405' from location (0.639999985694885, 0.0, 3.0) to '(0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '406' from location (0.620000004768372, 0.0, 3.0) to '(0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '407' from location (0.600000023841858, 0.0, 3.0) to '(0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '408' from location (0.579999983310699, 0.0, 3.0) to '(0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '409' from location (0.560000002384186, 0.0, 3.0) to '(0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '410' from location (0.540000021457672, 0.0, 3.0) to '(0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '411' from location (0.519999980926514, 0.0, 3.0) to '(0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '412' from location (0.5, 0.0, 3.0) to '(0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '413' from location (0.479999989271164, 0.0, 3.0) to '(0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '414' from location (0.46000000834465, 0.0, 3.0) to '(0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '415' from location (0.439999997615814, 0.0, 3.0) to '(0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '416' from location (0.419999986886978, 0.0, 3.0) to '(0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '417' from location (0.400000005960464, 0.0, 3.0) to '(0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '418' from location (0.379999995231628, 0.0, 3.0) to '(0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '419' from location (0.360000014305115, 0.0, 3.0) to '(0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '420' from location (0.340000003576279, 0.0, 3.0) to '(0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '421' from location (0.319999992847443, 0.0, 3.0) to '(0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '749' from location (-0.920000016689301, 0.0, 3.0) to '(-0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '750' from location (-0.939999997615814, 0.0, 3.0) to '(-0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '751' from location (-0.959999978542328, 0.0, 3.0) to '(-0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '752' from location (-0.980000019073486, 0.0, 3.0) to '(-0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '753' from location (-1.0, 0.0, 3.0) to '(-1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '754' from location (-1.01999998092651, 0.0, 3.0) to '(-1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '755' from location (-1.03999996185303, 0.0, 3.0) to '(-1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '756' from location (-1.05999994277954, 0.0, 3.0) to '(-1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '757' from location (-1.08000004291534, 0.0, 3.0) to '(-1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '758' from location (-1.10000002384186, 0.0, 3.0) to '(-1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '759' from location (-1.12000000476837, 0.0, 3.0) to '(-1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '760' from location (-1.13999998569489, 0.0, 3.0) to '(-1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '761' from location (-1.1599999666214, 0.0, 3.0) to '(-1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '762' from location (-1.17999994754791, 0.0, 3.0) to '(-1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '763' from location (-1.20000004768372, 0.0, 3.0) to '(-1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '764' from location (-1.22000002861023, 0.0, 3.0) to '(-1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '765' from location (-1.24000000953674, 0.0, 3.0) to '(-1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '766' from location (-1.25999999046326, 0.0, 3.0) to '(-1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '767' from location (-1.27999997138977, 0.0, 3.0) to '(-1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '768' from location (-1.29999995231628, 0.0, 3.0) to '(-1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '769' from location (-1.32000005245209, 0.0, 3.0) to '(-1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '770' from location (-1.3400000333786, 0.0, 3.0) to '(-1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '771' from location (-1.36000001430511, 0.0, 3.0) to '(-1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '772' from location (-1.37999999523163, 0.0, 3.0) to '(-1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '773' from location (-1.39999997615814, 0.0, 3.0) to '(-1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '774' from location (-1.41999995708466, 0.0, 3.0) to '(-1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '775' from location (-1.44000005722046, 0.0, 3.0) to '(-1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '776' from location (-1.46000003814697, 0.0, 3.0) to '(-1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '777' from location (-1.48000001907349, 0.0, 3.0) to '(-1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1134' from location (0.280000001192093, 0.0, 3.0) to '(0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1135' from location (0.259999990463257, 0.0, 3.0) to '(0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1136' from location (0.239999994635582, 0.0, 3.0) to '(0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1137' from location (0.219999998807907, 0.0, 3.0) to '(0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1138' from location (0.200000002980232, 0.0, 3.0) to '(0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1139' from location (0.180000007152557, 0.0, 3.0) to '(0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1140' from location (0.159999996423721, 0.0, 3.0) to '(0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1141' from location (0.140000000596046, 0.0, 3.0) to '(0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1142' from location (0.119999997317791, 0.0, 3.0) to '(0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1143' from location (0.100000001490116, 0.0, 3.0) to '(0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1144' from location (0.0799999982118607, 0.0, 3.0) to '(0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1145' from location (0.0599999986588955, 0.0, 3.0) to '(0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1146' from location (0.0399999991059303, 0.0, 3.0) to '(0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1147' from location (0.0199999995529652, 0.0, 3.0) to '(0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1148' from location (-4.44089209850063e-16, 0.0, 3.0) to '(-4.44089209850063e-16, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1149' from location (-0.0199999995529652, 0.0, 3.0) to '(-0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1150' from location (-0.0399999991059303, 0.0, 3.0) to '(-0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1151' from location (-0.0599999986588955, 0.0, 3.0) to '(-0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1152' from location (-0.0799999982118607, 0.0, 3.0) to '(-0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1153' from location (-0.100000001490116, 0.0, 3.0) to '(-0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1154' from location (-0.119999997317791, 0.0, 3.0) to '(-0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1155' from location (-0.140000000596046, 0.0, 3.0) to '(-0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1156' from location (-0.159999996423721, 0.0, 3.0) to '(-0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1157' from location (-0.180000007152557, 0.0, 3.0) to '(-0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1158' from location (-0.200000002980232, 0.0, 3.0) to '(-0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1159' from location (-0.219999998807907, 0.0, 3.0) to '(-0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1160' from location (-0.239999994635582, 0.0, 3.0) to '(-0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1161' from location (-0.259999990463257, 0.0, 3.0) to '(-0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1162' from location (-0.280000001192093, 0.0, 3.0) to '(-0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1163' from location (-0.319999992847443, 0.0, 3.0) to '(-0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1164' from location (-0.340000003576279, 0.0, 3.0) to '(-0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1165' from location (-0.360000014305115, 0.0, 3.0) to '(-0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1166' from location (-0.379999995231628, 0.0, 3.0) to '(-0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1167' from location (-0.400000005960464, 0.0, 3.0) to '(-0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1168' from location (-0.419999986886978, 0.0, 3.0) to '(-0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1169' from location (-0.439999997615814, 0.0, 3.0) to '(-0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1170' from location (-0.46000000834465, 0.0, 3.0) to '(-0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1171' from location (-0.479999989271164, 0.0, 3.0) to '(-0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1172' from location (-0.5, 0.0, 3.0) to '(-0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1173' from location (-0.519999980926514, 0.0, 3.0) to '(-0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1174' from location (-0.540000021457672, 0.0, 3.0) to '(-0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1175' from location (-0.560000002384186, 0.0, 3.0) to '(-0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1176' from location (-0.579999983310699, 0.0, 3.0) to '(-0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1177' from location (-0.600000023841858, 0.0, 3.0) to '(-0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1178' from location (-0.620000004768372, 0.0, 3.0) to '(-0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1179' from location (-0.639999985694885, 0.0, 3.0) to '(-0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1180' from location (-0.660000026226044, 0.0, 3.0) to '(-0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1181' from location (-0.680000007152557, 0.0, 3.0) to '(-0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1182' from location (-0.699999988079071, 0.0, 3.0) to '(-0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1183' from location (-0.720000028610229, 0.0, 3.0) to '(-0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1184' from location (-0.740000009536743, 0.0, 3.0) to '(-0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1185' from location (-0.759999990463257, 0.0, 3.0) to '(-0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1186' from location (-0.779999971389771, 0.0, 3.0) to '(-0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1187' from location (-0.800000011920929, 0.0, 3.0) to '(-0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1188' from location (-0.819999992847443, 0.0, 3.0) to '(-0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1189' from location (-0.839999973773956, 0.0, 3.0) to '(-0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1190' from location (-0.860000014305115, 0.0, 3.0) to '(-0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1191' from location (-0.879999995231628, 0.0, 3.0) to '(-0.879999995231628, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 474 elements.
#: Assigned section 'local-thickness' to 474 elements.
#* TypeError: arg1(labels)[0]; found tuple, expecting int
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 1275, in <module>
#*     nodes = 
#* assembly.instances['panel'].nodes.sequenceFromLabels((labels[1:],))
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 12:21:45
#: The model "eigen" has been created.
#: [move_closest_nodes_to_axis] Moved node '2' from location (0.899999976158142, 0.0, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '3' from location (1.5, 0.0, 0.0) to '(1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '6' from location (0.300000011920929, 0.0, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '7' from location (-0.899999976158142, 0.0, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '10' from location (-1.5, 0.0, 0.0) to '(-1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '12' from location (-0.300000011920929, 0.0, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '186' from location (0.920000016689301, 0.0, 0.0) to '(0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '187' from location (0.939999997615814, 0.0, 0.0) to '(0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '188' from location (0.959999978542328, 0.0, 0.0) to '(0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '189' from location (0.980000019073486, 0.0, 0.0) to '(0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '190' from location (1.0, 0.0, 0.0) to '(1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '191' from location (1.01999998092651, 0.0, 0.0) to '(1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '192' from location (1.03999996185303, 0.0, 0.0) to '(1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '193' from location (1.05999994277954, 0.0, 0.0) to '(1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '194' from location (1.08000004291534, 0.0, 0.0) to '(1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '195' from location (1.10000002384186, 0.0, 0.0) to '(1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '196' from location (1.12000000476837, 0.0, 0.0) to '(1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '197' from location (1.13999998569489, 0.0, 0.0) to '(1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '198' from location (1.1599999666214, 0.0, 0.0) to '(1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '199' from location (1.17999994754791, 0.0, 0.0) to '(1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '200' from location (1.20000004768372, 0.0, 0.0) to '(1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '201' from location (1.22000002861023, 0.0, 0.0) to '(1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '202' from location (1.24000000953674, 0.0, 0.0) to '(1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '203' from location (1.25999999046326, 0.0, 0.0) to '(1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '204' from location (1.27999997138977, 0.0, 0.0) to '(1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '205' from location (1.29999995231628, 0.0, 0.0) to '(1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '206' from location (1.32000005245209, 0.0, 0.0) to '(1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '207' from location (1.3400000333786, 0.0, 0.0) to '(1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '208' from location (1.36000001430511, 0.0, 0.0) to '(1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '209' from location (1.37999999523163, 0.0, 0.0) to '(1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '210' from location (1.39999997615814, 0.0, 0.0) to '(1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '211' from location (1.41999995708466, 0.0, 0.0) to '(1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '212' from location (1.44000005722046, 0.0, 0.0) to '(1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '213' from location (1.46000003814697, 0.0, 0.0) to '(1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '214' from location (1.48000001907349, 0.0, 0.0) to '(1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '571' from location (0.319999992847443, 0.0, 0.0) to '(0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '572' from location (0.340000003576279, 0.0, 0.0) to '(0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '573' from location (0.360000014305115, 0.0, 0.0) to '(0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '574' from location (0.379999995231628, 0.0, 0.0) to '(0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '575' from location (0.400000005960464, 0.0, 0.0) to '(0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '576' from location (0.419999986886978, 0.0, 0.0) to '(0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '577' from location (0.439999997615814, 0.0, 0.0) to '(0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '578' from location (0.46000000834465, 0.0, 0.0) to '(0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '579' from location (0.479999989271164, 0.0, 0.0) to '(0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '580' from location (0.5, 0.0, 0.0) to '(0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '581' from location (0.519999980926514, 0.0, 0.0) to '(0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '582' from location (0.540000021457672, 0.0, 0.0) to '(0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '583' from location (0.560000002384186, 0.0, 0.0) to '(0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '584' from location (0.579999983310699, 0.0, 0.0) to '(0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '585' from location (0.600000023841858, 0.0, 0.0) to '(0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '586' from location (0.620000004768372, 0.0, 0.0) to '(0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '587' from location (0.639999985694885, 0.0, 0.0) to '(0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '588' from location (0.660000026226044, 0.0, 0.0) to '(0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '589' from location (0.680000007152557, 0.0, 0.0) to '(0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '590' from location (0.699999988079071, 0.0, 0.0) to '(0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '591' from location (0.720000028610229, 0.0, 0.0) to '(0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '592' from location (0.740000009536743, 0.0, 0.0) to '(0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '593' from location (0.759999990463257, 0.0, 0.0) to '(0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '594' from location (0.779999971389771, 0.0, 0.0) to '(0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '595' from location (0.800000011920929, 0.0, 0.0) to '(0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '596' from location (0.819999992847443, 0.0, 0.0) to '(0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '597' from location (0.839999973773956, 0.0, 0.0) to '(0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '598' from location (0.860000014305115, 0.0, 0.0) to '(0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '599' from location (0.879999995231628, 0.0, 0.0) to '(0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '927' from location (-1.48000001907349, 0.0, 0.0) to '(-1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '928' from location (-1.46000003814697, 0.0, 0.0) to '(-1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '929' from location (-1.44000005722046, 0.0, 0.0) to '(-1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '930' from location (-1.41999995708466, 0.0, 0.0) to '(-1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '931' from location (-1.39999997615814, 0.0, 0.0) to '(-1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '932' from location (-1.37999999523163, 0.0, 0.0) to '(-1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '933' from location (-1.36000001430511, 0.0, 0.0) to '(-1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '934' from location (-1.3400000333786, 0.0, 0.0) to '(-1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '935' from location (-1.32000005245209, 0.0, 0.0) to '(-1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '936' from location (-1.29999995231628, 0.0, 0.0) to '(-1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '937' from location (-1.27999997138977, 0.0, 0.0) to '(-1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '938' from location (-1.25999999046326, 0.0, 0.0) to '(-1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '939' from location (-1.24000000953674, 0.0, 0.0) to '(-1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '940' from location (-1.22000002861023, 0.0, 0.0) to '(-1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '941' from location (-1.20000004768372, 0.0, 0.0) to '(-1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '942' from location (-1.17999994754791, 0.0, 0.0) to '(-1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '943' from location (-1.1599999666214, 0.0, 0.0) to '(-1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '944' from location (-1.13999998569489, 0.0, 0.0) to '(-1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '945' from location (-1.12000000476837, 0.0, 0.0) to '(-1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '946' from location (-1.10000002384186, 0.0, 0.0) to '(-1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '947' from location (-1.08000004291534, 0.0, 0.0) to '(-1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '948' from location (-1.05999994277954, 0.0, 0.0) to '(-1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '949' from location (-1.03999996185303, 0.0, 0.0) to '(-1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '950' from location (-1.01999998092651, 0.0, 0.0) to '(-1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '951' from location (-1.0, 0.0, 0.0) to '(-1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '952' from location (-0.980000019073486, 0.0, 0.0) to '(-0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '953' from location (-0.959999978542328, 0.0, 0.0) to '(-0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '954' from location (-0.939999997615814, 0.0, 0.0) to '(-0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '955' from location (-0.920000016689301, 0.0, 0.0) to '(-0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1105' from location (-0.280000001192093, 0.0, 0.0) to '(-0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1106' from location (-0.259999990463257, 0.0, 0.0) to '(-0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1107' from location (-0.239999994635582, 0.0, 0.0) to '(-0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1108' from location (-0.219999998807907, 0.0, 0.0) to '(-0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1109' from location (-0.200000002980232, 0.0, 0.0) to '(-0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1110' from location (-0.180000007152557, 0.0, 0.0) to '(-0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1111' from location (-0.159999996423721, 0.0, 0.0) to '(-0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1112' from location (-0.140000000596046, 0.0, 0.0) to '(-0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1113' from location (-0.119999997317791, 0.0, 0.0) to '(-0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1114' from location (-0.100000001490116, 0.0, 0.0) to '(-0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1115' from location (-0.0799999982118607, 0.0, 0.0) to '(-0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1116' from location (-0.0599999986588955, 0.0, 0.0) to '(-0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1117' from location (-0.0399999991059303, 0.0, 0.0) to '(-0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1118' from location (-0.0199999995529652, 0.0, 0.0) to '(-0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1119' from location (2.22044604925031e-16, 0.0, 0.0) to '(2.22044604925031e-16, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1120' from location (0.0199999995529652, 0.0, 0.0) to '(0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1121' from location (0.0399999991059303, 0.0, 0.0) to '(0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1122' from location (0.0599999986588955, 0.0, 0.0) to '(0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1123' from location (0.0799999982118607, 0.0, 0.0) to '(0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1124' from location (0.100000001490116, 0.0, 0.0) to '(0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1125' from location (0.119999997317791, 0.0, 0.0) to '(0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1126' from location (0.140000000596046, 0.0, 0.0) to '(0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1127' from location (0.159999996423721, 0.0, 0.0) to '(0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1128' from location (0.180000007152557, 0.0, 0.0) to '(0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1129' from location (0.200000002980232, 0.0, 0.0) to '(0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1130' from location (0.219999998807907, 0.0, 0.0) to '(0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1131' from location (0.239999994635582, 0.0, 0.0) to '(0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1132' from location (0.259999990463257, 0.0, 0.0) to '(0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1133' from location (0.280000001192093, 0.0, 0.0) to '(0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1192' from location (-0.879999995231628, 0.0, 0.0) to '(-0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1193' from location (-0.860000014305115, 0.0, 0.0) to '(-0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1194' from location (-0.839999973773956, 0.0, 0.0) to '(-0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1195' from location (-0.819999992847443, 0.0, 0.0) to '(-0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1196' from location (-0.800000011920929, 0.0, 0.0) to '(-0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1197' from location (-0.779999971389771, 0.0, 0.0) to '(-0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1198' from location (-0.759999990463257, 0.0, 0.0) to '(-0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1199' from location (-0.740000009536743, 0.0, 0.0) to '(-0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1200' from location (-0.720000028610229, 0.0, 0.0) to '(-0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1201' from location (-0.699999988079071, 0.0, 0.0) to '(-0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1202' from location (-0.680000007152557, 0.0, 0.0) to '(-0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1203' from location (-0.660000026226044, 0.0, 0.0) to '(-0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1204' from location (-0.639999985694885, 0.0, 0.0) to '(-0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1205' from location (-0.620000004768372, 0.0, 0.0) to '(-0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1206' from location (-0.600000023841858, 0.0, 0.0) to '(-0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1207' from location (-0.579999983310699, 0.0, 0.0) to '(-0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1208' from location (-0.560000002384186, 0.0, 0.0) to '(-0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1209' from location (-0.540000021457672, 0.0, 0.0) to '(-0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1210' from location (-0.519999980926514, 0.0, 0.0) to '(-0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1211' from location (-0.5, 0.0, 0.0) to '(-0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1212' from location (-0.479999989271164, 0.0, 0.0) to '(-0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1213' from location (-0.46000000834465, 0.0, 0.0) to '(-0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1214' from location (-0.439999997615814, 0.0, 0.0) to '(-0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1215' from location (-0.419999986886978, 0.0, 0.0) to '(-0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1216' from location (-0.400000005960464, 0.0, 0.0) to '(-0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1217' from location (-0.379999995231628, 0.0, 0.0) to '(-0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1218' from location (-0.360000014305115, 0.0, 0.0) to '(-0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1219' from location (-0.340000003576279, 0.0, 0.0) to '(-0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1220' from location (-0.319999992847443, 0.0, 0.0) to '(-0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1' from location (0.899999976158142, 0.0, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '4' from location (1.5, 0.0, 3.0) to '(1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '5' from location (0.300000011920929, 0.0, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '8' from location (-0.899999976158142, 0.0, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '9' from location (-1.5, 0.0, 3.0) to '(-1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '11' from location (-0.300000011920929, 0.0, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '364' from location (1.48000001907349, 0.0, 3.0) to '(1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '365' from location (1.46000003814697, 0.0, 3.0) to '(1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '366' from location (1.44000005722046, 0.0, 3.0) to '(1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '367' from location (1.41999995708466, 0.0, 3.0) to '(1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '368' from location (1.39999997615814, 0.0, 3.0) to '(1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '369' from location (1.37999999523163, 0.0, 3.0) to '(1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '370' from location (1.36000001430511, 0.0, 3.0) to '(1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '371' from location (1.3400000333786, 0.0, 3.0) to '(1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '372' from location (1.32000005245209, 0.0, 3.0) to '(1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '373' from location (1.29999995231628, 0.0, 3.0) to '(1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '374' from location (1.27999997138977, 0.0, 3.0) to '(1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '375' from location (1.25999999046326, 0.0, 3.0) to '(1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '376' from location (1.24000000953674, 0.0, 3.0) to '(1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '377' from location (1.22000002861023, 0.0, 3.0) to '(1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '378' from location (1.20000004768372, 0.0, 3.0) to '(1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '379' from location (1.17999994754791, 0.0, 3.0) to '(1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '380' from location (1.1599999666214, 0.0, 3.0) to '(1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '381' from location (1.13999998569489, 0.0, 3.0) to '(1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '382' from location (1.12000000476837, 0.0, 3.0) to '(1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '383' from location (1.10000002384186, 0.0, 3.0) to '(1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '384' from location (1.08000004291534, 0.0, 3.0) to '(1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '385' from location (1.05999994277954, 0.0, 3.0) to '(1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '386' from location (1.03999996185303, 0.0, 3.0) to '(1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '387' from location (1.01999998092651, 0.0, 3.0) to '(1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '388' from location (1.0, 0.0, 3.0) to '(1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '389' from location (0.980000019073486, 0.0, 3.0) to '(0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '390' from location (0.959999978542328, 0.0, 3.0) to '(0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '391' from location (0.939999997615814, 0.0, 3.0) to '(0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '392' from location (0.920000016689301, 0.0, 3.0) to '(0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '393' from location (0.879999995231628, 0.0, 3.0) to '(0.879999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '394' from location (0.860000014305115, 0.0, 3.0) to '(0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '395' from location (0.839999973773956, 0.0, 3.0) to '(0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '396' from location (0.819999992847443, 0.0, 3.0) to '(0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '397' from location (0.800000011920929, 0.0, 3.0) to '(0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '398' from location (0.779999971389771, 0.0, 3.0) to '(0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '399' from location (0.759999990463257, 0.0, 3.0) to '(0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '400' from location (0.740000009536743, 0.0, 3.0) to '(0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '401' from location (0.720000028610229, 0.0, 3.0) to '(0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '402' from location (0.699999988079071, 0.0, 3.0) to '(0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '403' from location (0.680000007152557, 0.0, 3.0) to '(0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '404' from location (0.660000026226044, 0.0, 3.0) to '(0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '405' from location (0.639999985694885, 0.0, 3.0) to '(0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '406' from location (0.620000004768372, 0.0, 3.0) to '(0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '407' from location (0.600000023841858, 0.0, 3.0) to '(0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '408' from location (0.579999983310699, 0.0, 3.0) to '(0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '409' from location (0.560000002384186, 0.0, 3.0) to '(0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '410' from location (0.540000021457672, 0.0, 3.0) to '(0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '411' from location (0.519999980926514, 0.0, 3.0) to '(0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '412' from location (0.5, 0.0, 3.0) to '(0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '413' from location (0.479999989271164, 0.0, 3.0) to '(0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '414' from location (0.46000000834465, 0.0, 3.0) to '(0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '415' from location (0.439999997615814, 0.0, 3.0) to '(0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '416' from location (0.419999986886978, 0.0, 3.0) to '(0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '417' from location (0.400000005960464, 0.0, 3.0) to '(0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '418' from location (0.379999995231628, 0.0, 3.0) to '(0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '419' from location (0.360000014305115, 0.0, 3.0) to '(0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '420' from location (0.340000003576279, 0.0, 3.0) to '(0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '421' from location (0.319999992847443, 0.0, 3.0) to '(0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '749' from location (-0.920000016689301, 0.0, 3.0) to '(-0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '750' from location (-0.939999997615814, 0.0, 3.0) to '(-0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '751' from location (-0.959999978542328, 0.0, 3.0) to '(-0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '752' from location (-0.980000019073486, 0.0, 3.0) to '(-0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '753' from location (-1.0, 0.0, 3.0) to '(-1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '754' from location (-1.01999998092651, 0.0, 3.0) to '(-1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '755' from location (-1.03999996185303, 0.0, 3.0) to '(-1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '756' from location (-1.05999994277954, 0.0, 3.0) to '(-1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '757' from location (-1.08000004291534, 0.0, 3.0) to '(-1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '758' from location (-1.10000002384186, 0.0, 3.0) to '(-1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '759' from location (-1.12000000476837, 0.0, 3.0) to '(-1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '760' from location (-1.13999998569489, 0.0, 3.0) to '(-1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '761' from location (-1.1599999666214, 0.0, 3.0) to '(-1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '762' from location (-1.17999994754791, 0.0, 3.0) to '(-1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '763' from location (-1.20000004768372, 0.0, 3.0) to '(-1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '764' from location (-1.22000002861023, 0.0, 3.0) to '(-1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '765' from location (-1.24000000953674, 0.0, 3.0) to '(-1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '766' from location (-1.25999999046326, 0.0, 3.0) to '(-1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '767' from location (-1.27999997138977, 0.0, 3.0) to '(-1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '768' from location (-1.29999995231628, 0.0, 3.0) to '(-1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '769' from location (-1.32000005245209, 0.0, 3.0) to '(-1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '770' from location (-1.3400000333786, 0.0, 3.0) to '(-1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '771' from location (-1.36000001430511, 0.0, 3.0) to '(-1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '772' from location (-1.37999999523163, 0.0, 3.0) to '(-1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '773' from location (-1.39999997615814, 0.0, 3.0) to '(-1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '774' from location (-1.41999995708466, 0.0, 3.0) to '(-1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '775' from location (-1.44000005722046, 0.0, 3.0) to '(-1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '776' from location (-1.46000003814697, 0.0, 3.0) to '(-1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '777' from location (-1.48000001907349, 0.0, 3.0) to '(-1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1134' from location (0.280000001192093, 0.0, 3.0) to '(0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1135' from location (0.259999990463257, 0.0, 3.0) to '(0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1136' from location (0.239999994635582, 0.0, 3.0) to '(0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1137' from location (0.219999998807907, 0.0, 3.0) to '(0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1138' from location (0.200000002980232, 0.0, 3.0) to '(0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1139' from location (0.180000007152557, 0.0, 3.0) to '(0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1140' from location (0.159999996423721, 0.0, 3.0) to '(0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1141' from location (0.140000000596046, 0.0, 3.0) to '(0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1142' from location (0.119999997317791, 0.0, 3.0) to '(0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1143' from location (0.100000001490116, 0.0, 3.0) to '(0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1144' from location (0.0799999982118607, 0.0, 3.0) to '(0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1145' from location (0.0599999986588955, 0.0, 3.0) to '(0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1146' from location (0.0399999991059303, 0.0, 3.0) to '(0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1147' from location (0.0199999995529652, 0.0, 3.0) to '(0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1148' from location (-4.44089209850063e-16, 0.0, 3.0) to '(-4.44089209850063e-16, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1149' from location (-0.0199999995529652, 0.0, 3.0) to '(-0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1150' from location (-0.0399999991059303, 0.0, 3.0) to '(-0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1151' from location (-0.0599999986588955, 0.0, 3.0) to '(-0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1152' from location (-0.0799999982118607, 0.0, 3.0) to '(-0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1153' from location (-0.100000001490116, 0.0, 3.0) to '(-0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1154' from location (-0.119999997317791, 0.0, 3.0) to '(-0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1155' from location (-0.140000000596046, 0.0, 3.0) to '(-0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1156' from location (-0.159999996423721, 0.0, 3.0) to '(-0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1157' from location (-0.180000007152557, 0.0, 3.0) to '(-0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1158' from location (-0.200000002980232, 0.0, 3.0) to '(-0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1159' from location (-0.219999998807907, 0.0, 3.0) to '(-0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1160' from location (-0.239999994635582, 0.0, 3.0) to '(-0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1161' from location (-0.259999990463257, 0.0, 3.0) to '(-0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1162' from location (-0.280000001192093, 0.0, 3.0) to '(-0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1163' from location (-0.319999992847443, 0.0, 3.0) to '(-0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1164' from location (-0.340000003576279, 0.0, 3.0) to '(-0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1165' from location (-0.360000014305115, 0.0, 3.0) to '(-0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1166' from location (-0.379999995231628, 0.0, 3.0) to '(-0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1167' from location (-0.400000005960464, 0.0, 3.0) to '(-0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1168' from location (-0.419999986886978, 0.0, 3.0) to '(-0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1169' from location (-0.439999997615814, 0.0, 3.0) to '(-0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1170' from location (-0.46000000834465, 0.0, 3.0) to '(-0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1171' from location (-0.479999989271164, 0.0, 3.0) to '(-0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1172' from location (-0.5, 0.0, 3.0) to '(-0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1173' from location (-0.519999980926514, 0.0, 3.0) to '(-0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1174' from location (-0.540000021457672, 0.0, 3.0) to '(-0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1175' from location (-0.560000002384186, 0.0, 3.0) to '(-0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1176' from location (-0.579999983310699, 0.0, 3.0) to '(-0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1177' from location (-0.600000023841858, 0.0, 3.0) to '(-0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1178' from location (-0.620000004768372, 0.0, 3.0) to '(-0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1179' from location (-0.639999985694885, 0.0, 3.0) to '(-0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1180' from location (-0.660000026226044, 0.0, 3.0) to '(-0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1181' from location (-0.680000007152557, 0.0, 3.0) to '(-0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1182' from location (-0.699999988079071, 0.0, 3.0) to '(-0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1183' from location (-0.720000028610229, 0.0, 3.0) to '(-0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1184' from location (-0.740000009536743, 0.0, 3.0) to '(-0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1185' from location (-0.759999990463257, 0.0, 3.0) to '(-0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1186' from location (-0.779999971389771, 0.0, 3.0) to '(-0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1187' from location (-0.800000011920929, 0.0, 3.0) to '(-0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1188' from location (-0.819999992847443, 0.0, 3.0) to '(-0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1189' from location (-0.839999973773956, 0.0, 3.0) to '(-0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1190' from location (-0.860000014305115, 0.0, 3.0) to '(-0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1191' from location (-0.879999995231628, 0.0, 3.0) to '(-0.879999995231628, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 474 elements.
#: Assigned section 'local-thickness' to 474 elements.
#* TypeError: keyword error on nam
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 1278, in <module>
#*     equation_sets(model, 'Web{}'.format(index), 
#* 'Web{}-Follower'.format(index), 'Web{}-Main'.format(index), linked_dof= [2])
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 829, in equation_sets
#*     (-1.0, set_two, dof)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 12:22:30
#: The model "eigen" has been created.
#: [move_closest_nodes_to_axis] Moved node '2' from location (0.899999976158142, 0.0, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '3' from location (1.5, 0.0, 0.0) to '(1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '6' from location (0.300000011920929, 0.0, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '7' from location (-0.899999976158142, 0.0, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '10' from location (-1.5, 0.0, 0.0) to '(-1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '12' from location (-0.300000011920929, 0.0, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '186' from location (0.920000016689301, 0.0, 0.0) to '(0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '187' from location (0.939999997615814, 0.0, 0.0) to '(0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '188' from location (0.959999978542328, 0.0, 0.0) to '(0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '189' from location (0.980000019073486, 0.0, 0.0) to '(0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '190' from location (1.0, 0.0, 0.0) to '(1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '191' from location (1.01999998092651, 0.0, 0.0) to '(1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '192' from location (1.03999996185303, 0.0, 0.0) to '(1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '193' from location (1.05999994277954, 0.0, 0.0) to '(1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '194' from location (1.08000004291534, 0.0, 0.0) to '(1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '195' from location (1.10000002384186, 0.0, 0.0) to '(1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '196' from location (1.12000000476837, 0.0, 0.0) to '(1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '197' from location (1.13999998569489, 0.0, 0.0) to '(1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '198' from location (1.1599999666214, 0.0, 0.0) to '(1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '199' from location (1.17999994754791, 0.0, 0.0) to '(1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '200' from location (1.20000004768372, 0.0, 0.0) to '(1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '201' from location (1.22000002861023, 0.0, 0.0) to '(1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '202' from location (1.24000000953674, 0.0, 0.0) to '(1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '203' from location (1.25999999046326, 0.0, 0.0) to '(1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '204' from location (1.27999997138977, 0.0, 0.0) to '(1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '205' from location (1.29999995231628, 0.0, 0.0) to '(1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '206' from location (1.32000005245209, 0.0, 0.0) to '(1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '207' from location (1.3400000333786, 0.0, 0.0) to '(1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '208' from location (1.36000001430511, 0.0, 0.0) to '(1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '209' from location (1.37999999523163, 0.0, 0.0) to '(1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '210' from location (1.39999997615814, 0.0, 0.0) to '(1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '211' from location (1.41999995708466, 0.0, 0.0) to '(1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '212' from location (1.44000005722046, 0.0, 0.0) to '(1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '213' from location (1.46000003814697, 0.0, 0.0) to '(1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '214' from location (1.48000001907349, 0.0, 0.0) to '(1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '571' from location (0.319999992847443, 0.0, 0.0) to '(0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '572' from location (0.340000003576279, 0.0, 0.0) to '(0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '573' from location (0.360000014305115, 0.0, 0.0) to '(0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '574' from location (0.379999995231628, 0.0, 0.0) to '(0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '575' from location (0.400000005960464, 0.0, 0.0) to '(0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '576' from location (0.419999986886978, 0.0, 0.0) to '(0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '577' from location (0.439999997615814, 0.0, 0.0) to '(0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '578' from location (0.46000000834465, 0.0, 0.0) to '(0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '579' from location (0.479999989271164, 0.0, 0.0) to '(0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '580' from location (0.5, 0.0, 0.0) to '(0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '581' from location (0.519999980926514, 0.0, 0.0) to '(0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '582' from location (0.540000021457672, 0.0, 0.0) to '(0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '583' from location (0.560000002384186, 0.0, 0.0) to '(0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '584' from location (0.579999983310699, 0.0, 0.0) to '(0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '585' from location (0.600000023841858, 0.0, 0.0) to '(0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '586' from location (0.620000004768372, 0.0, 0.0) to '(0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '587' from location (0.639999985694885, 0.0, 0.0) to '(0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '588' from location (0.660000026226044, 0.0, 0.0) to '(0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '589' from location (0.680000007152557, 0.0, 0.0) to '(0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '590' from location (0.699999988079071, 0.0, 0.0) to '(0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '591' from location (0.720000028610229, 0.0, 0.0) to '(0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '592' from location (0.740000009536743, 0.0, 0.0) to '(0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '593' from location (0.759999990463257, 0.0, 0.0) to '(0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '594' from location (0.779999971389771, 0.0, 0.0) to '(0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '595' from location (0.800000011920929, 0.0, 0.0) to '(0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '596' from location (0.819999992847443, 0.0, 0.0) to '(0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '597' from location (0.839999973773956, 0.0, 0.0) to '(0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '598' from location (0.860000014305115, 0.0, 0.0) to '(0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '599' from location (0.879999995231628, 0.0, 0.0) to '(0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '927' from location (-1.48000001907349, 0.0, 0.0) to '(-1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '928' from location (-1.46000003814697, 0.0, 0.0) to '(-1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '929' from location (-1.44000005722046, 0.0, 0.0) to '(-1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '930' from location (-1.41999995708466, 0.0, 0.0) to '(-1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '931' from location (-1.39999997615814, 0.0, 0.0) to '(-1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '932' from location (-1.37999999523163, 0.0, 0.0) to '(-1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '933' from location (-1.36000001430511, 0.0, 0.0) to '(-1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '934' from location (-1.3400000333786, 0.0, 0.0) to '(-1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '935' from location (-1.32000005245209, 0.0, 0.0) to '(-1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '936' from location (-1.29999995231628, 0.0, 0.0) to '(-1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '937' from location (-1.27999997138977, 0.0, 0.0) to '(-1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '938' from location (-1.25999999046326, 0.0, 0.0) to '(-1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '939' from location (-1.24000000953674, 0.0, 0.0) to '(-1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '940' from location (-1.22000002861023, 0.0, 0.0) to '(-1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '941' from location (-1.20000004768372, 0.0, 0.0) to '(-1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '942' from location (-1.17999994754791, 0.0, 0.0) to '(-1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '943' from location (-1.1599999666214, 0.0, 0.0) to '(-1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '944' from location (-1.13999998569489, 0.0, 0.0) to '(-1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '945' from location (-1.12000000476837, 0.0, 0.0) to '(-1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '946' from location (-1.10000002384186, 0.0, 0.0) to '(-1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '947' from location (-1.08000004291534, 0.0, 0.0) to '(-1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '948' from location (-1.05999994277954, 0.0, 0.0) to '(-1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '949' from location (-1.03999996185303, 0.0, 0.0) to '(-1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '950' from location (-1.01999998092651, 0.0, 0.0) to '(-1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '951' from location (-1.0, 0.0, 0.0) to '(-1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '952' from location (-0.980000019073486, 0.0, 0.0) to '(-0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '953' from location (-0.959999978542328, 0.0, 0.0) to '(-0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '954' from location (-0.939999997615814, 0.0, 0.0) to '(-0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '955' from location (-0.920000016689301, 0.0, 0.0) to '(-0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1105' from location (-0.280000001192093, 0.0, 0.0) to '(-0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1106' from location (-0.259999990463257, 0.0, 0.0) to '(-0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1107' from location (-0.239999994635582, 0.0, 0.0) to '(-0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1108' from location (-0.219999998807907, 0.0, 0.0) to '(-0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1109' from location (-0.200000002980232, 0.0, 0.0) to '(-0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1110' from location (-0.180000007152557, 0.0, 0.0) to '(-0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1111' from location (-0.159999996423721, 0.0, 0.0) to '(-0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1112' from location (-0.140000000596046, 0.0, 0.0) to '(-0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1113' from location (-0.119999997317791, 0.0, 0.0) to '(-0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1114' from location (-0.100000001490116, 0.0, 0.0) to '(-0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1115' from location (-0.0799999982118607, 0.0, 0.0) to '(-0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1116' from location (-0.0599999986588955, 0.0, 0.0) to '(-0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1117' from location (-0.0399999991059303, 0.0, 0.0) to '(-0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1118' from location (-0.0199999995529652, 0.0, 0.0) to '(-0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1119' from location (2.22044604925031e-16, 0.0, 0.0) to '(2.22044604925031e-16, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1120' from location (0.0199999995529652, 0.0, 0.0) to '(0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1121' from location (0.0399999991059303, 0.0, 0.0) to '(0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1122' from location (0.0599999986588955, 0.0, 0.0) to '(0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1123' from location (0.0799999982118607, 0.0, 0.0) to '(0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1124' from location (0.100000001490116, 0.0, 0.0) to '(0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1125' from location (0.119999997317791, 0.0, 0.0) to '(0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1126' from location (0.140000000596046, 0.0, 0.0) to '(0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1127' from location (0.159999996423721, 0.0, 0.0) to '(0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1128' from location (0.180000007152557, 0.0, 0.0) to '(0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1129' from location (0.200000002980232, 0.0, 0.0) to '(0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1130' from location (0.219999998807907, 0.0, 0.0) to '(0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1131' from location (0.239999994635582, 0.0, 0.0) to '(0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1132' from location (0.259999990463257, 0.0, 0.0) to '(0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1133' from location (0.280000001192093, 0.0, 0.0) to '(0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1192' from location (-0.879999995231628, 0.0, 0.0) to '(-0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1193' from location (-0.860000014305115, 0.0, 0.0) to '(-0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1194' from location (-0.839999973773956, 0.0, 0.0) to '(-0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1195' from location (-0.819999992847443, 0.0, 0.0) to '(-0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1196' from location (-0.800000011920929, 0.0, 0.0) to '(-0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1197' from location (-0.779999971389771, 0.0, 0.0) to '(-0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1198' from location (-0.759999990463257, 0.0, 0.0) to '(-0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1199' from location (-0.740000009536743, 0.0, 0.0) to '(-0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1200' from location (-0.720000028610229, 0.0, 0.0) to '(-0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1201' from location (-0.699999988079071, 0.0, 0.0) to '(-0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1202' from location (-0.680000007152557, 0.0, 0.0) to '(-0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1203' from location (-0.660000026226044, 0.0, 0.0) to '(-0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1204' from location (-0.639999985694885, 0.0, 0.0) to '(-0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1205' from location (-0.620000004768372, 0.0, 0.0) to '(-0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1206' from location (-0.600000023841858, 0.0, 0.0) to '(-0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1207' from location (-0.579999983310699, 0.0, 0.0) to '(-0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1208' from location (-0.560000002384186, 0.0, 0.0) to '(-0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1209' from location (-0.540000021457672, 0.0, 0.0) to '(-0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1210' from location (-0.519999980926514, 0.0, 0.0) to '(-0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1211' from location (-0.5, 0.0, 0.0) to '(-0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1212' from location (-0.479999989271164, 0.0, 0.0) to '(-0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1213' from location (-0.46000000834465, 0.0, 0.0) to '(-0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1214' from location (-0.439999997615814, 0.0, 0.0) to '(-0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1215' from location (-0.419999986886978, 0.0, 0.0) to '(-0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1216' from location (-0.400000005960464, 0.0, 0.0) to '(-0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1217' from location (-0.379999995231628, 0.0, 0.0) to '(-0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1218' from location (-0.360000014305115, 0.0, 0.0) to '(-0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1219' from location (-0.340000003576279, 0.0, 0.0) to '(-0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1220' from location (-0.319999992847443, 0.0, 0.0) to '(-0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1' from location (0.899999976158142, 0.0, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '4' from location (1.5, 0.0, 3.0) to '(1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '5' from location (0.300000011920929, 0.0, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '8' from location (-0.899999976158142, 0.0, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '9' from location (-1.5, 0.0, 3.0) to '(-1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '11' from location (-0.300000011920929, 0.0, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '364' from location (1.48000001907349, 0.0, 3.0) to '(1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '365' from location (1.46000003814697, 0.0, 3.0) to '(1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '366' from location (1.44000005722046, 0.0, 3.0) to '(1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '367' from location (1.41999995708466, 0.0, 3.0) to '(1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '368' from location (1.39999997615814, 0.0, 3.0) to '(1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '369' from location (1.37999999523163, 0.0, 3.0) to '(1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '370' from location (1.36000001430511, 0.0, 3.0) to '(1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '371' from location (1.3400000333786, 0.0, 3.0) to '(1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '372' from location (1.32000005245209, 0.0, 3.0) to '(1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '373' from location (1.29999995231628, 0.0, 3.0) to '(1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '374' from location (1.27999997138977, 0.0, 3.0) to '(1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '375' from location (1.25999999046326, 0.0, 3.0) to '(1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '376' from location (1.24000000953674, 0.0, 3.0) to '(1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '377' from location (1.22000002861023, 0.0, 3.0) to '(1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '378' from location (1.20000004768372, 0.0, 3.0) to '(1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '379' from location (1.17999994754791, 0.0, 3.0) to '(1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '380' from location (1.1599999666214, 0.0, 3.0) to '(1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '381' from location (1.13999998569489, 0.0, 3.0) to '(1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '382' from location (1.12000000476837, 0.0, 3.0) to '(1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '383' from location (1.10000002384186, 0.0, 3.0) to '(1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '384' from location (1.08000004291534, 0.0, 3.0) to '(1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '385' from location (1.05999994277954, 0.0, 3.0) to '(1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '386' from location (1.03999996185303, 0.0, 3.0) to '(1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '387' from location (1.01999998092651, 0.0, 3.0) to '(1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '388' from location (1.0, 0.0, 3.0) to '(1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '389' from location (0.980000019073486, 0.0, 3.0) to '(0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '390' from location (0.959999978542328, 0.0, 3.0) to '(0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '391' from location (0.939999997615814, 0.0, 3.0) to '(0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '392' from location (0.920000016689301, 0.0, 3.0) to '(0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '393' from location (0.879999995231628, 0.0, 3.0) to '(0.879999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '394' from location (0.860000014305115, 0.0, 3.0) to '(0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '395' from location (0.839999973773956, 0.0, 3.0) to '(0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '396' from location (0.819999992847443, 0.0, 3.0) to '(0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '397' from location (0.800000011920929, 0.0, 3.0) to '(0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '398' from location (0.779999971389771, 0.0, 3.0) to '(0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '399' from location (0.759999990463257, 0.0, 3.0) to '(0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '400' from location (0.740000009536743, 0.0, 3.0) to '(0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '401' from location (0.720000028610229, 0.0, 3.0) to '(0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '402' from location (0.699999988079071, 0.0, 3.0) to '(0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '403' from location (0.680000007152557, 0.0, 3.0) to '(0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '404' from location (0.660000026226044, 0.0, 3.0) to '(0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '405' from location (0.639999985694885, 0.0, 3.0) to '(0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '406' from location (0.620000004768372, 0.0, 3.0) to '(0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '407' from location (0.600000023841858, 0.0, 3.0) to '(0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '408' from location (0.579999983310699, 0.0, 3.0) to '(0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '409' from location (0.560000002384186, 0.0, 3.0) to '(0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '410' from location (0.540000021457672, 0.0, 3.0) to '(0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '411' from location (0.519999980926514, 0.0, 3.0) to '(0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '412' from location (0.5, 0.0, 3.0) to '(0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '413' from location (0.479999989271164, 0.0, 3.0) to '(0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '414' from location (0.46000000834465, 0.0, 3.0) to '(0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '415' from location (0.439999997615814, 0.0, 3.0) to '(0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '416' from location (0.419999986886978, 0.0, 3.0) to '(0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '417' from location (0.400000005960464, 0.0, 3.0) to '(0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '418' from location (0.379999995231628, 0.0, 3.0) to '(0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '419' from location (0.360000014305115, 0.0, 3.0) to '(0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '420' from location (0.340000003576279, 0.0, 3.0) to '(0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '421' from location (0.319999992847443, 0.0, 3.0) to '(0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '749' from location (-0.920000016689301, 0.0, 3.0) to '(-0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '750' from location (-0.939999997615814, 0.0, 3.0) to '(-0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '751' from location (-0.959999978542328, 0.0, 3.0) to '(-0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '752' from location (-0.980000019073486, 0.0, 3.0) to '(-0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '753' from location (-1.0, 0.0, 3.0) to '(-1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '754' from location (-1.01999998092651, 0.0, 3.0) to '(-1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '755' from location (-1.03999996185303, 0.0, 3.0) to '(-1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '756' from location (-1.05999994277954, 0.0, 3.0) to '(-1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '757' from location (-1.08000004291534, 0.0, 3.0) to '(-1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '758' from location (-1.10000002384186, 0.0, 3.0) to '(-1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '759' from location (-1.12000000476837, 0.0, 3.0) to '(-1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '760' from location (-1.13999998569489, 0.0, 3.0) to '(-1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '761' from location (-1.1599999666214, 0.0, 3.0) to '(-1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '762' from location (-1.17999994754791, 0.0, 3.0) to '(-1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '763' from location (-1.20000004768372, 0.0, 3.0) to '(-1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '764' from location (-1.22000002861023, 0.0, 3.0) to '(-1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '765' from location (-1.24000000953674, 0.0, 3.0) to '(-1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '766' from location (-1.25999999046326, 0.0, 3.0) to '(-1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '767' from location (-1.27999997138977, 0.0, 3.0) to '(-1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '768' from location (-1.29999995231628, 0.0, 3.0) to '(-1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '769' from location (-1.32000005245209, 0.0, 3.0) to '(-1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '770' from location (-1.3400000333786, 0.0, 3.0) to '(-1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '771' from location (-1.36000001430511, 0.0, 3.0) to '(-1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '772' from location (-1.37999999523163, 0.0, 3.0) to '(-1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '773' from location (-1.39999997615814, 0.0, 3.0) to '(-1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '774' from location (-1.41999995708466, 0.0, 3.0) to '(-1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '775' from location (-1.44000005722046, 0.0, 3.0) to '(-1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '776' from location (-1.46000003814697, 0.0, 3.0) to '(-1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '777' from location (-1.48000001907349, 0.0, 3.0) to '(-1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1134' from location (0.280000001192093, 0.0, 3.0) to '(0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1135' from location (0.259999990463257, 0.0, 3.0) to '(0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1136' from location (0.239999994635582, 0.0, 3.0) to '(0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1137' from location (0.219999998807907, 0.0, 3.0) to '(0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1138' from location (0.200000002980232, 0.0, 3.0) to '(0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1139' from location (0.180000007152557, 0.0, 3.0) to '(0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1140' from location (0.159999996423721, 0.0, 3.0) to '(0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1141' from location (0.140000000596046, 0.0, 3.0) to '(0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1142' from location (0.119999997317791, 0.0, 3.0) to '(0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1143' from location (0.100000001490116, 0.0, 3.0) to '(0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1144' from location (0.0799999982118607, 0.0, 3.0) to '(0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1145' from location (0.0599999986588955, 0.0, 3.0) to '(0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1146' from location (0.0399999991059303, 0.0, 3.0) to '(0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1147' from location (0.0199999995529652, 0.0, 3.0) to '(0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1148' from location (-4.44089209850063e-16, 0.0, 3.0) to '(-4.44089209850063e-16, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1149' from location (-0.0199999995529652, 0.0, 3.0) to '(-0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1150' from location (-0.0399999991059303, 0.0, 3.0) to '(-0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1151' from location (-0.0599999986588955, 0.0, 3.0) to '(-0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1152' from location (-0.0799999982118607, 0.0, 3.0) to '(-0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1153' from location (-0.100000001490116, 0.0, 3.0) to '(-0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1154' from location (-0.119999997317791, 0.0, 3.0) to '(-0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1155' from location (-0.140000000596046, 0.0, 3.0) to '(-0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1156' from location (-0.159999996423721, 0.0, 3.0) to '(-0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1157' from location (-0.180000007152557, 0.0, 3.0) to '(-0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1158' from location (-0.200000002980232, 0.0, 3.0) to '(-0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1159' from location (-0.219999998807907, 0.0, 3.0) to '(-0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1160' from location (-0.239999994635582, 0.0, 3.0) to '(-0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1161' from location (-0.259999990463257, 0.0, 3.0) to '(-0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1162' from location (-0.280000001192093, 0.0, 3.0) to '(-0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1163' from location (-0.319999992847443, 0.0, 3.0) to '(-0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1164' from location (-0.340000003576279, 0.0, 3.0) to '(-0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1165' from location (-0.360000014305115, 0.0, 3.0) to '(-0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1166' from location (-0.379999995231628, 0.0, 3.0) to '(-0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1167' from location (-0.400000005960464, 0.0, 3.0) to '(-0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1168' from location (-0.419999986886978, 0.0, 3.0) to '(-0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1169' from location (-0.439999997615814, 0.0, 3.0) to '(-0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1170' from location (-0.46000000834465, 0.0, 3.0) to '(-0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1171' from location (-0.479999989271164, 0.0, 3.0) to '(-0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1172' from location (-0.5, 0.0, 3.0) to '(-0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1173' from location (-0.519999980926514, 0.0, 3.0) to '(-0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1174' from location (-0.540000021457672, 0.0, 3.0) to '(-0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1175' from location (-0.560000002384186, 0.0, 3.0) to '(-0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1176' from location (-0.579999983310699, 0.0, 3.0) to '(-0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1177' from location (-0.600000023841858, 0.0, 3.0) to '(-0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1178' from location (-0.620000004768372, 0.0, 3.0) to '(-0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1179' from location (-0.639999985694885, 0.0, 3.0) to '(-0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1180' from location (-0.660000026226044, 0.0, 3.0) to '(-0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1181' from location (-0.680000007152557, 0.0, 3.0) to '(-0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1182' from location (-0.699999988079071, 0.0, 3.0) to '(-0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1183' from location (-0.720000028610229, 0.0, 3.0) to '(-0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1184' from location (-0.740000009536743, 0.0, 3.0) to '(-0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1185' from location (-0.759999990463257, 0.0, 3.0) to '(-0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1186' from location (-0.779999971389771, 0.0, 3.0) to '(-0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1187' from location (-0.800000011920929, 0.0, 3.0) to '(-0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1188' from location (-0.819999992847443, 0.0, 3.0) to '(-0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1189' from location (-0.839999973773956, 0.0, 3.0) to '(-0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1190' from location (-0.860000014305115, 0.0, 3.0) to '(-0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1191' from location (-0.879999995231628, 0.0, 3.0) to '(-0.879999995231628, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 474 elements.
#: Assigned section 'local-thickness' to 474 elements.
#* TypeError: keyword error on nam
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 1278, in <module>
#*     equation_sets(model, 'Web{}'.format(index), 
#* 'Web{}-Follower'.format(index), 'Web{}-Main'.format(index), linked_dof= [2])
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 829, in equation_sets
#*     (-1.0, set_two, dof)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 12:23:30
#: The model "eigen" has been created.
#: [move_closest_nodes_to_axis] Moved node '2' from location (0.899999976158142, 0.0, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '3' from location (1.5, 0.0, 0.0) to '(1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '6' from location (0.300000011920929, 0.0, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '7' from location (-0.899999976158142, 0.0, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '10' from location (-1.5, 0.0, 0.0) to '(-1.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '12' from location (-0.300000011920929, 0.0, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '186' from location (0.920000016689301, 0.0, 0.0) to '(0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '187' from location (0.939999997615814, 0.0, 0.0) to '(0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '188' from location (0.959999978542328, 0.0, 0.0) to '(0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '189' from location (0.980000019073486, 0.0, 0.0) to '(0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '190' from location (1.0, 0.0, 0.0) to '(1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '191' from location (1.01999998092651, 0.0, 0.0) to '(1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '192' from location (1.03999996185303, 0.0, 0.0) to '(1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '193' from location (1.05999994277954, 0.0, 0.0) to '(1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '194' from location (1.08000004291534, 0.0, 0.0) to '(1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '195' from location (1.10000002384186, 0.0, 0.0) to '(1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '196' from location (1.12000000476837, 0.0, 0.0) to '(1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '197' from location (1.13999998569489, 0.0, 0.0) to '(1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '198' from location (1.1599999666214, 0.0, 0.0) to '(1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '199' from location (1.17999994754791, 0.0, 0.0) to '(1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '200' from location (1.20000004768372, 0.0, 0.0) to '(1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '201' from location (1.22000002861023, 0.0, 0.0) to '(1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '202' from location (1.24000000953674, 0.0, 0.0) to '(1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '203' from location (1.25999999046326, 0.0, 0.0) to '(1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '204' from location (1.27999997138977, 0.0, 0.0) to '(1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '205' from location (1.29999995231628, 0.0, 0.0) to '(1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '206' from location (1.32000005245209, 0.0, 0.0) to '(1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '207' from location (1.3400000333786, 0.0, 0.0) to '(1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '208' from location (1.36000001430511, 0.0, 0.0) to '(1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '209' from location (1.37999999523163, 0.0, 0.0) to '(1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '210' from location (1.39999997615814, 0.0, 0.0) to '(1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '211' from location (1.41999995708466, 0.0, 0.0) to '(1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '212' from location (1.44000005722046, 0.0, 0.0) to '(1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '213' from location (1.46000003814697, 0.0, 0.0) to '(1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '214' from location (1.48000001907349, 0.0, 0.0) to '(1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '571' from location (0.319999992847443, 0.0, 0.0) to '(0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '572' from location (0.340000003576279, 0.0, 0.0) to '(0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '573' from location (0.360000014305115, 0.0, 0.0) to '(0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '574' from location (0.379999995231628, 0.0, 0.0) to '(0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '575' from location (0.400000005960464, 0.0, 0.0) to '(0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '576' from location (0.419999986886978, 0.0, 0.0) to '(0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '577' from location (0.439999997615814, 0.0, 0.0) to '(0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '578' from location (0.46000000834465, 0.0, 0.0) to '(0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '579' from location (0.479999989271164, 0.0, 0.0) to '(0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '580' from location (0.5, 0.0, 0.0) to '(0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '581' from location (0.519999980926514, 0.0, 0.0) to '(0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '582' from location (0.540000021457672, 0.0, 0.0) to '(0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '583' from location (0.560000002384186, 0.0, 0.0) to '(0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '584' from location (0.579999983310699, 0.0, 0.0) to '(0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '585' from location (0.600000023841858, 0.0, 0.0) to '(0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '586' from location (0.620000004768372, 0.0, 0.0) to '(0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '587' from location (0.639999985694885, 0.0, 0.0) to '(0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '588' from location (0.660000026226044, 0.0, 0.0) to '(0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '589' from location (0.680000007152557, 0.0, 0.0) to '(0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '590' from location (0.699999988079071, 0.0, 0.0) to '(0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '591' from location (0.720000028610229, 0.0, 0.0) to '(0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '592' from location (0.740000009536743, 0.0, 0.0) to '(0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '593' from location (0.759999990463257, 0.0, 0.0) to '(0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '594' from location (0.779999971389771, 0.0, 0.0) to '(0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '595' from location (0.800000011920929, 0.0, 0.0) to '(0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '596' from location (0.819999992847443, 0.0, 0.0) to '(0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '597' from location (0.839999973773956, 0.0, 0.0) to '(0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '598' from location (0.860000014305115, 0.0, 0.0) to '(0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '599' from location (0.879999995231628, 0.0, 0.0) to '(0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '927' from location (-1.48000001907349, 0.0, 0.0) to '(-1.48000001907349, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '928' from location (-1.46000003814697, 0.0, 0.0) to '(-1.46000003814697, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '929' from location (-1.44000005722046, 0.0, 0.0) to '(-1.44000005722046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '930' from location (-1.41999995708466, 0.0, 0.0) to '(-1.41999995708466, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '931' from location (-1.39999997615814, 0.0, 0.0) to '(-1.39999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '932' from location (-1.37999999523163, 0.0, 0.0) to '(-1.37999999523163, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '933' from location (-1.36000001430511, 0.0, 0.0) to '(-1.36000001430511, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '934' from location (-1.3400000333786, 0.0, 0.0) to '(-1.3400000333786, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '935' from location (-1.32000005245209, 0.0, 0.0) to '(-1.32000005245209, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '936' from location (-1.29999995231628, 0.0, 0.0) to '(-1.29999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '937' from location (-1.27999997138977, 0.0, 0.0) to '(-1.27999997138977, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '938' from location (-1.25999999046326, 0.0, 0.0) to '(-1.25999999046326, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '939' from location (-1.24000000953674, 0.0, 0.0) to '(-1.24000000953674, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '940' from location (-1.22000002861023, 0.0, 0.0) to '(-1.22000002861023, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '941' from location (-1.20000004768372, 0.0, 0.0) to '(-1.20000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '942' from location (-1.17999994754791, 0.0, 0.0) to '(-1.17999994754791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '943' from location (-1.1599999666214, 0.0, 0.0) to '(-1.1599999666214, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '944' from location (-1.13999998569489, 0.0, 0.0) to '(-1.13999998569489, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '945' from location (-1.12000000476837, 0.0, 0.0) to '(-1.12000000476837, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '946' from location (-1.10000002384186, 0.0, 0.0) to '(-1.10000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '947' from location (-1.08000004291534, 0.0, 0.0) to '(-1.08000004291534, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '948' from location (-1.05999994277954, 0.0, 0.0) to '(-1.05999994277954, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '949' from location (-1.03999996185303, 0.0, 0.0) to '(-1.03999996185303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '950' from location (-1.01999998092651, 0.0, 0.0) to '(-1.01999998092651, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '951' from location (-1.0, 0.0, 0.0) to '(-1.0, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '952' from location (-0.980000019073486, 0.0, 0.0) to '(-0.980000019073486, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '953' from location (-0.959999978542328, 0.0, 0.0) to '(-0.959999978542328, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '954' from location (-0.939999997615814, 0.0, 0.0) to '(-0.939999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '955' from location (-0.920000016689301, 0.0, 0.0) to '(-0.920000016689301, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1105' from location (-0.280000001192093, 0.0, 0.0) to '(-0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1106' from location (-0.259999990463257, 0.0, 0.0) to '(-0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1107' from location (-0.239999994635582, 0.0, 0.0) to '(-0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1108' from location (-0.219999998807907, 0.0, 0.0) to '(-0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1109' from location (-0.200000002980232, 0.0, 0.0) to '(-0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1110' from location (-0.180000007152557, 0.0, 0.0) to '(-0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1111' from location (-0.159999996423721, 0.0, 0.0) to '(-0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1112' from location (-0.140000000596046, 0.0, 0.0) to '(-0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1113' from location (-0.119999997317791, 0.0, 0.0) to '(-0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1114' from location (-0.100000001490116, 0.0, 0.0) to '(-0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1115' from location (-0.0799999982118607, 0.0, 0.0) to '(-0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1116' from location (-0.0599999986588955, 0.0, 0.0) to '(-0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1117' from location (-0.0399999991059303, 0.0, 0.0) to '(-0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1118' from location (-0.0199999995529652, 0.0, 0.0) to '(-0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1119' from location (2.22044604925031e-16, 0.0, 0.0) to '(2.22044604925031e-16, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1120' from location (0.0199999995529652, 0.0, 0.0) to '(0.0199999995529652, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1121' from location (0.0399999991059303, 0.0, 0.0) to '(0.0399999991059303, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1122' from location (0.0599999986588955, 0.0, 0.0) to '(0.0599999986588955, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1123' from location (0.0799999982118607, 0.0, 0.0) to '(0.0799999982118607, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1124' from location (0.100000001490116, 0.0, 0.0) to '(0.100000001490116, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1125' from location (0.119999997317791, 0.0, 0.0) to '(0.119999997317791, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1126' from location (0.140000000596046, 0.0, 0.0) to '(0.140000000596046, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1127' from location (0.159999996423721, 0.0, 0.0) to '(0.159999996423721, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1128' from location (0.180000007152557, 0.0, 0.0) to '(0.180000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1129' from location (0.200000002980232, 0.0, 0.0) to '(0.200000002980232, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1130' from location (0.219999998807907, 0.0, 0.0) to '(0.219999998807907, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1131' from location (0.239999994635582, 0.0, 0.0) to '(0.239999994635582, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1132' from location (0.259999990463257, 0.0, 0.0) to '(0.259999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1133' from location (0.280000001192093, 0.0, 0.0) to '(0.280000001192093, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1192' from location (-0.879999995231628, 0.0, 0.0) to '(-0.879999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1193' from location (-0.860000014305115, 0.0, 0.0) to '(-0.860000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1194' from location (-0.839999973773956, 0.0, 0.0) to '(-0.839999973773956, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1195' from location (-0.819999992847443, 0.0, 0.0) to '(-0.819999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1196' from location (-0.800000011920929, 0.0, 0.0) to '(-0.800000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1197' from location (-0.779999971389771, 0.0, 0.0) to '(-0.779999971389771, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1198' from location (-0.759999990463257, 0.0, 0.0) to '(-0.759999990463257, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1199' from location (-0.740000009536743, 0.0, 0.0) to '(-0.740000009536743, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1200' from location (-0.720000028610229, 0.0, 0.0) to '(-0.720000028610229, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1201' from location (-0.699999988079071, 0.0, 0.0) to '(-0.699999988079071, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1202' from location (-0.680000007152557, 0.0, 0.0) to '(-0.680000007152557, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1203' from location (-0.660000026226044, 0.0, 0.0) to '(-0.660000026226044, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1204' from location (-0.639999985694885, 0.0, 0.0) to '(-0.639999985694885, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1205' from location (-0.620000004768372, 0.0, 0.0) to '(-0.620000004768372, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1206' from location (-0.600000023841858, 0.0, 0.0) to '(-0.600000023841858, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1207' from location (-0.579999983310699, 0.0, 0.0) to '(-0.579999983310699, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1208' from location (-0.560000002384186, 0.0, 0.0) to '(-0.560000002384186, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1209' from location (-0.540000021457672, 0.0, 0.0) to '(-0.540000021457672, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1210' from location (-0.519999980926514, 0.0, 0.0) to '(-0.519999980926514, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1211' from location (-0.5, 0.0, 0.0) to '(-0.5, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1212' from location (-0.479999989271164, 0.0, 0.0) to '(-0.479999989271164, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1213' from location (-0.46000000834465, 0.0, 0.0) to '(-0.46000000834465, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1214' from location (-0.439999997615814, 0.0, 0.0) to '(-0.439999997615814, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1215' from location (-0.419999986886978, 0.0, 0.0) to '(-0.419999986886978, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1216' from location (-0.400000005960464, 0.0, 0.0) to '(-0.400000005960464, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1217' from location (-0.379999995231628, 0.0, 0.0) to '(-0.379999995231628, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1218' from location (-0.360000014305115, 0.0, 0.0) to '(-0.360000014305115, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1219' from location (-0.340000003576279, 0.0, 0.0) to '(-0.340000003576279, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1220' from location (-0.319999992847443, 0.0, 0.0) to '(-0.319999992847443, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1' from location (0.899999976158142, 0.0, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '4' from location (1.5, 0.0, 3.0) to '(1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '5' from location (0.300000011920929, 0.0, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '8' from location (-0.899999976158142, 0.0, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '9' from location (-1.5, 0.0, 3.0) to '(-1.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '11' from location (-0.300000011920929, 0.0, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '364' from location (1.48000001907349, 0.0, 3.0) to '(1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '365' from location (1.46000003814697, 0.0, 3.0) to '(1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '366' from location (1.44000005722046, 0.0, 3.0) to '(1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '367' from location (1.41999995708466, 0.0, 3.0) to '(1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '368' from location (1.39999997615814, 0.0, 3.0) to '(1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '369' from location (1.37999999523163, 0.0, 3.0) to '(1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '370' from location (1.36000001430511, 0.0, 3.0) to '(1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '371' from location (1.3400000333786, 0.0, 3.0) to '(1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '372' from location (1.32000005245209, 0.0, 3.0) to '(1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '373' from location (1.29999995231628, 0.0, 3.0) to '(1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '374' from location (1.27999997138977, 0.0, 3.0) to '(1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '375' from location (1.25999999046326, 0.0, 3.0) to '(1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '376' from location (1.24000000953674, 0.0, 3.0) to '(1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '377' from location (1.22000002861023, 0.0, 3.0) to '(1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '378' from location (1.20000004768372, 0.0, 3.0) to '(1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '379' from location (1.17999994754791, 0.0, 3.0) to '(1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '380' from location (1.1599999666214, 0.0, 3.0) to '(1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '381' from location (1.13999998569489, 0.0, 3.0) to '(1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '382' from location (1.12000000476837, 0.0, 3.0) to '(1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '383' from location (1.10000002384186, 0.0, 3.0) to '(1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '384' from location (1.08000004291534, 0.0, 3.0) to '(1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '385' from location (1.05999994277954, 0.0, 3.0) to '(1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '386' from location (1.03999996185303, 0.0, 3.0) to '(1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '387' from location (1.01999998092651, 0.0, 3.0) to '(1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '388' from location (1.0, 0.0, 3.0) to '(1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '389' from location (0.980000019073486, 0.0, 3.0) to '(0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '390' from location (0.959999978542328, 0.0, 3.0) to '(0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '391' from location (0.939999997615814, 0.0, 3.0) to '(0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '392' from location (0.920000016689301, 0.0, 3.0) to '(0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '393' from location (0.879999995231628, 0.0, 3.0) to '(0.879999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '394' from location (0.860000014305115, 0.0, 3.0) to '(0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '395' from location (0.839999973773956, 0.0, 3.0) to '(0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '396' from location (0.819999992847443, 0.0, 3.0) to '(0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '397' from location (0.800000011920929, 0.0, 3.0) to '(0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '398' from location (0.779999971389771, 0.0, 3.0) to '(0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '399' from location (0.759999990463257, 0.0, 3.0) to '(0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '400' from location (0.740000009536743, 0.0, 3.0) to '(0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '401' from location (0.720000028610229, 0.0, 3.0) to '(0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '402' from location (0.699999988079071, 0.0, 3.0) to '(0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '403' from location (0.680000007152557, 0.0, 3.0) to '(0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '404' from location (0.660000026226044, 0.0, 3.0) to '(0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '405' from location (0.639999985694885, 0.0, 3.0) to '(0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '406' from location (0.620000004768372, 0.0, 3.0) to '(0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '407' from location (0.600000023841858, 0.0, 3.0) to '(0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '408' from location (0.579999983310699, 0.0, 3.0) to '(0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '409' from location (0.560000002384186, 0.0, 3.0) to '(0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '410' from location (0.540000021457672, 0.0, 3.0) to '(0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '411' from location (0.519999980926514, 0.0, 3.0) to '(0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '412' from location (0.5, 0.0, 3.0) to '(0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '413' from location (0.479999989271164, 0.0, 3.0) to '(0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '414' from location (0.46000000834465, 0.0, 3.0) to '(0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '415' from location (0.439999997615814, 0.0, 3.0) to '(0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '416' from location (0.419999986886978, 0.0, 3.0) to '(0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '417' from location (0.400000005960464, 0.0, 3.0) to '(0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '418' from location (0.379999995231628, 0.0, 3.0) to '(0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '419' from location (0.360000014305115, 0.0, 3.0) to '(0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '420' from location (0.340000003576279, 0.0, 3.0) to '(0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '421' from location (0.319999992847443, 0.0, 3.0) to '(0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '749' from location (-0.920000016689301, 0.0, 3.0) to '(-0.920000016689301, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '750' from location (-0.939999997615814, 0.0, 3.0) to '(-0.939999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '751' from location (-0.959999978542328, 0.0, 3.0) to '(-0.959999978542328, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '752' from location (-0.980000019073486, 0.0, 3.0) to '(-0.980000019073486, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '753' from location (-1.0, 0.0, 3.0) to '(-1.0, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '754' from location (-1.01999998092651, 0.0, 3.0) to '(-1.01999998092651, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '755' from location (-1.03999996185303, 0.0, 3.0) to '(-1.03999996185303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '756' from location (-1.05999994277954, 0.0, 3.0) to '(-1.05999994277954, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '757' from location (-1.08000004291534, 0.0, 3.0) to '(-1.08000004291534, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '758' from location (-1.10000002384186, 0.0, 3.0) to '(-1.10000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '759' from location (-1.12000000476837, 0.0, 3.0) to '(-1.12000000476837, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '760' from location (-1.13999998569489, 0.0, 3.0) to '(-1.13999998569489, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '761' from location (-1.1599999666214, 0.0, 3.0) to '(-1.1599999666214, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '762' from location (-1.17999994754791, 0.0, 3.0) to '(-1.17999994754791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '763' from location (-1.20000004768372, 0.0, 3.0) to '(-1.20000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '764' from location (-1.22000002861023, 0.0, 3.0) to '(-1.22000002861023, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '765' from location (-1.24000000953674, 0.0, 3.0) to '(-1.24000000953674, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '766' from location (-1.25999999046326, 0.0, 3.0) to '(-1.25999999046326, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '767' from location (-1.27999997138977, 0.0, 3.0) to '(-1.27999997138977, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '768' from location (-1.29999995231628, 0.0, 3.0) to '(-1.29999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '769' from location (-1.32000005245209, 0.0, 3.0) to '(-1.32000005245209, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '770' from location (-1.3400000333786, 0.0, 3.0) to '(-1.3400000333786, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '771' from location (-1.36000001430511, 0.0, 3.0) to '(-1.36000001430511, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '772' from location (-1.37999999523163, 0.0, 3.0) to '(-1.37999999523163, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '773' from location (-1.39999997615814, 0.0, 3.0) to '(-1.39999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '774' from location (-1.41999995708466, 0.0, 3.0) to '(-1.41999995708466, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '775' from location (-1.44000005722046, 0.0, 3.0) to '(-1.44000005722046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '776' from location (-1.46000003814697, 0.0, 3.0) to '(-1.46000003814697, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '777' from location (-1.48000001907349, 0.0, 3.0) to '(-1.48000001907349, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1134' from location (0.280000001192093, 0.0, 3.0) to '(0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1135' from location (0.259999990463257, 0.0, 3.0) to '(0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1136' from location (0.239999994635582, 0.0, 3.0) to '(0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1137' from location (0.219999998807907, 0.0, 3.0) to '(0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1138' from location (0.200000002980232, 0.0, 3.0) to '(0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1139' from location (0.180000007152557, 0.0, 3.0) to '(0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1140' from location (0.159999996423721, 0.0, 3.0) to '(0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1141' from location (0.140000000596046, 0.0, 3.0) to '(0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1142' from location (0.119999997317791, 0.0, 3.0) to '(0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1143' from location (0.100000001490116, 0.0, 3.0) to '(0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1144' from location (0.0799999982118607, 0.0, 3.0) to '(0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1145' from location (0.0599999986588955, 0.0, 3.0) to '(0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1146' from location (0.0399999991059303, 0.0, 3.0) to '(0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1147' from location (0.0199999995529652, 0.0, 3.0) to '(0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1148' from location (-4.44089209850063e-16, 0.0, 3.0) to '(-4.44089209850063e-16, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1149' from location (-0.0199999995529652, 0.0, 3.0) to '(-0.0199999995529652, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1150' from location (-0.0399999991059303, 0.0, 3.0) to '(-0.0399999991059303, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1151' from location (-0.0599999986588955, 0.0, 3.0) to '(-0.0599999986588955, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1152' from location (-0.0799999982118607, 0.0, 3.0) to '(-0.0799999982118607, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1153' from location (-0.100000001490116, 0.0, 3.0) to '(-0.100000001490116, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1154' from location (-0.119999997317791, 0.0, 3.0) to '(-0.119999997317791, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1155' from location (-0.140000000596046, 0.0, 3.0) to '(-0.140000000596046, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1156' from location (-0.159999996423721, 0.0, 3.0) to '(-0.159999996423721, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1157' from location (-0.180000007152557, 0.0, 3.0) to '(-0.180000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1158' from location (-0.200000002980232, 0.0, 3.0) to '(-0.200000002980232, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1159' from location (-0.219999998807907, 0.0, 3.0) to '(-0.219999998807907, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1160' from location (-0.239999994635582, 0.0, 3.0) to '(-0.239999994635582, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1161' from location (-0.259999990463257, 0.0, 3.0) to '(-0.259999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1162' from location (-0.280000001192093, 0.0, 3.0) to '(-0.280000001192093, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1163' from location (-0.319999992847443, 0.0, 3.0) to '(-0.319999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1164' from location (-0.340000003576279, 0.0, 3.0) to '(-0.340000003576279, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1165' from location (-0.360000014305115, 0.0, 3.0) to '(-0.360000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1166' from location (-0.379999995231628, 0.0, 3.0) to '(-0.379999995231628, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1167' from location (-0.400000005960464, 0.0, 3.0) to '(-0.400000005960464, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1168' from location (-0.419999986886978, 0.0, 3.0) to '(-0.419999986886978, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1169' from location (-0.439999997615814, 0.0, 3.0) to '(-0.439999997615814, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1170' from location (-0.46000000834465, 0.0, 3.0) to '(-0.46000000834465, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1171' from location (-0.479999989271164, 0.0, 3.0) to '(-0.479999989271164, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1172' from location (-0.5, 0.0, 3.0) to '(-0.5, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1173' from location (-0.519999980926514, 0.0, 3.0) to '(-0.519999980926514, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1174' from location (-0.540000021457672, 0.0, 3.0) to '(-0.540000021457672, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1175' from location (-0.560000002384186, 0.0, 3.0) to '(-0.560000002384186, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1176' from location (-0.579999983310699, 0.0, 3.0) to '(-0.579999983310699, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1177' from location (-0.600000023841858, 0.0, 3.0) to '(-0.600000023841858, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1178' from location (-0.620000004768372, 0.0, 3.0) to '(-0.620000004768372, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1179' from location (-0.639999985694885, 0.0, 3.0) to '(-0.639999985694885, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1180' from location (-0.660000026226044, 0.0, 3.0) to '(-0.660000026226044, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1181' from location (-0.680000007152557, 0.0, 3.0) to '(-0.680000007152557, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1182' from location (-0.699999988079071, 0.0, 3.0) to '(-0.699999988079071, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1183' from location (-0.720000028610229, 0.0, 3.0) to '(-0.720000028610229, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1184' from location (-0.740000009536743, 0.0, 3.0) to '(-0.740000009536743, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1185' from location (-0.759999990463257, 0.0, 3.0) to '(-0.759999990463257, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1186' from location (-0.779999971389771, 0.0, 3.0) to '(-0.779999971389771, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1187' from location (-0.800000011920929, 0.0, 3.0) to '(-0.800000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1188' from location (-0.819999992847443, 0.0, 3.0) to '(-0.819999992847443, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1189' from location (-0.839999973773956, 0.0, 3.0) to '(-0.839999973773956, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1190' from location (-0.860000014305115, 0.0, 3.0) to '(-0.860000014305115, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1191' from location (-0.879999995231628, 0.0, 3.0) to '(-0.879999995231628, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 474 elements.
#: Assigned section 'local-thickness' to 474 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
a = mdb.models['eigen'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
session.viewports['Viewport: 1'].view.setProjection(projection=PERSPECTIVE)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.48548, 
    farPlane=9.86708, width=2.85945, height=3.34969, cameraPosition=(0.731894, 
    2.91726, 8.20124), cameraUpVector=(0.379009, 0.680491, -0.627124))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.53938, 
    farPlane=10.6884, width=2.49804, height=2.92632, cameraPosition=(-4.77486, 
    -3.51954, 6.30877), cameraUpVector=(0.288245, 0.915963, 0.279152), 
    cameraTarget=(0.11954, 0.0429103, 0.00328833))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.49551, 
    farPlane=10.7583, width=2.48128, height=2.90669, cameraPosition=(-1.4533, 
    -7.69717, 3.67754), cameraUpVector=(-0.0951291, 0.714859, 0.692768), 
    cameraTarget=(0.070521, 0.104563, 0.0421196))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.03567, 
    farPlane=11.1653, width=2.30562, height=2.70092, cameraPosition=(-5.19363, 
    -6.53316, 2.13982), cameraUpVector=(0.160632, 0.564967, 0.809327), 
    cameraTarget=(0.119995, 0.0891665, 0.0624593))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.20401, 
    farPlane=10.9892, width=2.36992, height=2.77625, cameraPosition=(-6.49636, 
    -4.38587, 3.59312), cameraUpVector=(0.443398, 0.556113, 0.702948), 
    cameraTarget=(0.14128, 0.0540816, 0.0387136))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.31523, 
    farPlane=10.8977, width=2.4124, height=2.82602, cameraPosition=(-5.36603, 
    -4.81202, 4.7661), cameraUpVector=(0.59552, 0.522017, 0.610618), 
    cameraTarget=(0.122293, 0.06124, 0.0190103))
