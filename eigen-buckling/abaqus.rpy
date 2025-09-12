# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Fri Sep 12 11:04:28 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=244.630065917969, 
    height=228.433334350586)
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
#: 
#: 
#: Start Time: 2025-09-12 11:04:36
#: The model "eigen" has been created.
#: []
#: [assign_section] Assigned section 't-0.01' to 0 face(s) on part 'panel' using sets method.
#: []
#: [assign_section] Assigned section 't-0.0078' to 0 face(s) on part 'panel' using sets method.
#: []
#: [assign_section] Assigned section 't-0.004' to 0 face(s) on part 'panel' using sets method.
p1 = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:08:58
#: The model "eigen" has been created.
#* AttributeError: 'MeshElementArray' object has no attribute 'getFromFaces'
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 281, in <module>
#*     plate_elems = p.elements.getFromFaces(plate_faces)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:12:36
#: The model "eigen" has been created.
#* KeyError: Unknown key FlangeFaces
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 212, in <module>
#*     mesh_from_faces(model.parts['plate'], face_seed_map)
#* File 
#* "C:\Users\sbonnell\Desktop\lase\projects\ultimate-strength\eigen-buckling\utils\mesh_utilities.py", 
#* line 53, in mesh_from_faces
#*     for f in part.sets[face_set_name].faces:
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:14:24
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['eigen'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p = mdb.models['eigen'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:15:25
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.025000000372529, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1684' from location (0.300000011920929, 0.025000000372529, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2143' from location (0.899999976158142, 0.025000000372529, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2602' from location (-0.300000011920929, 0.025000000372529, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1224' from location (-0.899999976158142, 0.025000000372529, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1683' from location (0.300000011920929, 0.025000000372529, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2142' from location (0.899999976158142, 0.025000000372529, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2601' from location (-0.300000011920929, 0.025000000372529, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#* ValueError: Set Load-Follower may not be used; If the first term uses a set 
#* consisting of a single point all the subsequent terms must use a single 
#* point set.
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 483, in <module>
#*     equation_sets(model, 'Load', 'Load-Main', 'Load-Follower', linked_dof= 
#* [1])
#* File 
#* "C:\Users\sbonnell\Desktop\lase\projects\ultimate-strength\eigen-buckling\utils\constraint_utilities.py", 
#* line 50, in equation_sets
#*     (-1.0, set_two, dof)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['eigen'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.78843, 
    farPlane=11.2504, width=2.64981, height=2.24464, viewOffsetX=-0.160345, 
    viewOffsetY=-0.184138)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:20:11
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.025000000372529, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1684' from location (0.300000011920929, 0.025000000372529, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2143' from location (0.899999976158142, 0.025000000372529, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2602' from location (-0.300000011920929, 0.025000000372529, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1224' from location (-0.899999976158142, 0.025000000372529, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1683' from location (0.300000011920929, 0.025000000372529, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2142' from location (0.899999976158142, 0.025000000372529, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2601' from location (-0.300000011920929, 0.025000000372529, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.75716, 
    farPlane=11.2817, width=3.1756, height=2.69003, viewOffsetX=-0.0855029, 
    viewOffsetY=-0.153335)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p = mdb.models['eigen'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['eigen'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.05032, 
    farPlane=10.9885, width=1.14788, height=0.972364, viewOffsetX=0.87202, 
    viewOffsetY=0.217654)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:20:35
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.025000000372529, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1684' from location (0.300000011920929, 0.025000000372529, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2143' from location (0.899999976158142, 0.025000000372529, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2602' from location (-0.300000011920929, 0.025000000372529, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1224' from location (-0.899999976158142, 0.025000000372529, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1683' from location (0.300000011920929, 0.025000000372529, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2142' from location (0.899999976158142, 0.025000000372529, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2601' from location (-0.300000011920929, 0.025000000372529, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['eigen'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:22:41
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.025000000372529, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1684' from location (0.300000011920929, 0.025000000372529, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2143' from location (0.899999976158142, 0.025000000372529, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2602' from location (-0.300000011920929, 0.025000000372529, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1224' from location (-0.899999976158142, 0.025000000372529, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1683' from location (0.300000011920929, 0.025000000372529, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2142' from location (0.899999976158142, 0.025000000372529, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2601' from location (-0.300000011920929, 0.025000000372529, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
#: Warning: The assembly is empty; therefore, an analysis 
#: cannot be performed on this model.
#: Error in job parametric-panel: NO STEP DEFINITION WAS FOUND
#: Job parametric-panel: Analysis Input File Processor aborted due to errors.
#: Error in job parametric-panel: Analysis Input File Processor exited with an error.
#: Job parametric-panel aborted due to errors.
a = mdb.models['eigen'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 11:23:32
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.025000000372529, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1684' from location (0.300000011920929, 0.025000000372529, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2143' from location (0.899999976158142, 0.025000000372529, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2602' from location (-0.300000011920929, 0.025000000372529, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1224' from location (-0.899999976158142, 0.025000000372529, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1683' from location (0.300000011920929, 0.025000000372529, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2142' from location (0.899999976158142, 0.025000000372529, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2601' from location (-0.300000011920929, 0.025000000372529, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
#: Abaqus Warning: The following input options are not supported by parallel execution of element operations: buckle. Only the solver will be executed in parallel for this analysis.
#: Job parametric-panel: Analysis Input File Processor completed successfully.
#: Job parametric-panel: Abaqus/Standard completed successfully.
#: Job parametric-panel completed successfully. 
session.viewports['Viewport: 1'].setValues(displayedObject=None)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/parametric-panel.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/parametric-panel.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          18
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.99271, 
    farPlane=10.4209, width=3.57473, height=3.12919, cameraPosition=(1.60872, 
    -3.91552, 7.66028), cameraUpVector=(0.274177, 0.960161, 0.0540223), 
    cameraTarget=(0.0758518, -0.0117384, -0.00889258))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.29882, 
    farPlane=11.0672, width=3.22001, height=2.81868, cameraPosition=(-2.88108, 
    -6.66162, 4.81815), cameraUpVector=(0.609287, 0.566506, 0.554834), 
    cameraTarget=(0.0934945, -0.000947434, 0.0022755))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.34487, 
    farPlane=10.9148, width=3.24355, height=2.83929, cameraPosition=(-6.65098, 
    -2.61355, 4.88973), cameraUpVector=(0.721166, 0.358898, 0.592547), 
    cameraTarget=(0.118677, -0.0279884, 0.00179732))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.54315, 
    farPlane=10.6963, width=3.34491, height=2.92802, cameraPosition=(-6.79937, 
    -1.18891, 5.21432), cameraUpVector=(0.781938, 0.302592, 0.544988), 
    cameraTarget=(0.120589, -0.0463416, -0.00238423))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.15693, 
    farPlane=11.1039, width=3.14747, height=2.75519, cameraPosition=(-7.14088, 
    -3.29627, 3.60508), cameraUpVector=(0.540825, 0.459954, 0.704238), 
    cameraTarget=(0.125394, -0.0166888, 0.0202594))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.10648, 
    farPlane=11.1912, width=3.12168, height=2.73261, cameraPosition=(-5.79362, 
    -5.20302, 3.81527), cameraUpVector=(0.557674, 0.440354, 0.703625), 
    cameraTarget=(0.108129, 0.00774644, 0.0175657))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.1087, 
    farPlane=11.2267, width=3.12282, height=2.7336, cameraPosition=(-4.12558, 
    -7.22724, 2.47592), cameraUpVector=(0.389391, 0.446611, 0.805551), 
    cameraTarget=(0.0903525, 0.0293188, 0.0318393))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.14022, 
    farPlane=11.1972, width=3.13893, height=2.7477, cameraPosition=(-3.93925, 
    -7.16012, 2.94343), cameraUpVector=(0.377581, 0.507781, 0.774333), 
    cameraTarget=(0.0887765, 0.0287511, 0.0278852))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.21224, 
    farPlane=11.122, width=3.17574, height=2.77993, cameraPosition=(-3.88304, 
    -6.57592, 4.15017), cameraUpVector=(0.427364, 0.604977, 0.671836), 
    cameraTarget=(0.0883077, 0.0238787, 0.0178205))
