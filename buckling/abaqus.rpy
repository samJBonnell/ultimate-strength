# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Tue Sep 16 08:53:57 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=466.173034667969, 
    height=228.433334350586)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.91107, 
    farPlane=10.5338, width=0.579645, height=0.231882, viewOffsetX=-0.685964, 
    viewOffsetY=-0.105272)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/eigen_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/eigen_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.56943, 
    farPlane=10.9092, width=5.07249, height=2.0292, viewOffsetX=-0.835945, 
    viewOffsetY=0.156653)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.64508, 
    farPlane=10.3477, width=5.13089, height=2.05257, cameraPosition=(6.69459, 
    0.664608, 5.313), cameraUpVector=(0.0196242, 0.6341, -0.773002), 
    cameraTarget=(-0.276041, -0.726638, 0.225403), viewOffsetX=-0.845571, 
    viewOffsetY=0.158457)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.3801, 
    farPlane=10.268, width=6.47056, height=2.58849, cameraPosition=(-0.696721, 
    1.44048, 9.29306), cameraUpVector=(0.747137, 0.512627, -0.423084), 
    cameraTarget=(-0.0293778, -0.799823, 0.870193), viewOffsetX=-1.06635, 
    viewOffsetY=0.19983)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.03844, 
    farPlane=11.1953, width=6.20675, height=2.48296, cameraPosition=(-5.53153, 
    -1.8726, 7.74557), cameraUpVector=(0.695303, 0.69331, 0.189408), 
    cameraTarget=(-0.318643, -0.995913, 0.783806), viewOffsetX=-1.02287, 
    viewOffsetY=0.191683)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.27232, 
    farPlane=11.8675, width=5.6152, height=2.24632, cameraPosition=(-3.61138, 
    -7.92642, 4.12371), cameraUpVector=(0.343963, 0.654689, 0.673106), 
    cameraTarget=(0.22048, -1.24291, -0.0062438), viewOffsetX=-0.925383, 
    viewOffsetY=0.173414)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.23631, 
    farPlane=12.0222, width=5.58739, height=2.2352, cameraPosition=(-3.57888, 
    -8.7942, 1.89229), cameraUpVector=(0.213045, 0.507691, 0.834782), 
    cameraTarget=(0.275601, -1.23274, -0.199655), viewOffsetX=-0.9208, 
    viewOffsetY=0.172555)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.27141, 
    farPlane=12.2975, width=5.61449, height=2.24604, cameraPosition=(-6.42993, 
    -7.33058, 1.29594), cameraUpVector=(0.359192, 0.347797, 0.866036), 
    cameraTarget=(-0.225095, -1.38428, -0.30109), viewOffsetX=-0.925267, 
    viewOffsetY=0.173392)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.28045, 
    farPlane=12.1886, width=5.62151, height=2.24883, cameraPosition=(-6.11848, 
    -7.23313, 2.49453), cameraUpVector=(0.468893, 0.393196, 0.790908), 
    cameraTarget=(-0.219955, -1.3839, -0.226295), viewOffsetX=-0.926417, 
    viewOffsetY=0.173608)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.65952, 
    farPlane=11.8095, width=0.49775, height=0.19912, viewOffsetX=-1.13315, 
    viewOffsetY=-0.00662656)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.65999, 
    farPlane=11.7896, width=0.49778, height=0.199132, cameraPosition=(-6.05771, 
    -7.23113, 2.61094), cameraUpVector=(0.475523, 0.401167, 0.782907), 
    cameraTarget=(-0.210764, -1.38068, -0.216494), viewOffsetX=-1.13322, 
    viewOffsetY=-0.00662696)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.44175, 
    farPlane=12.0078, width=3.63581, height=1.45447, viewOffsetX=-1.01677, 
    viewOffsetY=0.187896)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.4532, 
    farPlane=11.709, width=3.6414, height=1.45671, cameraPosition=(-1.81849, 
    -9.42369, -0.776053), cameraUpVector=(0.110215, 0.288897, 0.950995), 
    cameraTarget=(0.595097, -1.02551, -0.543114), viewOffsetX=-1.01833, 
    viewOffsetY=0.188185)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.36176, 
    farPlane=11.7934, width=3.59672, height=1.43884, cameraPosition=(-2.83527, 
    -9.14151, 1.07394), cameraUpVector=(0.194575, 0.442632, 0.875338), 
    cameraTarget=(0.44963, -1.16876, -0.359117), viewOffsetX=-1.00584, 
    viewOffsetY=0.185876)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.40847, 
    farPlane=11.8481, width=3.61954, height=1.44797, cameraPosition=(-5.11569, 
    -7.39688, 3.61231), cameraUpVector=(0.418965, 0.559613, 0.715053), 
    cameraTarget=(0.0342483, -1.36027, -0.0546415), viewOffsetX=-1.01222, 
    viewOffsetY=0.187055)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=34 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.33933, 
    farPlane=11.9172, width=4.26438, height=1.70592, viewOffsetX=-0.865822, 
    viewOffsetY=0.475051)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=36 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.37164, 
    farPlane=11.893, width=4.37149, height=1.74877, viewOffsetX=-0.521767, 
    viewOffsetY=0.486848)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/models/riks.py', 
    __main__.__dict__)
#: The model "riks" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1691' from location (-0.899999976158142, 0.0138888889923692, 0.0) to '(-0.899999976158142, 0.0121904900297523, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2389' from location (0.300000011920929, 0.0138888889923692, 0.0) to '(0.300000011920929, 0.0121904900297523, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '3087' from location (0.899999976158142, 0.0138888889923692, 0.0) to '(0.899999976158142, 0.0121904900297523, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '3785' from location (-0.300000011920929, 0.0138888889923692, 0.0) to '(-0.300000011920929, 0.0121904900297523, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1690' from location (-0.899999976158142, 0.0138888889923692, 3.0) to '(-0.899999976158142, 0.0121904900297523, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2388' from location (0.300000011920929, 0.0138888889923692, 3.0) to '(0.300000011920929, 0.0121904900297523, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '3086' from location (0.899999976158142, 0.0138888889923692, 3.0) to '(0.899999976158142, 0.0121904900297523, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '3784' from location (-0.300000011920929, 0.0138888889923692, 3.0) to '(-0.300000011920929, 0.0121904900297523, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
#: [equation_sets] Linked 'Free-End-Follower' with Free-End-Main.
#: [equation_sets] Linked 'Fixed-End-Follower' with Fixed-End-Main.
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['riks'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
#: 
#: Point 1: 1.5, -300.E-03, 12.19E-03  Point 2: 1.5, -300.E-03, 0.
#:    Distance: 12.19E-03  Components: 0., 0., -12.19E-03
#: 
#: Point 1: 1.433921, -300.E-03, 41.667E-03  Point 2: 1.433921, -300.E-03, 27.778E-03
#:    Distance: 13.889E-03  Components: 0., 0., -13.889E-03
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.9337, 
    farPlane=10.5483, width=0.494165, height=0.197686, viewOffsetX=0.853097, 
    viewOffsetY=-0.676733)
openMdb(
    pathName='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/abaqus/SP1-T2.cae')
#: The model database "C:\Users\sbonnell\Desktop\lase\projects\ultimate-strength\buckling\abaqus\SP1-T2.cae" has been opened.
a = mdb.models['eigen-complete'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['eigen-complete'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
#: 
#: Point 1: 1.42, -300.E-03, 41.667E-03  Point 2: 1.42, -300.E-03, 20.833E-03
#:    Distance: 20.833E-03  Components: 0., 0., -20.833E-03
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.92701, 
    farPlane=10.555, width=0.578589, height=0.231459, viewOffsetX=0.822415, 
    viewOffsetY=-0.678601)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb'])
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/eigen_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/eigen_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
#: 
#: Node: PANEL.1225
#:                                         1             2             3        Magnitude
#: Base coordinates:                  1.50000e+00, -9.00000e-01,  1.20000e-02,      -      
#: No deformed coordinates for current plot.
#: 
#: Node: PANEL.8
#:                                         1             2             3        Magnitude
#: Base coordinates:                  1.50000e+00, -9.00000e-01,  6.66134e-16,      -      
#: No deformed coordinates for current plot.
#: 
#: Nodes for distance: PANEL.1225, PANEL.8
#:                                        1             2             3        Magnitude
#: Base distance:                     0.00000e+00,  0.00000e+00, -1.20000e-02,  1.20000e-02
#: No deformed coordinates for current plot.
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.50161, 
    farPlane=10.9804, width=5.96644, height=2.38682, viewOffsetX=0.685883, 
    viewOffsetY=0.0237277)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=2 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=3 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=4 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=5 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.42388, 
    farPlane=11.8716, width=4.40246, height=1.76116, cameraPosition=(-3.71662, 
    -8.27128, 3.51387), cameraUpVector=(0.34255, 0.601875, 0.721391), 
    cameraTarget=(0.239556, -1.36049, -0.0914937), viewOffsetX=-0.525464, 
    viewOffsetY=0.490297)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.33219, 
    farPlane=11.9681, width=4.34809, height=1.73941, cameraPosition=(-3.32071, 
    -9.09618, 0.655224), cameraUpVector=(0.242973, 0.38217, 0.891577), 
    cameraTarget=(0.351302, -1.24027, -0.444789), viewOffsetX=-0.518974, 
    viewOffsetY=0.484241)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.61677, 
    farPlane=11.6835, width=0.55104, height=0.220438, viewOffsetX=-1.58175, 
    viewOffsetY=0.273694)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.76746, 
    farPlane=12.0399, width=0.561941, height=0.224799, cameraPosition=(
    -5.44328, -8.29957, 0.882836), cameraUpVector=(0.363427, 0.318842, 
    0.875363), cameraTarget=(-0.099571, -1.51702, -0.477585), 
    viewOffsetX=-1.61305, viewOffsetY=0.279109)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.70075, 
    farPlane=12.1066, width=1.53652, height=0.61467, viewOffsetX=-1.58423, 
    viewOffsetY=0.298881)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.01321, 
    farPlane=12.3204, width=1.59886, height=0.63961, cameraPosition=(-7.4062, 
    -6.68988, 2.27434), cameraUpVector=(0.484061, 0.32963, 0.810573), 
    cameraTarget=(-0.681018, -1.677, -0.185703), viewOffsetX=-1.64851, 
    viewOffsetY=0.311008)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.6892, 
    farPlane=12.3251, width=1.73374, height=0.693567, cameraPosition=(-9.20061, 
    -3.58081, 3.86214), cameraUpVector=(0.679987, 0.189093, 0.708422), 
    cameraTarget=(-1.58046, -1.51629, 0.109813), viewOffsetX=-1.78758, 
    viewOffsetY=0.337244)
session.viewports['Viewport: 1'].view.setValues(width=1.44235, height=0.576999, 
    cameraPosition=(10.0649, 1.03366, -0.289656), cameraUpVector=(0, 1, 0), 
    cameraTarget=(-0.442286, 1.03366, -0.289656), viewOffsetX=0, viewOffsetY=0)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.46869, 
    farPlane=11.6272, width=0.970906, height=0.438061, viewOffsetX=-0.0436629, 
    viewOffsetY=0.0123818)
session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.49046, 
    farPlane=11.6054, width=0.695896, height=0.31398, cameraPosition=(10.0649, 
    0.982791, -0.140294), cameraTarget=(0.016964, 0.982791, -0.140294))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.48636, 
    farPlane=11.6935, cameraPosition=(10.0476, 1.30439, -0.634351), 
    cameraUpVector=(-0.0320838, 0.999485, -0.000786096), cameraTarget=(
    0.016964, 0.982791, -0.140295))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.46755, 
    farPlane=11.7123, width=1.00874, height=0.455129, cameraPosition=(10.0463, 
    1.31506, -0.65297), cameraTarget=(0.015705, 0.993456, -0.158914))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.46724, 
    farPlane=11.6337, cameraPosition=(10.0636, 1.03053, -0.153493), 
    cameraUpVector=(-0.00380655, 0.999988, -0.00312312), cameraTarget=(
    0.015777, 0.992272, -0.156835))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.42901, 
    farPlane=11.6077, cameraPosition=(9.8701, 1.16, 1.79901), cameraUpVector=(
    -0.0096045, 0.999258, -0.0372936), cameraTarget=(0.0157286, 0.992304, 
    -0.156346))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    0.992304, -0.156346), cameraUpVector=(0, 1, 0))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.43171, 
    farPlane=11.6616, width=1.71677, height=0.774585, cameraPosition=(10.0636, 
    1.05045, -0.0563012), cameraTarget=(0.0157286, 1.05045, -0.0563012))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    0.683538, -0.0758901), cameraTarget=(0.0157286, 0.683538, -0.0758901))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    0.361317, -0.0484656), cameraTarget=(0.0157286, 0.361317, -0.0484656))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    0.268022, -0.054734), cameraTarget=(0.0157286, 0.268022, -0.054734))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    -0.0447912, -0.0657038), cameraTarget=(0.0157286, -0.0447912, -0.0657038))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    -0.297237, -0.0931283), cameraTarget=(0.0157286, -0.297237, -0.0931283))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    -0.512051, -0.103315), cameraTarget=(0.0157286, -0.512051, -0.103315))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    -0.758225, -0.111151), cameraTarget=(0.0157286, -0.758225, -0.111151))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    -0.9064, -0.103315), cameraTarget=(0.0157286, -0.9064, -0.103315))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(10.0636, 
    -0.932272, -0.111934), cameraTarget=(0.0157286, -0.932272, -0.111934))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.48344, 
    farPlane=11.6098, width=0.768023, height=0.346522, cameraPosition=(10.0636, 
    -0.918585, -0.00947173), cameraTarget=(0.0157286, -0.918585, -0.00947173))
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.43466, 
    farPlane=12.0801, cameraPosition=(8.99011, -3.55757, 3.6584), 
    cameraUpVector=(0.194859, 0.957587, 0.212265), cameraTarget=(0.0158603, 
    -0.918261, -0.0099217))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.95311, 
    farPlane=13.5616, width=21.954, height=9.90535, cameraPosition=(9.60529, 
    -3.5052, 2.1911), cameraTarget=(0.631035, -0.865894, -1.47722))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.76377, 
    farPlane=13.7428, cameraPosition=(5.37709, -8.26444, 3.47837), 
    cameraUpVector=(0.471605, 0.686108, 0.553936), cameraTarget=(0.544677, 
    -0.963098, -1.45093))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.88795, 
    farPlane=12.974, cameraPosition=(1.77331, -8.74999, 4.80324), 
    cameraUpVector=(0.546813, 0.577153, 0.606539), cameraTarget=(0.472481, 
    -0.972825, -1.42439))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.37169, 
    farPlane=11.926, cameraPosition=(-1.71961, -5.94834, 7.00802), 
    cameraUpVector=(0.582384, 0.625468, 0.519248), cameraTarget=(0.513598, 
    -1.0058, -1.45034))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.13495, 
    farPlane=13.2238, cameraPosition=(-4.53657, -7.99796, 3.46265), 
    cameraUpVector=(0.389259, 0.270472, 0.880524), cameraTarget=(0.790412, 
    -0.80439, -1.10195))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.05915, 
    farPlane=13.2698, cameraPosition=(-5.18435, -7.76757, 2.99068), 
    cameraUpVector=(0.345442, 0.234406, 0.908693), cameraTarget=(0.815073, 
    -0.813161, -1.08398))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.67028, 
    farPlane=11.6587, width=0.326744, height=0.147423, cameraPosition=(
    -6.33402, -6.58046, 3.32403), cameraTarget=(-0.334599, 0.37395, -0.750629))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.72626, 
    farPlane=11.6832, cameraPosition=(-7.06238, -5.74985, 3.50919), 
    cameraUpVector=(0.385584, 0.20286, 0.900096), cameraTarget=(-0.305704, 
    0.340998, -0.757975))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.73436, 
    farPlane=11.6751, width=0.205065, height=0.0925229, cameraPosition=(
    -6.97168, -5.86905, 3.48267), cameraTarget=(-0.215002, 0.221802, 
    -0.784493))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
a = mdb.models['eigen-complete'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
openMdb(
    pathName='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/abaqus/SP1-T2.cae')
#: The model database "C:\Users\sbonnell\Desktop\lase\projects\ultimate-strength\buckling\abaqus\SP1-T2.cae" has been opened.
a = mdb.models['eigen-complete'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/models/riks.py', 
    __main__.__dict__)
#: The model "riks" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1226' from location (-0.899999976158142, 0.020833333954215, 0.0) to '(-0.899999976158142, 0.0120000001043081, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1687' from location (0.300000011920929, 0.020833333954215, 0.0) to '(0.300000011920929, 0.0120000001043081, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2148' from location (0.899999976158142, 0.020833333954215, 0.0) to '(0.899999976158142, 0.0120000001043081, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2609' from location (-0.300000011920929, 0.020833333954215, 0.0) to '(-0.300000011920929, 0.0120000001043081, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.020833333954215, 3.0) to '(-0.899999976158142, 0.0120000001043081, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1686' from location (0.300000011920929, 0.020833333954215, 3.0) to '(0.300000011920929, 0.0120000001043081, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2147' from location (0.899999976158142, 0.020833333954215, 3.0) to '(0.899999976158142, 0.0120000001043081, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2608' from location (-0.300000011920929, 0.020833333954215, 3.0) to '(-0.300000011920929, 0.0120000001043081, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
#: [equation_sets] Linked 'Free-End-Follower' with Free-End-Main.
#: [equation_sets] Linked 'Fixed-End-Follower' with Fixed-End-Main.
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.62923, 
    farPlane=10.3107, cameraPosition=(5.91185, 0.535239, 6.12786), 
    cameraUpVector=(-0.273449, 0.920249, -0.279943), cameraTarget=(-0.00166478, 
    0.00719109, 0.0603199))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.8024, 
    farPlane=10.1351, cameraPosition=(1.98604, -3.38903, 7.57725), 
    cameraUpVector=(-0.00169446, 0.997132, 0.0756589), cameraTarget=(
    0.00716921, 0.0160216, 0.0570586))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.40756, 
    farPlane=10.5206, cameraPosition=(0.454555, -7.28924, 4.35003), 
    cameraUpVector=(0.14792, 0.763858, 0.628204), cameraTarget=(0.0108361, 
    0.0253599, 0.0647857))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.15226, 
    farPlane=10.7724, cameraPosition=(-1.08035, -8.19579, 1.86981), 
    cameraUpVector=(0.121978, 0.513503, 0.849374), cameraTarget=(0.01536, 
    0.0280318, 0.0720957))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.09168, 
    farPlane=10.8359, cameraPosition=(-1.84334, -7.72503, 2.98626), 
    cameraUpVector=(0.219579, 0.601453, 0.76814), cameraTarget=(0.0177672, 
    0.0265466, 0.0685734))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.06306, 
    farPlane=10.8654, cameraPosition=(-2.19206, -7.52643, 3.25166), 
    cameraUpVector=(0.241681, 0.618968, 0.747308), cameraTarget=(0.0188069, 
    0.0259545, 0.0677821))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.64405, 
    farPlane=10.2844, width=0.825802, height=0.372592, cameraPosition=(
    -3.26922, -7.40397, 2.79417), cameraTarget=(-1.05835, 0.148413, -0.389711))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
a = mdb.models['eigen-complete'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.51663, 
    farPlane=10.2484, width=0.254862, height=0.114991, cameraPosition=(4.29412, 
    4.78916, 5.68238), cameraTarget=(-0.607019, -0.111979, 0.781237))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.49942, 
    farPlane=10.2695, width=0.473543, height=0.213657, cameraPosition=(4.31339, 
    4.79497, 5.66153), cameraTarget=(-0.587993, -0.106419, 0.76015))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.47017, 
    farPlane=10.3026, width=0.879187, height=0.396678, cameraPosition=(4.33574, 
    4.80296, 5.63119), cameraTarget=(-0.565647, -0.0984271, 0.729811))
del session.xyDataObjects['_U:U1 PI: PANEL N: 1226_1']
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
a = mdb.models['eigen-complete'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
xyp = session.xyPlots['XYPlot-1']
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.82586, 
    farPlane=10.9469, width=10.1871, height=4.59631, cameraPosition=(4.91159, 
    4.70764, 5.15065), cameraTarget=(0.0102074, -0.193741, 0.249271))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.54408, 
    farPlane=10.4424, cameraPosition=(5.39697, -0.592371, 6.76957), 
    cameraUpVector=(-0.137157, 0.958049, -0.251653), cameraTarget=(0.00424296, 
    -0.128612, 0.229377))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.50877, 
    farPlane=10.5599, cameraPosition=(3.66327, -2.66375, 7.4584), 
    cameraUpVector=(0.0741112, 0.993994, -0.0805194), cameraTarget=(0.00346854, 
    -0.129537, 0.229685))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.69772, 
    farPlane=10.4944, cameraPosition=(0.222999, -5.4596, 6.84167), 
    cameraUpVector=(0.361317, 0.890933, 0.275116), cameraTarget=(-0.0146265, 
    -0.144243, 0.226441))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.04608, 
    farPlane=11.265, cameraPosition=(-2.90079, -7.31987, 3.76656), 
    cameraUpVector=(0.420542, 0.578054, 0.699283), cameraTarget=(-0.0533689, 
    -0.167315, 0.188302))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.81095, 
    farPlane=11.579, cameraPosition=(-5.81357, -6.40344, 0.781229), 
    cameraUpVector=(0.312352, 0.262513, 0.912975), cameraTarget=(-0.109261, 
    -0.14973, 0.131017))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.02998, 
    farPlane=11.3442, cameraPosition=(-2.63752, -8.2162, 1.09496), 
    cameraUpVector=(0.0673196, 0.437727, 0.896584), cameraTarget=(-0.0341893, 
    -0.192578, 0.138433))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.00669, 
    farPlane=11.3451, cameraPosition=(-2.83921, -7.9306, 2.2259), 
    cameraUpVector=(0.170018, 0.525795, 0.833447), cameraTarget=(-0.0387778, 
    -0.18608, 0.164163))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.87396, 
    farPlane=11.479, cameraPosition=(-4.36631, -7.16104, 2.34504), 
    cameraUpVector=(0.34043, 0.452443, 0.824259), cameraTarget=(-0.0715953, 
    -0.169542, 0.166723))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.9102, 
    farPlane=11.431, cameraPosition=(-4.04592, -7.16455, 2.87848), 
    cameraUpVector=(0.382118, 0.489616, 0.783749), cameraTarget=(-0.0646888, 
    -0.169618, 0.178222))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.4751, 
    farPlane=10.8661, width=2.77803, height=1.25341, cameraPosition=(-4.64801, 
    -6.99788, 2.42252), cameraTarget=(-0.666777, -0.00294764, -0.277736))
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.51141, 
    farPlane=10.2994, width=0.730239, height=0.329475, cameraPosition=(4.31729, 
    4.8101, 5.64251), cameraTarget=(-0.584097, -0.0912896, 0.741124))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.49723, 
    farPlane=10.2606, width=0.398395, height=0.179751, cameraPosition=(4.31067, 
    4.7963, 5.66292), cameraTarget=(-0.590718, -0.105086, 0.76154))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.67371, 
    farPlane=10.3052, width=0.572466, height=0.258289, cameraPosition=(4.26522, 
    4.8272, 5.67747), cameraTarget=(-0.636165, -0.0741803, 0.776082))
odb = session.odbs['C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks_1.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'U', NODAL, ((COMPONENT, 'U1'), )), ), nodePick=(('PANEL', 1, (
    '[#0:38 #200 ]', )), ), )
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
