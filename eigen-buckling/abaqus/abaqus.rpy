# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Tue Sep  9 16:07:22 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=538.466003417969, 
    height=243.600006103516)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/buckling_eigen_panel.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/buckling_eigen_panel.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          115
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.21205, 
    farPlane=10.333, width=6.90443, height=3.22734, cameraPosition=(4.4022, 
    -0.564372, 7.57452), cameraUpVector=(-0.0716068, 0.958626, -0.275514), 
    cameraTarget=(0.0758187, -0.0255291, -0.00158894))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.24114, 
    farPlane=10.2675, width=6.93228, height=3.24036, cameraPosition=(0.637973, 
    -3.82828, 7.85448), cameraUpVector=(0.426744, 0.903735, 0.0339585), 
    cameraTarget=(0.0622982, -0.0372525, -0.000583379))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.93643, 
    farPlane=10.4718, width=6.64057, height=3.104, cameraPosition=(-5.23842, 
    -0.751875, 6.91788), cameraUpVector=(0.800847, 0.545092, 0.248028), 
    cameraTarget=(0.0533653, -0.032576, -0.00200712))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.36838, 
    farPlane=11.0966, width=6.09675, height=2.8498, cameraPosition=(-2.83016, 
    -7.36807, 3.74343), cameraUpVector=(0.321941, 0.62879, 0.707797), 
    cameraTarget=(0.0431577, -0.00453284, 0.011448))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.29012, 
    farPlane=11.1585, width=6.02183, height=2.81478, cameraPosition=(-4.36803, 
    -6.41646, 3.99023), cameraUpVector=(0.392427, 0.611054, 0.687469), 
    cameraTarget=(0.044659, -0.00546182, 0.0112071))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.3135, 
    farPlane=11.144, width=6.04421, height=2.82524, cameraPosition=(-3.48276, 
    -7.41336, 3.02461), cameraUpVector=(0.28199, 0.572074, 0.770203), 
    cameraTarget=(0.0429638, -0.00355289, 0.0130561))
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/buckling_eigen_panel.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/buckling_eigen_panel.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          115
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.26936, 
    farPlane=10.2766, width=6.95929, height=3.25298, cameraPosition=(4.6094, 
    -0.0442564, 7.47189), cameraUpVector=(-0.22751, 0.941331, -0.249269), 
    cameraTarget=(0.0765731, -0.0236354, -0.00196262))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.74588, 
    farPlane=10.8103, width=6.45814, height=3.01873, cameraPosition=(3.29614, 
    -4.36916, 6.87059), cameraUpVector=(0.147674, 0.981126, 0.124837), 
    cameraTarget=(0.0717888, -0.0393914, -0.00415322))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.27687, 
    farPlane=10.2248, width=6.96648, height=3.25634, cameraPosition=(-0.172507, 
    -4.16626, 7.70095), cameraUpVector=(0.369577, 0.920603, 0.126103), 
    cameraTarget=(0.057141, -0.0385346, -0.000646668))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.33285, 
    farPlane=11.1424, width=6.06272, height=2.8339, cameraPosition=(-4.09141, 
    -6.01058, 4.8532), cameraUpVector=(0.461715, 0.654575, 0.598624), 
    cameraTarget=(0.052746, -0.040603, -0.00384036))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.40453, 
    farPlane=11.1164, width=6.13134, height=2.86598, cameraPosition=(-2.63573, 
    -8.2139, 1.53492), cameraUpVector=(0.194733, 0.455776, 0.868531), 
    cameraTarget=(0.0521797, -0.0397458, -0.00254946))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.42402, 
    farPlane=11.0956, width=6.15, height=2.8747, cameraPosition=(-2.47984, 
    -8.0562, 2.39172), cameraUpVector=(0.179046, 0.550268, 0.815566), 
    cameraTarget=(0.0525257, -0.0393958, -0.000647866))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.58636, 
    farPlane=10.9179, width=6.30542, height=2.94735, cameraPosition=(-2.41656, 
    -6.55539, 5.27889), cameraUpVector=(0.324917, 0.769084, 0.550399), 
    cameraTarget=(0.0526614, -0.0361769, 0.00554457))
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Model-1" has been created.
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
del mdb.models['Model-1']
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.01235, 
    farPlane=11.3166, width=8.24922, height=3.85594, cameraPosition=(9.6441, 
    0.873819, 1.85255), cameraUpVector=(-0.42066, 0.898901, -0.122566), 
    cameraTarget=(0.0244937, -0.079049, 1.57956))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.25212, 
    farPlane=11.9704, width=7.46652, height=3.49008, cameraPosition=(7.15649, 
    0.360734, -4.93409), cameraUpVector=(-0.349032, 0.920663, 0.174804), 
    cameraTarget=(0.0260546, -0.078727, 1.58382))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.36647, 
    farPlane=11.7729, width=7.58425, height=3.54511, cameraPosition=(1.19503, 
    0.34337, -7.99208), cameraUpVector=(-0.0577929, 0.927261, 0.369928), 
    cameraTarget=(0.0628246, -0.0786199, 1.60268))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.32854, 
    farPlane=11.7866, width=7.5452, height=3.52686, cameraPosition=(-0.508705, 
    0.764528, -8.01238), cameraUpVector=(0.0216567, 0.909958, 0.414134), 
    cameraTarget=(0.0807806, -0.0830586, 1.60289))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.10868, 
    farPlane=11.9808, width=7.31884, height=3.42105, cameraPosition=(-1.48759, 
    1.74999, -7.76065), cameraUpVector=(0.0521058, 0.861355, 0.505324), 
    cameraTarget=(0.0923508, -0.0947065, 1.59991))
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.23276, 
    farPlane=11.8567, width=6.18503, height=2.89107, viewOffsetX=0.118935, 
    viewOffsetY=0.0471229)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.11533, 
    farPlane=12.0355, width=6.08461, height=2.84413, cameraPosition=(-4.09614, 
    1.78525, -6.95737), cameraUpVector=(0.223054, 0.859733, 0.459464), 
    cameraTarget=(0.107067, -0.0999701, 1.54546), viewOffsetX=0.117004, 
    viewOffsetY=0.0463578)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.06591, 
    farPlane=12.0726, width=6.04235, height=2.82438, cameraPosition=(-3.62963, 
    2.2962, -7.04118), cameraUpVector=(0.228604, 0.830428, 0.508064), 
    cameraTarget=(0.104093, -0.106051, 1.54995), viewOffsetX=0.116191, 
    viewOffsetY=0.0460358)
session.viewports['Viewport: 1'].view.setValues(width=6.10629, height=2.85427, 
    cameraPosition=(9.49666, -0.019349, 1.53699), cameraUpVector=(0, 1, 0), 
    cameraTarget=(-0.0725927, -0.019349, 1.53699), viewOffsetX=0, 
    viewOffsetY=0)
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(-0.0725927, 
    -0.019349, 11.1062))
session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.11862, 
    farPlane=10.8593, width=8.83857, height=4.13141, cameraPosition=(4.90529, 
    4.94061, 6.41992), cameraTarget=(0.00418174, 0.0395012, 1.51882))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.95162, 
    farPlane=11.037, cameraPosition=(4.53529, 3.86617, 7.59238), 
    cameraUpVector=(-0.47291, 0.69007, -0.547869), cameraTarget=(0.00418176, 
    0.0395012, 1.51882))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.89132, 
    farPlane=11.0963, cameraPosition=(6.40999, 3.23349, 6.08285), 
    cameraUpVector=(-0.505879, 0.746395, -0.432412), cameraTarget=(0.00536224, 
    0.0391028, 1.51787))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.92975, 
    farPlane=11.0579, width=7.80976, height=3.65052, cameraPosition=(6.43036, 
    3.20916, 6.07129), cameraTarget=(0.0257327, 0.0147749, 1.50631))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.97337, 
    farPlane=11.0116, cameraPosition=(6.27345, 3.37079, 6.17167), 
    cameraUpVector=(-0.512908, 0.732955, -0.446881), cameraTarget=(0.0256431, 
    0.0148673, 1.50637))
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 805, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 811, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
cliCommand("""web_locations""")
#: array([-0.9, -0.3, 0.3, 0.9], 'd')
cliCommand("""for web_location in web_locations:
    face = p.faces.findAt(((web_location, panel.h_longitudinal_web/2.0, panel.length/2.0),))
    web_faces.extend(face)
p.Set(faces=web_faces, name='WebFaces')""")
#*     p.Set(faces=web_faces, name='WebFaces')
#*     ^
#* SyntaxError: invalid syntax
cliCommand("""p = model.parts['plate]""")
#*     p = model.parts['plate]
#*                           ^
#* SyntaxError: EOL while scanning string literal
cliCommand("""p = model.parts['plate']""")
cliCommand("""for web_location in web_locations:
    face = p.faces.findAt(((web_location, panel.h_longitudinal_web/2.0, panel.length/2.0),))
    web_faces.extend(face)
p.Set(faces=web_faces, name='WebFaces')""")
#*     p.Set(faces=web_faces, name='WebFaces')
#*     ^
#* SyntaxError: invalid syntax
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 803, in <module>
#*     p.Set(edges=plate_edges, name='PlateFace')
cliCommand("""plate_edges = []""")
cliCommand("""plate_edges += p.edges.findAt(((-half_width, 0.0, panel.length/2.0),))""")
cliCommand("""plate_edges""")
#: [mdb.models['Parametric-Panel'].parts['plate'].edges.findAt((-1.5, 0.0, 0.75),)]
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* TypeError: keyword error on xMax
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 847, in <module>
#*     zMin=-capture_offset, zMax=panel.length + capture_offset
cliCommand("""plate_faces = p.faces.getBoundingBox(
    xMin=-half_width - capture_offset, xMax=half_width + capture_offset,
    yMin=-capture_offset, yMax=capture_offset,
    zMin=-capture_offset, zMax=panel.length + capture_offset
)""")
#* TypeError: keyword error on xMax
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 864, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.23567, 
    farPlane=10.7423, width=2.25544, height=2.83764, cameraPosition=(4.70903, 
    4.75389, 6.80289), cameraTarget=(-0.192078, -0.147212, 1.90179))
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 864, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 864, in <module>
#*     p.Set(faces=(web_faces,), name='WebFaces')
cliCommand("""web_faces = []""")
cliCommand("""for web_location in web_locations:
    face = p.faces.findAt(((web_location, panel.h_longitudinal_web/2.0, panel.length/2.0),))
    web_faces.extend(face)
p.Set(faces=(web_faces,), name='WebFaces')""")
#*     p.Set(faces=(web_faces,), name='WebFaces')
#*     ^
#* SyntaxError: invalid syntax
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
p1 = mdb.models['Parametric-Panel'].parts['plate']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 865, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
cliCommand("""web_faces""")
#: [mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.9, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.9, 0.083333, 2.0), (1.0, 0.0, 0.0))]
execfile(
    'C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py', 
    __main__.__dict__)
#: The model "Parametric-Panel" has been created.
#* Feature creation failed.
#* 
#* File 
#* "C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/eigen-buckling/models/temp.py", 
#* line 869, in <module>
#*     p.Set(faces=web_faces, name='WebFaces')
cliCommand("""web_faces""")
#: [mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.9, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.9, 0.083333, 2.0), (1.0, 0.0, 0.0))]
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
cliCommand("""web_faces""")
#: (mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.9, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((-0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.3, 0.083333, 2.0), (1.0, 0.0, 0.0)), mdb.models['Parametric-Panel'].parts['plate'].faces.findAt((0.9, 0.083333, 2.0), (1.0, 0.0, 0.0)))
