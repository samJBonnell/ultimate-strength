# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Tue Sep 23 11:54:59 2025
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
openMdb(
    pathName='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/abaqus/SP1-T2.cae')
#: The model database "C:\Users\sbonnell\Desktop\lase\projects\ultimate-strength\buckling\abaqus\SP1-T2.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['eigen-complete'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
p1 = mdb.models['eigen-complete'].parts['panel']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.20934, 
    farPlane=10.8295, width=0.528943, height=0.231889, viewOffsetX=-0.961935, 
    viewOffsetY=-0.276255)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.84207, 
    farPlane=10.5599, width=0.501997, height=0.220076, cameraPosition=(7.28209, 
    3.60588, 4.68333), cameraUpVector=(-0.607655, 0.714223, -0.347332), 
    cameraTarget=(-0.22891, -0.166203, 1.41129), viewOffsetX=-0.91293, 
    viewOffsetY=-0.262181)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.76057, 
    farPlane=10.6414, width=1.7238, height=0.755714, viewOffsetX=-0.783417, 
    viewOffsetY=-0.213112)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.6774, 
    farPlane=10.5285, width=1.70259, height=0.746417, cameraPosition=(8.08614, 
    2.05742, 3.65723), cameraUpVector=(-0.531139, 0.843103, -0.0840712), 
    cameraTarget=(-0.422278, 0.0425693, 1.44429), viewOffsetX=-0.773779, 
    viewOffsetY=-0.21049)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.68388, 
    farPlane=10.5334, width=1.70424, height=0.747142, cameraPosition=(7.98809, 
    2.34852, 3.75283), cameraUpVector=(-0.551548, 0.822279, -0.140188), 
    cameraTarget=(-0.399553, -0.0174327, 1.4291), viewOffsetX=-0.77453, 
    viewOffsetY=-0.210694)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.77573, 
    farPlane=10.4415, width=0.442866, height=0.194153, viewOffsetX=-1.05564, 
    viewOffsetY=-0.210008)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
mdb.meshEditOptions.setValues(enableUndo=True, maxUndoCacheElements=0.5)
#: Coordinates of node 25720 : 300.E-03,62.5E-03,2.96
p = mdb.models['eigen-complete'].parts['panel']
n = p.nodes
nodes = n.getSequenceFromMask(mask=('[#0:803 #800000 ]', ), )
p.editNode(nodes=nodes, coordinate1=0.300000011920929, 
    coordinate2=0.0546381212770939, coordinate3=2.96655201911926)
#: Coordinates of node 25720 : 300.E-03,54.638E-03,2.966552
p = mdb.models['eigen-complete'].parts['panel']
n = p.nodes
nodes = n.getSequenceFromMask(mask=('[#0:803 #800000 ]', ), )
p.editNode(nodes=nodes, coordinate1=0.300000131130219, 
    coordinate2=0.0637023448944092, coordinate3=2.96108388900757)
p = mdb.models['eigen-complete'].parts['panel']
n = p.nodes
nodes = n.getSequenceFromMask(mask=('[#0:803 #10800000 #2 ]', ), )
p.editNode(nodes=nodes, offset1=0.05)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.96286, 
    farPlane=10.8253, width=0.455097, height=0.19988, cameraPosition=(6.99386, 
    3.68414, 5.63239), cameraUpVector=(-0.624655, 0.719816, -0.302772), 
    cameraTarget=(-0.0743067, -0.0353269, 1.44232), viewOffsetX=-1.0848, 
    viewOffsetY=-0.215808)
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.49476, 
    farPlane=11.2934, width=7.07558, height=3.10761, viewOffsetX=-1.6239, 
    viewOffsetY=0.0233415)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.73163, 
    farPlane=12.4065, width=8.42307, height=3.69944, cameraPosition=(4.14644, 
    3.49385, 10.071), cameraUpVector=(-0.206826, 0.711446, -0.671616), 
    cameraTarget=(0.956517, -0.314296, 2.54289), viewOffsetX=-1.93316, 
    viewOffsetY=0.0277867)
session.viewports['Viewport: 1'].view.setValues(nearPlane=8.82598, 
    farPlane=13.4699, width=9.61528, height=4.22306, cameraPosition=(-1.75313, 
    4.30647, 11.7288), cameraUpVector=(-0.0392339, 0.681633, -0.730642), 
    cameraTarget=(0.4143, 0.428242, 3.87952), viewOffsetX=-2.20678, 
    viewOffsetY=0.0317197)
session.viewports['Viewport: 1'].view.setValues(nearPlane=9.25018, 
    farPlane=13.0457, width=2.58323, height=1.13456, viewOffsetX=-0.873979, 
    viewOffsetY=0.0558669)
session.viewports['Viewport: 1'].view.setValues(nearPlane=9.26221, 
    farPlane=13.0336, width=2.58659, height=1.13604, cameraPosition=(-1.75436, 
    4.32149, 11.721), cameraUpVector=(-0.0537883, 0.678159, -0.732944), 
    cameraTarget=(0.413068, 0.443259, 3.87176), viewOffsetX=-0.875116, 
    viewOffsetY=0.0559396)
session.viewports['Viewport: 1'].view.setValues(nearPlane=9.38098, 
    farPlane=12.7587, width=2.61976, height=1.15061, cameraPosition=(2.18995, 
    4.80893, 11.3303), cameraUpVector=(-0.180016, 0.656387, -0.732633), 
    cameraTarget=(1.18536, 0.401996, 3.52526), viewOffsetX=-0.886338, 
    viewOffsetY=0.0566569)
session.viewports['Viewport: 1'].view.setValues(nearPlane=9.52633, 
    farPlane=12.6134, width=0.602573, height=0.264651, viewOffsetX=-0.740081, 
    viewOffsetY=0.103953)
session.viewports['Viewport: 1'].view.setValues(nearPlane=9.49816, 
    farPlane=12.6207, width=0.600791, height=0.263869, cameraPosition=(2.46189, 
    4.95941, 11.1788), cameraUpVector=(-0.197172, 0.644681, -0.738586), 
    cameraTarget=(1.23341, 0.425808, 3.479), viewOffsetX=-0.737892, 
    viewOffsetY=0.103645)
