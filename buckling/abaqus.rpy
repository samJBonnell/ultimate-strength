# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Mon Sep 15 11:46:38 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=274.246856689453, 
    height=228.433334350586)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
o1 = session.openOdb(
    name='C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
#: Model: C:/Users/sbonnell/Desktop/lase/projects/ultimate-strength/buckling/riks.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       6
#: Number of Node Sets:          22
#: Number of Steps:              1
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.74821, 
    farPlane=10.598, width=3.89201, height=3.01991, cameraPosition=(4.97869, 
    -2.58583, 6.73123), cameraUpVector=(-0.0348641, 0.999208, -0.0192019), 
    cameraTarget=(0.079126, 0.0295446, -0.0187932))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.58022, 
    farPlane=10.7252, width=3.79512, height=2.94473, cameraPosition=(0.586861, 
    -8.02882, 3.25762), cameraUpVector=(0.317762, 0.653695, 0.686811), 
    cameraTarget=(0.113616, 0.0722902, 0.00848606))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.03508, 
    farPlane=11.2118, width=3.48071, height=2.70077, cameraPosition=(-4.19572, 
    -7.32256, 1.70878), cameraUpVector=(0.415363, 0.341599, 0.84308), 
    cameraTarget=(0.162534, 0.0650663, 0.0243282))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.25059, 
    farPlane=11.0392, width=3.605, height=2.79721, cameraPosition=(-3.09107, 
    -7.50715, -2.83661), cameraUpVector=(0.0230264, -0.0109451, 0.999675), 
    cameraTarget=(0.147448, 0.0675873, 0.0864057))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.12793, 
    farPlane=11.1358, width=3.53426, height=2.74232, cameraPosition=(-3.49939, 
    -7.6971, 1.69509), cameraUpVector=(0.237985, 0.447428, 0.862074), 
    cameraTarget=(0.151997, 0.0697035, 0.0359193))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.19222, 
    farPlane=11.0744, width=3.57134, height=2.77109, cameraPosition=(-3.11204, 
    -7.75657, 2.15713), cameraUpVector=(0.286614, 0.478482, 0.830005), 
    cameraTarget=(0.14709, 0.0704569, 0.0300665))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.1156, 
    farPlane=11.1179, width=3.52715, height=2.7368, cameraPosition=(-4.77218, 
    -6.33944, 3.35352), cameraUpVector=(0.424903, 0.515926, 0.743826), 
    cameraTarget=(0.167838, 0.0527462, 0.0151146))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.07887, 
    farPlane=11.1569, width=3.50597, height=2.72036, cameraPosition=(-4.87093, 
    -6.52741, 2.79286), cameraUpVector=(0.392009, 0.474167, 0.78835), 
    cameraTarget=(0.169264, 0.0554613, 0.0232128))
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.07621, 
    farPlane=11.1174, width=3.50443, height=2.71918, cameraPosition=(-7.65683, 
    -3.77016, -0.347016), cameraUpVector=(0.227334, 0.191687, 0.954764), 
    cameraTarget=(0.212171, 0.0129959, 0.0715712))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.42783, 
    farPlane=10.7027, width=3.70723, height=2.87654, cameraPosition=(-7.95107, 
    -1.0836, 2.94607), cameraUpVector=(0.61012, 0.180866, 0.771389), 
    cameraTarget=(0.217113, -0.0321296, 0.016258))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.51624, 
    farPlane=10.5926, width=3.75822, height=2.9161, cameraPosition=(-7.79404, 
    0.357389, 3.48457), cameraUpVector=(0.663636, -0.216507, 0.716039), 
    cameraTarget=(0.213887, -0.0617286, 0.00519683))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.39954, 
    farPlane=10.7232, width=3.69091, height=2.86387, cameraPosition=(-8.43983, 
    0.562364, 0.998815), cameraUpVector=(0.412036, -0.25907, 0.873561), 
    cameraTarget=(0.227987, -0.0662041, 0.0594718))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.10317, 
    farPlane=10.9697, width=3.51998, height=2.73124, cameraPosition=(-7.13494, 
    3.29976, 3.30921), cameraUpVector=(0.542675, -0.399919, 0.738626), 
    cameraTarget=(0.200579, -0.123701, 0.0109433))
session.viewports['Viewport: 1'].view.setValues(nearPlane=5.98042, 
    farPlane=11.0913, width=3.44918, height=2.67631, cameraPosition=(-7.26588, 
    4.1399, 1.5573), cameraUpVector=(0.422501, -0.249458, 0.871357), 
    cameraTarget=(0.20372, -0.143853, 0.0529656))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.05775, 
    farPlane=11.0199, width=3.49378, height=2.71092, cameraPosition=(-7.48138, 
    3.52376, 2.02297), cameraUpVector=(0.472843, -0.258991, 0.842225), 
    cameraTarget=(0.208904, -0.129031, 0.0417637))
