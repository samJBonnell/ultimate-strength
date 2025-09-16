# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Mon Sep 15 18:30:06 2025
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
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.viewports['Viewport: 1'].animationController.stop()
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=45 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=47 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=48 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=48 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=48 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=49 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=49 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=49 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=49 )
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.46992, 
    farPlane=11.0208, width=7.23768, height=2.89536, cameraPosition=(7.8133, 
    -2.0209, 3.50933), cameraUpVector=(-0.302332, 0.887007, 0.349017), 
    cameraTarget=(0.0728265, 0.00965786, -0.00788618))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.47477, 
    farPlane=11.0057, width=7.2431, height=2.89753, cameraPosition=(1.50084, 
    -8.61672, 0.0141951), cameraUpVector=(0.107005, 0.358204, 0.927491), 
    cameraTarget=(0.0698374, 0.00653464, -0.00954125))
session.viewports['Viewport: 1'].view.setValues(nearPlane=6.93465, 
    farPlane=10.5458, width=1.0068, height=0.402762, viewOffsetX=-1.23521, 
    viewOffsetY=-0.176157)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.66536, 
    farPlane=11.6212, width=1.11289, height=0.445201, cameraPosition=(-4.38201, 
    -7.99516, 3.28074), cameraUpVector=(0.661487, 0.373312, 0.650441), 
    cameraTarget=(-0.269085, -1.18107, -0.333245), viewOffsetX=-1.36536, 
    viewOffsetY=-0.194718)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.56457, 
    farPlane=11.7219, width=2.50557, height=1.00233, viewOffsetX=-1.27862, 
    viewOffsetY=-0.19865)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.47098, 
    farPlane=11.6056, width=2.47457, height=0.989929, cameraPosition=(-2.4578, 
    -9.15955, -1.19667), cameraUpVector=(0.42022, 0.170166, 0.891324), 
    cameraTarget=(0.14736, -0.827722, -0.746546), viewOffsetX=-1.2628, 
    viewOffsetY=-0.196192)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.41872, 
    farPlane=11.5734, width=2.45726, height=0.983004, cameraPosition=(-2.9511, 
    -8.78799, 2.1397), cameraUpVector=(0.330124, 0.475782, 0.81526), 
    cameraTarget=(0.105499, -0.915988, -0.118076), viewOffsetX=-1.25397, 
    viewOffsetY=-0.19482)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.42571, 
    farPlane=11.618, width=2.45957, height=0.98393, cameraPosition=(-2.7943, 
    -9.07795, 0.790157), cameraUpVector=(0.267265, 0.369757, 0.889859), 
    cameraTarget=(0.131452, -0.902045, -0.211476), viewOffsetX=-1.25515, 
    viewOffsetY=-0.195003)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.40243, 
    farPlane=11.4798, width=2.45186, height=0.980846, cameraPosition=(-2.46264, 
    -8.90908, 1.99394), cameraUpVector=(0.244084, 0.491265, 0.836111), 
    cameraTarget=(0.175478, -0.827433, -0.039783), viewOffsetX=-1.25122, 
    viewOffsetY=-0.194392)
session.viewports['Viewport: 1'].view.setValues(nearPlane=7.45093, 
    farPlane=11.4313, width=1.70255, height=0.681089, viewOffsetX=-1.28294, 
    viewOffsetY=-0.201531)
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=49 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=50 )
