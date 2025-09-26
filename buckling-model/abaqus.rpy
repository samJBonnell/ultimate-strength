# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Thu Sep 25 18:00:54 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.16602, 1.16667), width=171.637, 
    height=115.733)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('models/eigen.py', __main__.__dict__)
#: The model "eigen" has been created.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.01' to 5 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.0078' to 4 face(s) on part 'plate' using sets method.
#: ['Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object', 'Face object']
#: [assign_section] Assigned section 't-0.004' to 8 face(s) on part 'plate' using sets method.
#: [move_closest_nodes_to_axis] Moved node '1226' from location (-0.899999976158142, 0.0219999998807907, 0.0) to '(-0.899999976158142, 0.0130000002682209, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1687' from location (0.300000011920929, 0.0219999998807907, 0.0) to '(0.300000011920929, 0.0130000002682209, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2148' from location (0.899999976158142, 0.0219999998807907, 0.0) to '(0.899999976158142, 0.0130000002682209, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2609' from location (-0.300000011920929, 0.0219999998807907, 0.0) to '(-0.300000011920929, 0.0130000002682209, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.0219999998807907, 3.0) to '(-0.899999976158142, 0.0130000002682209, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1686' from location (0.300000011920929, 0.0219999998807907, 3.0) to '(0.300000011920929, 0.0130000002682209, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2147' from location (0.899999976158142, 0.0219999998807907, 3.0) to '(0.899999976158142, 0.0130000002682209, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2608' from location (-0.300000011920929, 0.0219999998807907, 3.0) to '(-0.300000011920929, 0.0130000002682209, 3.0)'.
#: Assigned section 'local-thickness' to 36 elements.
#: Assigned section 'local-thickness' to 36 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#: [equation_sets] Linked 'Load-Follower' with Load-Main.
#: [equation_sets] Linked 'Free-End-Follower' with Free-End-Main.
#: [equation_sets] Linked 'Fixed-End-Follower' with Fixed-End-Main.
#: Abaqus Warning: The following input options are not supported by parallel execution of element operations: buckle. Only the solver will be executed in parallel for this analysis.
print 'RT script done'
#: RT script done
