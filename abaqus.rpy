# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Fri Sep 12 09:17:44 2025
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
execfile('eigen-buckling/models/temp.py', __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-12 09:17:44
#: The model "eigen" has been created.
#: [assign_section_bounds] Assigned section 't-0.01' to 0 faces.
#: [assign_section_bounds] Assigned section 't-0.0078' to 0 faces.
#: [assign_section_bounds] Assigned section 't-0.004' to 0 faces.
#: [move_closest_nodes_to_axis] Moved node '1225' from location (-0.899999976158142, 0.025000000372529, 0.0) to '(-0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1684' from location (0.300000011920929, 0.025000000372529, 0.0) to '(0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2143' from location (0.899999976158142, 0.025000000372529, 0.0) to '(0.899999976158142, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '2602' from location (-0.300000011920929, 0.025000000372529, 0.0) to '(-0.300000011920929, 0.0101904906332493, 0.0)'.
#: [move_closest_nodes_to_axis] Moved node '1224' from location (-0.899999976158142, 0.025000000372529, 3.0) to '(-0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '1683' from location (0.300000011920929, 0.025000000372529, 3.0) to '(0.300000011920929, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2142' from location (0.899999976158142, 0.025000000372529, 3.0) to '(0.899999976158142, 0.0101904906332493, 3.0)'.
#: [move_closest_nodes_to_axis] Moved node '2601' from location (-0.300000011920929, 0.025000000372529, 3.0) to '(-0.300000011920929, 0.0101904906332493, 3.0)'.
#: Assigned section 'local-thickness' to 60 elements.
#: Assigned section 'local-thickness' to 60 elements.
#: [equation_sets] Linked 'Web0-Follower' with Web0-Main.
#: [equation_sets] Linked 'Web1-Follower' with Web1-Main.
#: [equation_sets] Linked 'Web2-Follower' with Web2-Main.
#: [equation_sets] Linked 'Web3-Follower' with Web3-Main.
#* ValueError: Set Load-Follower may not be used; If the first term uses a set 
#* consisting of a single point all the subsequent terms must use a single 
#* point set.
#* File "eigen-buckling/models/temp.py", line 510, in <module>
#*     equation_sets(model, 'Load', 'Load-Main', 'Load-Follower', linked_dof= 
#* [1])
#* File ".\utils\constraint_utilities.py", line 50, in equation_sets
#*     (-1.0, set_two, dof)
