# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-10.49.31 163176
# Run by sbonnell on Thu Sep 11 15:59:59 2025
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
execfile('models/temp.py', __main__.__dict__)
#: 
#: 
#: Start Time: 2025-09-11 15:59:59
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
#* ValueError: Set Load-Follower may not be used; If the first term uses a set 
#* consisting of a single point all the subsequent terms must use a single 
#* point set.
#* File "models/temp.py", line 1235, in <module>
#*     equation_sets(model, 'Load', 'Load-Main', 'Load-Follower', linked_dof= 
#* [1])
#* File "models/temp.py", line 764, in equation_sets
#*     (-1.0, set_two, dof)
