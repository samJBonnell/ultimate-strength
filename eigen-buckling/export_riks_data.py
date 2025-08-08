from odbAccess import openOdb

odb_path = "buckling_riks_uc.odb"
imperfection = '0.006'
odb = openOdb(odb_path, readOnly=True)

load_point_node = odb.rootAssembly.nodeSets['MONITORING-SET']

fd_curve = []

for step_name, step in odb.steps.items():
    for frame in step.frames:
        disp = frame.fieldOutputs['U'].getSubset(region=load_point_node).values[0].data[0]
        force = frame.fieldOutputs['CF'].getSubset(region=load_point_node).values[0].data[0]
        fd_curve.append((imperfection, disp, force))

odb.close()

import json
with open("force_displacement.jsonl", "a") as fout:
    for i, d, fz in fd_curve:
        record = {
            "imperfection": float(i),
            "displacement": float(d),
            "force": float(fz)
        }
        fout.write(json.dumps(record) + "\n")