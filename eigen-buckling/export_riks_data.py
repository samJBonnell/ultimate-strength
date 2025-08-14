from odbAccess import openOdb
import json

odb_path = "buckling_riks_panel.odb"
trial_label_value = 0.002  # numeric; could also be string like "case_A"

odb = openOdb(odb_path, readOnly=True)
load_point_node = odb.rootAssembly.nodeSets['LOAD_SET']

fd_curve = []

for step_name, step in odb.steps.items():
    for frame in step.frames:
        disp = frame.fieldOutputs['U'].getSubset(region=load_point_node).values[0].data[0]
        force = frame.fieldOutputs['RF'].getSubset(region=load_point_node).values[0].data[0]
        fd_curve.append((trial_label_value, disp, force))

odb.close()

with open("force_displacement.jsonl", "a") as fout:
    for trial_label, displacement, force in fd_curve:
        record = {
            "trial_label": trial_label,  # matches reader
            "displacement": float(displacement),
            "force": float(force)
        }
        fout.write(json.dumps(record) + "\n")