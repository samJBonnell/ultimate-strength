import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.integrate as integrate

import json
from collections import defaultdict

displacements = []
forces = []

grouped_data = defaultdict(list)
with open('simply_supported.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        imp = data['imperfection']
        grouped_data[imp].append((abs(data['displacement']), abs(data['force'])))

fig, ax = plt.subplots()

for imp, curve in grouped_data.items():
    displacements, forces = zip(*curve)
    displacements = 100 * (np.array(displacements) / 0.600)
    forces = np.array(forces)
    ax.plot(displacements, forces, label=f"x={imp:.2f}", ls='--', linewidth=0.7)

# grouped_data = defaultdict(list)
# with open('linear.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         imp = data['imperfection']
#         grouped_data[imp].append((abs(data['displacement']), abs(data['force'])))

# for imp, curve in grouped_data.items():
#     displacements, forces = zip(*curve)
#     displacements = 100 * (np.array(displacements) / 0.600)
#     forces = np.array(forces)
#     ax.plot(displacements, forces, label=f"x={imp:.2f}", ls='-', linewidth=0.7)

ax.set_title("Force-Displacement Curve - UC1", fontsize = 10)
fig.suptitle("Buckling Prediction of Unit Cell Stiffened Panels", fontsize = 12)

ax.set_xlabel("Membrane Strain (%)", fontsize = 8)
ax.set_ylabel("Force", fontsize = 8)
ax.set_xlim([-0.01,0.5])
# ax.set_ylim([0,1e7])

plt.tight_layout()
ax.grid(True, which="both", ls="-",color='0.95')
ax.legend(loc='best')
plt.show()