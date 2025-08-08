import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class ForceDisplacementData:
    def __init__(self, filename, normalize_by=0.600, force_normalize=0.825):
        self.filename = filename
        self.normalize_by = normalize_by
        self.force_normalize = force_normalize
        self.grouped = defaultdict(list)
        self._load()

    def _load(self):
        with open(self.filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                imp = data.get('imperfection', 0.0)
                displacement = abs(data['displacement']) / self.normalize_by * 100  # strain (%)
                displacement = abs(data['displacement'])  # strain (%)
                force = abs(data['force']) / (1e6 / self.force_normalize)
                force = abs(data['force']) / 1e6
                self.grouped[imp].append((displacement, force))

    def get_imperfections(self):
        return sorted(self.grouped.keys())

    def get_curve(self, imperfection):
        """Returns (displacements, forces) as np.arrays for a given imperfection."""
        curve = self.grouped[imperfection]
        displacements, forces = zip(*curve)
        return np.array(displacements), np.array(forces)

    def __repr__(self):
        return f"<ForceDisplacementData {self.filename} | {len(self.grouped)} imperfection groups>"
    
    def plot_curve(self, ax, ls='--', lw=0.7, label=""):
        ini_label = label
        for imp in self.get_imperfections():
            x, y = self.get_curve(imp)
            label = ini_label + f"x={imp:.2f}"
            ax.plot(x, y, label=label, ls=ls, lw=lw)

# Load data
simply_supported = ForceDisplacementData("exported_data/unit_cell.jsonl")
simply_supported = ForceDisplacementData("exported_data/jasmin_test.jsonl")

simply_supported = ForceDisplacementData("exported_data/fd_coarse.jsonl")
# simply_supported = ForceDisplacementData("exported_data/fd_fine.jsonl")
# simply_supported = ForceDisplacementData("exported_data/fd_01.jsonl")
# simply_supported = ForceDisplacementData("exported_data/uc_00.jsonl")
# linear = ForceDisplacementData("linear.jsonl")

fig, ax = plt.subplots()

simply_supported.plot_curve(ax, label='Plastic ', lw=1.0)
# linear.plot_curve(ax, ls='-.', label='Elastic ')s

ax.set_title("Force-Displacement Curve - UC1", fontsize = 10)
fig.suptitle("Buckling Prediction of Unit Cell Stiffened Panels", fontsize = 12)
ax.set_xlabel("Load Point Strain (%)", fontsize = 8)
ax.set_ylabel("Applied Force (MN/m)", fontsize = 8)
ax.set_xlim([-0.001,0.0037])
# ax.set_ylim([0,10])

ax.grid(True, which="both", ls="-",color='0.95')
ax.legend(loc='best', prop={'size': 8})
plt.tight_layout()
plt.show()