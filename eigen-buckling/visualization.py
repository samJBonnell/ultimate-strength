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
                displacement = abs(data['displacement']) * 1000
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

# Load the data from the plot digitizer
data = np.genfromtxt(
    "exported_data/wpd_datasets.csv",
    delimiter=",",
    skip_header=2,   # skip both label lines
    filling_values=np.nan  # fill missing entries with NaN
)

nl_esl_x, nl_esl_y = data[:, 0], data[:, 1]
lin_esl_x, lin_esl_y = data[:, 2], data[:, 3]
fem_x, fem_y       = data[:, 4], data[:, 5]

# Load data
simply_supported = ForceDisplacementData("exported_data/force_displacement.jsonl")

fig, ax = plt.subplots()

simply_supported.plot_curve(ax, label='Plastic ', lw=1.0)
ax.plot(fem_x, fem_y, marker='x', color='black', label="3D FEM", lw=0.7)

ax.set_title("Axial Force vs. End Shortening Curve - UC1", fontsize = 10)
fig.suptitle("Buckling Prediction of Stiffened Panel", fontsize = 12)
ax.set_xlabel("End Shortening [mm]", fontsize = 8)
ax.set_ylabel("Applied Force [MN]", fontsize = 8)
ax.set_xlim([0,10])

px = 4
py = 8
ax.annotate(
    "[2022]",
    xy=(px, py),           # point to label
    xytext=(px, py), # label position
)

# ax.annotate(
#     "(2022)",
#     xy=(px, py),           # point to label
#     xytext=(px+0.5, py+2), # label position
#     arrowprops=dict(arrowstyle="->", color='black')
# )

ax.grid(True, which="both", ls="-",color='0.95')
ax.legend(loc='best', prop={'size': 8})
plt.tight_layout()
plt.show()