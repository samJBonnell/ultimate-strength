import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class ForceDisplacementData:
    def __init__(self, filename, normalize_disp=0.600, normalize_force=0.825):
        self.filename = filename
        self.normalize_disp = normalize_disp
        self.normalize_force = normalize_force
        self.grouped = defaultdict(list)
        self._load()

    def _load(self):
        with open(self.filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                try:
                    trial_label = float(data.get('trial_label', 0.0))
                except ValueError:
                    trial_label = data.get('trial_label', "")
                displacement = abs(data['displacement']) / self.normalize_disp
                force = abs(data['force']) / self.normalize_force
                self.grouped[trial_label].append((displacement, force))

    def get_trial_labels(self):
        """Sorted list of trial labels (numeric first, then strings)."""
        numeric_labels = [l for l in self.grouped if isinstance(l, (int, float))]
        string_labels = [l for l in self.grouped if isinstance(l, str)]
        return sorted(numeric_labels) + sorted(string_labels)

    def get_curve(self, trial_label):
        """Returns (displacements, forces) as np.arrays for a given trial_label."""
        curve = self.grouped[trial_label]
        displacements, forces = zip(*curve)
        return np.array(displacements), np.array(forces)

    def __repr__(self):
        return f"<ForceDisplacementData {self.filename} | {len(self.grouped)} trial groups>"
    
    def plot_curve(self, ax, ls='--', lw=0.7, base_label=""):
        for label in self.get_trial_labels():
            x, y = self.get_curve(label)
            curve_label = f"{base_label}{label}"
            ax.plot(x, y, label=curve_label, ls=ls, lw=lw)


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
simply_supported = ForceDisplacementData("exported_data/test.jsonl", normalize_disp = 1e-3, normalize_force = 1e6)
simply_supported = ForceDisplacementData("exported_data/new.jsonl", normalize_disp = 1e-3, normalize_force = 1e6)

fig, ax = plt.subplots()

simply_supported.plot_curve(ax, lw=1.0, base_label='Centroid Offset: ')
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