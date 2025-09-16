from __future__ import print_function, division, absolute_import
import sys

# Python 2/3 compatibility
if sys.version_info[0] == 3:
    string_types = str
else:
    string_types = basestring

# ----------------------------------------------------------------------------------------------------------------------------------------
# INPUT INFORMATION
# ----------------------------------------------------------------------------------------------------------------------------------------

class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dict__)

def from_dict(d):
    if isinstance(d, dict):
        return Struct(**{k: from_dict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [from_dict(i) for i in d]
    else:
        return d

class ThicknessGroup(object):
    def __init__(self, panel, longitudinal_web, longitudinal_flange):
        self.panel = panel
        self.longitudinal_web = longitudinal_web
        self.longitudinal_flange = longitudinal_flange
    
    def unique(self):
        """Return unique thicknesses in the order they first appear."""
        seen = set()
        ordered = []
        for t in [self.panel, self.longitudinal_web, self.longitudinal_flange]:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered
    
    def __repr__(self):
        return "ThicknessGroup({})".format(self.__dict__)

class PanelInput(object):
    def __init__(self, model_name, job_name, num_longitudinal, width, length, t_panel, 
                 t_longitudinal_web, t_longitudinal_flange, h_longitudinal_web,
                 w_longitudinal_flange, axial_force, mesh_plate, 
                 mesh_longitudinal_web, mesh_longitudinal_flange, numCpus=None, numGpus=None, centroid=None):
        
        self.model_name = model_name
        self.job_name = job_name
        
        # Global Geometry
        self.num_longitudinal = num_longitudinal
        self.width = width
        self.length = length

        # Thickness List
        self.t_panel = t_panel
        self.t_longitudinal_web = t_longitudinal_web
        self.t_longitudinal_flange = t_longitudinal_flange

        # Local stiffener geometry
        self.h_longitudinal_web = h_longitudinal_web
        self.w_longitudinal_flange = w_longitudinal_flange

        # Applied Pressure
        self.axial_force = axial_force

        # Mesh Settings
        self.mesh_plate = mesh_plate
        self.mesh_longitudinal_web = mesh_longitudinal_web
        self.mesh_longitudinal_flange = mesh_longitudinal_flange

        # Model Parameters
        self.numCpus = numCpus if numCpus is not None else 1
        self.numGpus = numGpus if numGpus is not None else 0

        self.centroid = centroid if centroid is not None else -1
    
    def __repr__(self):
        return "PanelInput({})".format(self.__dict__)


# ----------------------------------------------------------------------------------------------------------------------------------------
# OUTPUT INFORMATION
# ----------------------------------------------------------------------------------------------------------------------------------------

class ElementStress(object):
    def __init__(self, element_id, stress):
        self.element_id = element_id
        self.stress = stress
    
    @staticmethod
    def from_dict(d):
        return ElementStress(d["element_id"], d["stress"])
    
    def to_dict(self):
        return {"element_id": self.element_id, "stress": self.stress}

class ElementDisplacement(object):
    def __init__(self, element_id, x, y, z):
        self.element_id = element_id
        self.x = x
        self.y = y
        self.z = z
    
    @staticmethod
    def from_dict(d):
        return ElementDisplacement(d["element_id"], d["x"], d["y"], d["z"])
    
    def to_dict(self):
        return {
            "element_id": self.element_id,
            "x": self.x,
            "y": self.y,
            "z": self.z
        }

class PanelOutput(object):
    def __init__(self, id, element_counts, stress_field, displacement_field, job_name, steps):
        self.id = id
        self.element_counts = element_counts                  # Dict[str, int]
        self.stress_field = stress_field                      # Dict[str, List[ElementStress]]
        self.displacement_field = displacement_field          # Dict[str, List[ElementDisplacement]]
        self.job_name = job_name
        self.steps = steps                                    # List of step names (str)
    
    @staticmethod
    def from_dict(d):
        stress_field = {}
        for step, stresses in d.get("stress_field", {}).items():
            stress_field[step] = [ElementStress.from_dict(s) for s in stresses]
        
        displacement_field = {}
        for step, disps in d.get("displacement_field", {}).items():
            displacement_field[step] = [ElementDisplacement.from_dict(s) for s in disps]
        
        return PanelOutput(
            id=d["id"],
            element_counts=d["element_counts"],
            stress_field=stress_field,
            displacement_field=displacement_field,
            job_name=d["job_name"],
            steps=d["steps"]
        )
    
    def to_dict(self):
        return {
            "id": self.id,
            "element_counts": self.element_counts,
            "stress_field": {
                step: [s.to_dict() for s in stresses]
                for step, stresses in self.stress_field.items()
            },
            "displacement_field": {
                step: [d.to_dict() for d in disps]
                for step, disps in self.displacement_field.items()
            },
            "job_name": self.job_name,
            "steps": self.steps
        }
    
def write_debug_file(object, file):
    f = open(file, "w")
    f.write("{}\n".format(object))
    f.close()