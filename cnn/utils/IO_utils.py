from __future__ import print_function, division, absolute_import
import sys
import gzip
import json

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
    def __init__(self, panel):
        self.panel = panel
    
    def unique(self):
        """Return unique thicknesses in the order they first appear."""
        seen = set()
        ordered = []
        for t in [self.panel]:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered
    
    def __repr__(self):
        return "ThicknessGroup({})".format(self.__dict__)

class ModelInput(object):
    """Input parameters for model analysis"""
    def __init__(self, model_name, job_name, width, length, t_panel, pressure, mesh_plate,
            pressure_location, pressure_patch_size, numCpus=None, numGpus=None):
       
        self.model_name = model_name
        self.job_name = job_name
       
        # Global Geometry
        self.width = width
        self.length = length

        # Thickness List
        self.t_panel = t_panel

        # Applied Pressure
        self.pressure = pressure
        self.pressure_location = pressure_location
        self.pressure_patch_size = pressure_patch_size

        # Mesh Settings
        self.mesh_plate = mesh_plate

        # Model Parameters
        self.numCpus = numCpus if numCpus is not None else 1
        self.numGpus = numGpus if numGpus is not None else 0
   
    def __repr__(self):
        return "ModelInput({})".format(self.__dict__)
    
    @staticmethod
    def from_dict(d):
        """Create ModelInput from dictionary"""
        return ModelInput(
            model_name=d["model_name"],
            job_name=d["job_name"],

            width=d["width"],
            length=d["length"],

            t_panel=d["t_panel"],

            pressure=d["pressure"],
            pressure_location=d["pressure_location"],
            pressure_patch_size=d["pressure_patch_size"],

            mesh_plate=d["mesh_plate"],

            numCpus=d["numCpus"],
            numGpus=d["numGpus"],
        )
    
    def to_dict(self):
        """Convert ModelInput to dictionary"""
        return {
            "model_name": self.model_name,
            "job_name": self.job_name,

            "width": self.width,
            "length": self.length,

            "t_panel": self.t_panel,

            "pressure": self.pressure,
            "pressure_location": self.pressure_location,
            "pressure_patch_size": self.pressure_patch_size,

            "mesh_plate": self.mesh_plate,

            "numCpus": self.numCpus,
            "numGpus": self.numGpus,
        }
    
    def copy(self, **kwargs):
        """Create a copy with optionally updated fields"""
        # Start with current values
        current_values = {
            'model_name': self.model_name,
            'job_name': self.job_name,
            'width': self.width,
            'length': self.length,
            't_panel': self.t_panel,
            'pressure': self.pressure,
            'pressure_location': self.pressure_location,
            'pressure_patch_size': self.pressure_patch_size,
            'mesh_plate': self.mesh_plate,
            'numCpus': self.numCpus,
            'numGpus': self.numGpus,
        }
        
        # Update with any provided kwargs
        current_values.update(kwargs)
        
        # Create new instance
        return ModelInput(**current_values)

# ----------------------------------------------------------------------------------------------------------------------------------------
# OUTPUT INFORMATION
# ----------------------------------------------------------------------------------------------------------------------------------------

class Element(object):
    """
    Flexible element that can hold any combination of attributes (stress, displacement, location, etc.)
    """
    def __init__(self, element_id):
        self.element_id = element_id
        self.attributes = {}
    
    def add_attribute(self, name, attribute):
        """Add an attribute to this element"""
        self.attributes[name] = attribute
    
    def get_attribute(self, name):
        """Get an attribute by name, returns None if not found"""
        return self.attributes.get(name)
    
    def has_attribute(self, name):
        """Check if element has a specific attribute"""
        return name in self.attributes
    
    def remove_attribute(self, name):
        """Remove an attribute by name"""
        if name in self.attributes:
            del self.attributes[name]
    
    def get_attribute_names(self):
        """Get list of all attribute names"""
        return list(self.attributes.keys())
    
    @staticmethod
    def from_dict(d):
        element = Element(d["element_id"])
        for attr_name, attr_data in d.get("attributes", {}).items():
            if attr_data.get("type") == "displacement":
                element.add_attribute(attr_name, Displacement.from_dict(attr_data))
            elif attr_data.get("type") == "stress":
                element.add_attribute(attr_name, Stress.from_dict(attr_data))
            elif attr_data.get("type") == "location":
                element.add_attribute(attr_name, Location.from_dict(attr_data))
            elif attr_data.get("type") == "strain":
                element.add_attribute(attr_name, Strain.from_dict(attr_data))
            else:
                element.add_attribute(attr_name, attr_data)
        return element
    
    def to_dict(self):
        result = {"element_id": self.element_id, "attributes": {}}
        for name, attr in self.attributes.items():
            if hasattr(attr, 'to_dict'):
                result["attributes"][name] = attr.to_dict()
            else:
                result["attributes"][name] = attr
        return result


# ----------------------------------------------------------------------------------------------------------------------------------------
# ATTRIBUTE CLASSES
# ----------------------------------------------------------------------------------------------------------------------------------------

# Displacement Data
class Displacement(object):
    """3D displacement vector"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.type = "displacement"

    @staticmethod
    def from_dict(d):
        return Displacement(d["x"], d["y"], d["z"])
    
    def to_dict(self):
        return {
            "type": "displacement",
            "x": self.x,
            "y": self.y,
            "z": self.z
        }
    
    def magnitude(self):
        """Calculate the magnitude of displacement"""
        return (self.x**2 + self.y**2 + self.z**2)**0.5

# Stress Data
class Stress(object):
    """Stress data (von Mises by default, can be extended)"""
    def __init__(self, vm_stress):
        self.vm = vm_stress
        self.type = "stress"

    @staticmethod
    def from_dict(d):
        return Stress(d["vm"])
    
    def to_dict(self):
        return {
            "type": "stress",
            "vm": self.vm
        }

# Location Data
class Location(object):
    """3D spatial coordinates"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.type = "location"
    
    @staticmethod
    def from_dict(d):
        return Location(d["x"], d["y"], d["z"])
    
    def to_dict(self):
        return {
            "type": "location",
            "x": self.x,
            "y": self.y,
            "z": self.z
        }
    
    def distance_to(self, other_location):
        """Calculate distance to another location"""
        dx = self.x - other_location.x
        dy = self.y - other_location.y
        dz = self.z - other_location.z
        return (dx**2 + dy**2 + dz**2)**0.5

# Strain Data
class Strain(object):
    """6-component strain tensor"""
    def __init__(self, xx, yy, zz, xy=0.0, yz=0.0, xz=0.0):
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.xy = xy
        self.yz = yz
        self.xz = xz
        self.type = "strain"
    
    @staticmethod
    def from_dict(d):
        return Strain(
            d["xx"], d["yy"], d["zz"],
            d.get("xy", 0.0), d.get("yz", 0.0), d.get("xz", 0.0)
        )
    
    def to_dict(self):
        return {
            "type": "strain",
            "xx": self.xx,
            "yy": self.yy,
            "zz": self.zz,
            "xy": self.xy,
            "yz": self.yz,
            "xz": self.xz
        }


# ----------------------------------------------------------------------------------------------------------------------------------------
# MODEL OUTPUT CLASS
# ----------------------------------------------------------------------------------------------------------------------------------------

class ModelOutput(object):
    """
    Main output class for storing element data from multiple analysis steps
    """
    def __init__(self, job_name, steps, element_counts=None, elements_by_step=None, 
                 max_stress=None, assembly_mass=None):
        self.job_name = job_name
        self.steps = steps if isinstance(steps, list) else [steps]
        self.element_counts = element_counts or {}
        self.elements_by_step = elements_by_step or {}
        self.max_stress = max_stress
        self.assembly_mass = assembly_mass
    
    def add_step(self, step_name, elements):
        """Add elements for a specific step"""
        if step_name not in self.steps:
            self.steps.append(step_name)
        self.elements_by_step[step_name] = elements
    
    def get_elements_with_attribute(self, step, attribute_name):
        """Get all elements in a step that have a specific attribute"""
        if step not in self.elements_by_step:
            return []
        return [elem for elem in self.elements_by_step[step] if elem.has_attribute(attribute_name)]
    
    def get_element_by_id(self, step, element_id):
        """Get a specific element by ID in a given step"""
        if step not in self.elements_by_step:
            return None
        for elem in self.elements_by_step[step]:
            if elem.element_id == element_id:
                return elem
        return None
    
    def calculate_max_stress(self, step=None):
        """Calculate maximum stress across elements"""
        max_val = 0.0
        steps_to_check = [step] if step else self.steps
        
        for step_name in steps_to_check:
            if step_name in self.elements_by_step:
                for element in self.elements_by_step[step_name]:
                    stress = element.get_attribute("stress")
                    if stress and stress.vm > max_val:
                        max_val = stress.vm
        return max_val
    
    def get_element_count(self, step):
        """Get number of elements in a step"""
        if step in self.elements_by_step:
            return len(self.elements_by_step[step])
        return 0
    
    @staticmethod
    def from_dict(d):
        elements_by_step = {}
        for step, elements_data in d.get("elements_by_step", {}).items():
            elements_by_step[step] = [Element.from_dict(elem_data) for elem_data in elements_data]
        
        return ModelOutput(
            job_name=d["job_name"],
            steps=d["steps"],
            element_counts=d.get("element_counts"),
            elements_by_step=elements_by_step,
            max_stress=d.get("max_stress"),
            assembly_mass=d.get("assembly_mass")
        )
    
    def to_dict(self):
        result = {
            "job_name": self.job_name,
            "steps": self.steps,
            "elements_by_step": {
                step: [elem.to_dict() for elem in elements]
                for step, elements in self.elements_by_step.items()
            }
        }
        if self.element_counts:
            result["element_counts"] = self.element_counts
        if self.max_stress is not None:
            result["max_stress"] = self.max_stress
        if self.assembly_mass is not None:
            result["assembly_mass"] = self.assembly_mass
        return result
    
def write_trial_ndjson(output, path="results.jsonl"):
    """Write model output to NDJSON format (append mode)"""
    with open(path, "a") as f:
        json_line = json.dumps(clean_json(output))
        f.write(json_line + "\n")

def write_trial_ndjson_gz(output, path="results.jsonl.gz"):
    """Write model output to compressed NDJSON format"""
    with gzip.open(path, "ab") as f:
        json_line = json.dumps(output.to_dict()) + "\n"
        f.write(json_line.encode("utf-8"))

def clean_json(obj):
    """Clean object for JSON serialization"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(item) for item in obj]
    else:
        return obj