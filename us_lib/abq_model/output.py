from __future__ import print_function, division, absolute_import
import sys

# Python 2/3 compatibility
if sys.version_info[0] == 3:
    string_types = str
else:
    string_types = basestring


# ----------------------------------------------------------------------------------------------------------------------------------------
# ATTRIBUTE CLASSES (Your domain objects - keep these simple)
# ----------------------------------------------------------------------------------------------------------------------------------------

class Displacement(object):
    """3D displacement vector"""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.type = "displacement"

    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    @staticmethod
    def from_dict(d):
        return Displacement(d["x"], d["y"], d["z"])
    
    def to_dict(self):
        return {"type": "displacement", "x": self.x, "y": self.y, "z": self.z}


class Stress(object):
    """Stress data"""
    
    def __init__(self, vm_stress):
        self.vm = vm_stress
        self.type = "stress"

    @staticmethod
    def from_dict(d):
        return Stress(d["vm"])
    
    def to_dict(self):
        return {"type": "stress", "vm": self.vm}


class Location(object):
    """3D spatial coordinates"""
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.type = "location"
    
    def distance_to(self, other_location):
        dx = self.x - other_location.x
        dy = self.y - other_location.y
        dz = self.z - other_location.z
        return (dx**2 + dy**2 + dz**2)**0.5
    
    @staticmethod
    def from_dict(d):
        return Location(d["x"], d["y"], d["z"])
    
    def to_dict(self):
        return {"type": "location", "x": self.x, "y": self.y, "z": self.z}


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
            "xx": self.xx, "yy": self.yy, "zz": self.zz,
            "xy": self.xy, "yz": self.yz, "xz": self.xz
        }


# Registry for deserialization
ATTRIBUTE_TYPES = {
    "displacement": Displacement,
    "stress": Stress,
    "location": Location,
    "strain": Strain
}


class Element(object):
    """Flexible element that can hold any combination of attributes"""
    
    def __init__(self, element_id):
        self.element_id = element_id
        self.attributes = {}
    
    def add_attribute(self, name, attribute):
        self.attributes[name] = attribute
        return self  # Allow chaining
    
    def get_attribute(self, name):
        return self.attributes.get(name)
    
    def has_attribute(self, name):
        return name in self.attributes
    
    def remove_attribute(self, name):
        if name in self.attributes:
            del self.attributes[name]
    
    def get_attribute_names(self):
        return list(self.attributes.keys())
    
    @staticmethod
    def from_dict(d):
        element = Element(d["element_id"])
        for attr_name, attr_data in d.get("attributes", {}).items():
            attr_type = attr_data.get("type")
            if attr_type in ATTRIBUTE_TYPES:
                element.add_attribute(attr_name, ATTRIBUTE_TYPES[attr_type].from_dict(attr_data))
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
# MODEL OUTPUT
# ----------------------------------------------------------------------------------------------------------------------------------------

class ModelOutput(object):
    """
    Container for FEA analysis results.
    Organizes elements by analysis steps and stores model-level metadata.
    """
    
    def __init__(self, job_name):
        self.job_name = job_name
        
        # Core element data
        self.steps = []
        self.elements_by_step = {}
        
        # Analysis metadata
        self.max_stress = None
        self.assembly_mass = None
        self.element_count = None
        
        # Extensible metadata dictionary for additional properties
        self.metadata = {}
    
    # ---------- Step Management ----------
    
    def add_step(self, step_name, elements):
        """Add elements for a specific analysis step"""
        if step_name not in self.steps:
            self.steps.append(step_name)
        self.elements_by_step[step_name] = elements
        return self  # Allow chaining
    
    def get_steps(self):
        """Get list of all step names"""
        return list(self.steps)
    
    def has_step(self, step_name):
        """Check if step exists"""
        return step_name in self.elements_by_step
      
    # ---------- Element Queries ----------
    
    def get_elements(self, step):
        """Get all elements for a step"""
        return self.elements_by_step.get(step, [])
    
    def get_element_by_id(self, step, element_id):
        """Get a specific element by ID in a given step"""
        if step not in self.elements_by_step:
            return None
        for elem in self.elements_by_step[step]:
            if elem.element_id == element_id:
                return elem
        return None
    
    def get_elements_with_attribute(self, step, attribute_name):
        """Get all elements in a step that have a specific attribute"""
        if step not in self.elements_by_step:
            return []
        return [elem for elem in self.elements_by_step[step] 
                if elem.has_attribute(attribute_name)]
    
    def get_element_count(self, step):
        """Get number of elements in a step"""
        if step in self.elements_by_step:
            return len(self.elements_by_step[step])
        return 0
    
    # ---------- Analysis Properties ----------
    
    def set_max_stress(self, value):
        """Set maximum stress value"""
        self.max_stress = value
        return self
    
    def set_assembly_mass(self, value):
        """Set assembly mass"""
        self.assembly_mass = value
        return self
    
    def set_element_count(self, value):
        """Set assembly mass"""
        self.element_count = value
    
    def add_metadata(self, key, value):
        """Add custom metadata"""
        self.metadata[key] = value
        return self
    
    def get_metadata(self, key, default=None):
        """Get metadata value"""
        return self.metadata.get(key, default)
    
    # ---------- Serialization ----------
    
    def to_dict(self):
        """Serialize to dictionary"""
        result = {
            "job_name": self.job_name,
            "steps": self.steps,
            "elements_by_step": {
                step: [elem.to_dict() for elem in elements]
                for step, elements in self.elements_by_step.items()
            }
        }
        
        # Add optional fields only if set
        if self.max_stress is not None:
            result["max_stress"] = self.max_stress
        if self.assembly_mass is not None:
            result["assembly_mass"] = self.assembly_mass
        if self.element_count is not None:
            result["element_count"] = self.element_count
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @staticmethod
    def from_dict(d):
        """Deserialize from dictionary"""
        output = ModelOutput(d["job_name"])
        output.steps = d.get("steps", [])
        
        # Deserialize elements
        for step, elements_data in d.get("elements_by_step", {}).items():
            output.elements_by_step[step] = [
                Element.from_dict(elem_data) for elem_data in elements_data
            ]
        
        # Restore optional fields
        output.max_stress = d.get("max_stress")
        output.assembly_mass = d.get("assembly_mass")
        output.element_count = d.get("element_count")
        output.metadata = d.get("metadata", {})
        
        return output


# # ----------------------------------------------------------------------------------------------------------------------------------------
# # EXAMPLE USAGE
# # ----------------------------------------------------------------------------------------------------------------------------------------

# if __name__ == "__main__":
#     # Create model output
#     output = ModelOutput("bracket_analysis")
    
#     # Create elements with attributes
#     elem1 = Element(1)
#     elem1.add_attribute("location", Location(0, 0, 0))
#     elem1.add_attribute("stress", Stress(150.5))
#     elem1.add_attribute("displacement", Displacement(0.1, 0.2, 0.3))
    
#     elem2 = Element(2)
#     elem2.add_attribute("location", Location(1, 0, 0))
#     elem2.add_attribute("stress", Stress(200.0))
    
#     # Add data to output
#     output.add_step("Step-1", [elem1, elem2])
#     output.set_max_stress(200.0)
#     output.set_assembly_mass(15.5)
#     output.add_metadata("solver", "Abaqus")
#     output.add_metadata("mesh_size", 0.01)
    
#     # Query data
#     print("Steps:", output.get_steps())
#     print("Element count:", output.get_element_count("Step-1"))
#     print("Max stress:", output.max_stress)
    
#     # Find elements with stress
#     stressed = output.get_elements_with_attribute("Step-1", "stress")
#     print("Elements with stress:", len(stressed))
    
#     # Serialize and restore
#     data = output.to_dict()
#     restored = ModelOutput.from_dict(data)
#     print("Restored job:", restored.job_name)
#     print("Restored metadata:", restored.metadata)