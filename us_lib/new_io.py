class ModelInput(object):
    """Base class for all model inputs"""
    _registry = {}
   
    def __init__(self, model_name, job_name, numCpus=1, numGpus=0):
        self.model_name = model_name
        self.job_name = job_name
        self.numCpus = numCpus
        self.numGpus = numGpus
   
    @classmethod
    def register(cls, model_name):
        def decorator(subclass):
            cls._registry[model_name] = subclass
            return subclass
        return decorator
   
    @classmethod
    def from_dict(cls, d):
        model_name = d.get("model_name")
        if model_name in cls._registry:
            return cls._registry[model_name].from_dict(d)
        raise ValueError("Unknown model_name: {}".format(model_name))
   
    def to_dict(self):
        """Automatically convert all instance attributes to dict"""
        return self.__dict__.copy()
   
    def copy(self, **kwargs):
        """Create a copy with updated fields"""
        current = self.to_dict()
        current.update(kwargs)
        return self.__class__.from_dict(current)
   
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.to_dict())

@ModelInput.register("flat_panel")
class FlatPanelInput(ModelInput):
    def __init__(self, model_name, job_name, width, length, t_panel,
                 pressure, pressure_location, pressure_patch_size, mesh_plate,
                 numCpus=1, numGpus=0):
        super(FlatPanelInput, self).__init__(model_name, job_name, numCpus, numGpus)
        self.width = width
        self.length = length
        self.t_panel = t_panel
        self.pressure = pressure
        self.pressure_location = pressure_location
        self.pressure_patch_size = pressure_patch_size
        self.mesh_plate = mesh_plate
   
    @classmethod
    def from_dict(cls, d):
        # Only pass parameters that __init__ expects
        return cls(
            model_name=d["model_name"],
            job_name=d["job_name"],
            width=d["width"],
            length=d["length"],
            t_panel=d["t_panel"],
            pressure=d["pressure"],
            pressure_location=d["pressure_location"],
            pressure_patch_size=d["pressure_patch_size"],
            mesh_plate=d["mesh_plate"],
            
            # Variables inherited from the base class
            numCpus=d.get("numCpus", 1),
            numGpus=d.get("numGpus", 0),
        )

@ModelInput.register("stiffened_panel")
class StiffenedPanelInput(ModelInput):
    def __init__(self, model_name, job_name, width, length,
                 t_panel, t_transverse_web, t_transverse_flange,
                 t_longitudinal_web, t_longitudinal_flange,
                 pressure, mesh_plate, numCpus=1, numGpus=0):
        super(StiffenedPanelInput, self).__init__(model_name, job_name, numCpus, numGpus)
        self.width = width
        self.length = length
        self.t_panel = t_panel
        self.t_transverse_web = t_transverse_web
        self.t_transverse_flange = t_transverse_flange
        self.t_longitudinal_web = t_longitudinal_web
        self.t_longitudinal_flange = t_longitudinal_flange
        self.pressure = pressure
        self.mesh_plate = mesh_plate
   
    @classmethod
    def from_dict(cls, d):
        return cls(
            model_name=d["model_name"],
            job_name=d["job_name"],
            width=d["width"],
            length=d["length"],
            t_panel=d["t_panel"],
            t_transverse_web=d["t_transverse_web"],
            t_transverse_flange=d["t_transverse_flange"],
            t_longitudinal_web=d["t_longitudinal_web"],
            t_longitudinal_flange=d["t_longitudinal_flange"],
            pressure=d["pressure"],
            mesh_plate=d["mesh_plate"],

            # Variables inherited from the base class
            numCpus=d.get("numCpus", 1),
            numGpus=d.get("numGpus", 0),
        )