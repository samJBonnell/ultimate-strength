"""
Each of these will be added within the abaqus_imports.py file to add more functions as more models are made

Index of each model type:
Model_01: Variable thickness, varibly patch-loaded simply supported panel
Model_02: Single direction stiffened panel subjected to axial force
    Usage: Buckling analysis - eigen and riks
Model_03: Complete stiffened panel subjected to uniform hydrostatic pressure

"""

class ModelClass(object):
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

# ----------------------------------------------------------------------------------------------------------------------------------
#   FLAT PANEL - CNN_FLAT & POD_MLP_FLAT 
# ----------------------------------------------------------------------------------------------------------------------------------

@ModelClass.register("model_01")
class Model_01(ModelClass):
    def __init__(self, model_name, job_name, width, length, t_panel,
                 pressure, pressure_location, pressure_patch_size, mesh_plate,
                 numCpus=1, numGpus=0):
        super(Model_01, self).__init__(model_name, job_name, numCpus, numGpus)
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
    
# ----------------------------------------------------------------------------------------------------------------------------------
#  SINGLE DIRECTION STIFFENED PANEL - BUCKLING EIGEN, BUCKLING RIKS
# ----------------------------------------------------------------------------------------------------------------------------------

@ModelClass.register("model_02")
class Model_02(ModelClass):
    def __init__(   self, 
                    model_name, 
                    job_name,
                    job_type,

                    # Geometry 
                    width,
                    length,
                    num_longitudinal,

                    # Thickness
                    t_panel,
                    t_longitudinal_web,
                    t_longitudinal_flange,
                    
                    # Internal dimensions
                    h_longitudinal_web,
                    w_longitudinal_flange,
                    
                    # Boundary conditions
                    axial_force,
                    
                    # Mesh
                    mesh_plate,
                    mesh_longitudinal_web,
                    mesh_longitudinal_flange,

                    # Modifiers
                    centroid,

                    # Model parameters
                    numCpus=1,
                    numGpus=0
                ):
        super(Model_02, self).__init__(model_name, job_name, numCpus, numGpus)
        # Simulation information
        self.job_type = job_type
        
        # Geometry
        self.width = width
        self.length = length
        self.num_longitudinal = num_longitudinal

        # Thickness
        self.t_panel = t_panel
        self.t_longitudinal_web = t_longitudinal_web
        self.t_longitudinal_flange = t_longitudinal_flange
        
        # Internal dimensions
        self.h_longitudinal_web = h_longitudinal_web
        self.w_longitudinal_flange = w_longitudinal_flange

        # Boundary conditions
        self.axial_force = axial_force

        # Mesh
        self.mesh_plate = mesh_plate
        self.mesh_longitudinal_web = mesh_longitudinal_web
        self.mesh_longitudinal_flange = mesh_longitudinal_flange

        self.centroid = centroid

    @classmethod
    def from_dict(cls, d):
        return cls(
            
            # Variables inherited from the base class
            model_name=d["model_name"],
            job_name=d["job_name"],
            numCpus=d.get("numCpus", 1),
            numGpus=d.get("numGpus", 0),
            
            job_type = d["job_type"],

            # Geometry
            width=d["width"],
            length=d["length"],
            num_longitudinal = d['num_longitudinal'],
            
            # Thickness
            t_panel=d["t_panel"],
            t_longitudinal_web=d["t_longitudinal_web"],
            t_longitudinal_flange=d["t_longitudinal_flange"],
            
            # Internal dimensions
            h_longitudinal_web = d['h_longitudinal_web'],
            w_longitudinal_flange = d['w_longitudinal_flange'],

            # Boundary conditions
            axial_force = d['axial_force'],

            # Mesh
            mesh_plate=d["mesh_plate"],
            mesh_longitudinal_web=d["mesh_longitudinal_web"],
            mesh_longitudinal_flange=d["mesh_longitudinal_flange"],
        
            # Modifiers
            centroid = d['centroid']
        )

# ----------------------------------------------------------------------------------------------------------------------------------
#  DOUBLE DIRECTION STIFFENED PANEL - BUCKLING EIGEN, BUCKLING RIKS
# ----------------------------------------------------------------------------------------------------------------------------------

@ModelClass.register("model_03")
class Model_03(ModelClass):
    def __init__(   self, 
                    model_name, 
                    job_name,
                    job_type,

                    # Geometry 
                    width,
                    length,
                    num_transverse,

                    location_longitudinals,

                    # Thickness
                    t_panel,
                    t_longitudinal_web,
                    t_longitudinal_flange,
                    t_transverse_web,
                    t_transverse_flange,
                    
                    # Internal dimensions
                    h_longitudinal_web,
                    w_longitudinal_flange,
                    h_transverse_web,
                    w_transverse_flange,

                    # Boundary conditions
                    axial_force,
                    
                    # Mesh
                    mesh_plate,
                    mesh_longitudinal_web,
                    mesh_longitudinal_flange,
                    mesh_transverse_web,
                    mesh_transverse_flange,

                    # Modifiers
                    centroid,

                    # Model parameters
                    numCpus=1,
                    numGpus=0
                ):
        super(Model_03, self).__init__(model_name, job_name, numCpus, numGpus)
        # Simulation information
        self.job_type = job_type
        
        # Geometry
        self.width = width
        self.length = length
        self.num_transverse = num_transverse

        self.location_longitudinals = location_longitudinals

        # Thickness
        self.t_panel = t_panel
        self.t_longitudinal_web = t_longitudinal_web
        self.t_longitudinal_flange = t_longitudinal_flange
        self.t_transverse_web = t_transverse_web
        self.t_transverse_flange = t_transverse_flange

        # Internal dimensions
        self.h_longitudinal_web = h_longitudinal_web
        self.w_longitudinal_flange = w_longitudinal_flange
        self.h_transverse_web = h_transverse_web
        self.w_transverse_flange = w_transverse_flange
        
        # Boundary conditions
        self.axial_force = axial_force

        # Mesh
        self.mesh_plate = mesh_plate
        self.mesh_longitudinal_web = mesh_longitudinal_web
        self.mesh_longitudinal_flange = mesh_longitudinal_flange
        self.mesh_transverse_web = mesh_transverse_web
        self.mesh_transverse_flange = mesh_transverse_flange

        self.centroid = centroid

    @classmethod
    def from_dict(cls, d):
        return cls(
            
            # Variables inherited from the base class
            model_name=d["model_name"],
            job_name=d["job_name"],
            numCpus=d.get("numCpus", 1),
            numGpus=d.get("numGpus", 0),
            
            job_type = d["job_type"],

            # Geometry
            width=d["width"],
            length=d["length"],
            num_transverse = d['num_transverse'],

            location_longitudinals = d['location_longitudinals'],
            # Thickness
            t_panel=d["t_panel"],
            t_longitudinal_web=d["t_longitudinal_web"],
            t_longitudinal_flange=d["t_longitudinal_flange"],
            t_transverse_web = d['t_transverse_web'],
            t_transverse_flange = d['t_transverse_flange'],
        
            # Internal dimensions
            h_longitudinal_web = d['h_longitudinal_web'],
            w_longitudinal_flange = d['w_longitudinal_flange'],
            h_transverse_web = d['h_transverse_web'],
            w_transverse_flange = d['w_transverse_flange'],

            # Boundary conditions
            axial_force = d['axial_force'],

            # Mesh
            mesh_plate=d["mesh_plate"],
            mesh_longitudinal_web=d["mesh_longitudinal_web"],
            mesh_longitudinal_flange=d["mesh_longitudinal_flange"],
            mesh_transverse_web = d['mesh_transverse_web'],
            mesh_transverse_flange = d['mesh_transverse_flange'],


            # Modifiers
            centroid = d['centroid']
        )