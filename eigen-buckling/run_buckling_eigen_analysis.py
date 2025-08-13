from utils.FiniteElementModel import FiniteElementModel
from models.IO_buckling_eigen import PanelInput, PanelOutput

UC1 = PanelInput(
    id = "",

    # Global Geometry
    num_longitudinal = 1,

    width = 0.6,
    length = 0.6,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.078,
    t_longitudinal_flange = 0.004,

    # Local stiffener geometry
    h_longitudinal_web = 0.125,
    w_longitudinal_flange = 0.100,

    # Applied Pressure
    axial_force = 10000000,

    # Mesh Settings
    mesh_plate = 0.025,
    mesh_longitudinal_web = 0.025,
    mesh_longitudinal_flange = 0.025
)

panel = PanelInput(
    id = "",

    # Global Geometry
    num_longitudinal = 4,

    width = 3.0,
    length = 3.0,

    # Thickness List
    t_panel = 0.010,
    t_longitudinal_web = 0.078,
    t_longitudinal_flange = 0.004,

    # Local stiffener geometry
    h_longitudinal_web = 0.125,
    w_longitudinal_flange = 0.100,

    # Applied Pressure
    axial_force = 10000000,

    # Mesh Settings
    mesh_plate = 0.025,
    mesh_longitudinal_web = 0.025,
    mesh_longitudinal_flange = 0.025
)

fem = FiniteElementModel("models\\buckling_eigen_panel.py", "data\\input.jsonl", "data\\output.jsonl", PanelInput, PanelOutput)
fem.write(panel)
fem.run()

#fem = FiniteElementModel("models\\buckling_eigen_uc.py", "data\\input.jsonl", "data\\output.jsonl", PanelInput, PanelOutput)
#fem.write(UC1)
#fem.run()