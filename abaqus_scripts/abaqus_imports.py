"""
Wrapper to handle imports for ABAQUS scripts.
"""
import sys
from os.path import dirname, abspath

# Get directories
script_dir = dirname(abspath(__file__))  # abaqus_scripts/
project_root = dirname(script_dir)       # ultimate-strength/

# Add both to path
if project_root not in sys.path:
    sys.path.insert(0, project_root)     # For us_lib
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)       # For abq_lib

# Import from local abq_lib
from abq_lib.FEMPipeline import FEMPipeline
from abq_lib.mesh_utilities import *

# Import from us_lib
from us_lib.input_output import ModelInput, ModelOutput

__all__ = ['FEMPipeline', 'ModelInput', 'ModelOutput']