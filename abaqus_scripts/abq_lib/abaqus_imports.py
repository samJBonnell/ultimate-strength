"""
Wrapper to handle imports for ABAQUS scripts.
"""
import sys
from os.path import join, dirname, abspath, exists
import os

# Get project root
# This file is in: ultimate-strength/abaqus_scripts/abq_lib/abaqus_imports.py
try:
    # Try __file__ first
    this_file = abspath(__file__)
except NameError:
    # Fallback for ABAQUS noGUI
    this_file = join(sys.path[0], 'abq_lib', 'abaqus_imports.py')

abq_lib_dir = dirname(this_file)          # abaqus_scripts/abq_lib/
abaqus_dir = dirname(abq_lib_dir)         # abaqus_scripts/
project_root = dirname(abaqus_dir)        # ultimate-strength/

# Also check environment variable
env_project_root = os.environ.get('PROJECT_ROOT')
if env_project_root:
    project_root = env_project_root

# Add project root to path for us_lib imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ----------------------------------------------
# IMPORT THE REQUIRED LIBRARIES
# ----------------------------------------------
try:
    from us_lib.abq_model.classes import ModelClass
    from us_lib.abq_model.output import ModelOutput, Element, Stress

    from us_lib.abq_model.classes import FlatPanel
        
except ImportError as e:
    print("Import error: {}".format(e))
    print("sys.path: {}".format(sys.path[:5]))
    raise

__all__ = [
    'ModelClass', 
    'ModelOutput', 
    'Element', 
    'Stress', 
    'FlatPanel', 
    'write_trial_ndjson', 
    'load_last_input'
]