from pathlib import Path
from typing import Type, TypeVar, Generic
import os
import json
import subprocess

T = TypeVar("T") # ModelInput object
U = TypeVar("U") # ModelOutput object

def to_dict_recursive(obj):
    """Recursively convert for JSON serialization"""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = to_dict_recursive(value)
        return result
    elif isinstance(obj, list):
        return [to_dict_recursive(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict_recursive(v) for k, v in obj.items()}
    else:
        return obj

class ModelWrapper(Generic[T, U]):
    def __init__(self,
                 model: str,
                 input_path: str,
                 output_path: str,
                 input_class: Type[T],
                 output_class: Type[U]):
        self.model = model
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.input_class = input_class
        self.output_class = output_class

    def write(self, input_instance: T):
        with self.input_path.open("a") as f:
            json.dump(to_dict_recursive(input_instance), f)
            f.write("\n")

    def run(self):
        with open("abaqus_log.txt", "w") as log_file:
            # Set environment variable with project root
            env = os.environ.copy()
            env['PROJECT_ROOT'] = str(self.input_path.parent.parent)  # Go up from data/ to project root
            print(str(self.input_path.parent.parent))

            functionCall = subprocess.Popen([
                "abaqus", "cae", "noGUI={}".format(self.model)
            ], stdout=log_file, stderr=log_file, shell=True, env=env)
            functionCall.communicate()