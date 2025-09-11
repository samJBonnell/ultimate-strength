from pathlib import Path
from typing import Type, TypeVar, Generic
import json
import subprocess
import uuid
import datetime

# Import from your new location
from utils.IO_utils import PanelInput, PanelOutput

T = TypeVar("T")
U = TypeVar("U")

def to_dict(obj):
    """Convert an object to dictionary, handling both dataclass-like and regular objects."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        # For regular classes, convert __dict__ but handle nested objects
        result = {}
        for key, value in obj.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
                result[key] = to_dict(value)
            elif isinstance(value, list):
                result[key] = [to_dict(item) if hasattr(item, '__dict__') else item for item in value]
            elif isinstance(value, dict):
                result[key] = {k: to_dict(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
            else:
                result[key] = value
        return result
    else:
        return obj

class FiniteElementModel(Generic[T, U]):
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
        self.history = []

    def write(self, input_instance: T):
        with self.input_path.open("a") as f:
            json.dump(to_dict(input_instance), f)
            f.write("\n")

    def run(self):
        with open("abaqus_log.txt", "w") as log_file:
            functionCall = subprocess.Popen([
                "abaqus", "cae", f"noGUI={self.model}"
            ], stdout=log_file, stderr=log_file, shell=True)
            functionCall.communicate()

    def read(self, input_id: str) -> U:
        with self.output_path.open("r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("id") == input_id:
                    # Use from_dict if available, otherwise try direct instantiation
                    if hasattr(self.output_class, 'from_dict'):
                        return self.output_class.from_dict(data)
                    else:
                        return self.output_class(**data)
        raise ValueError(f"Output with id={input_id} not found.")

    def evaluate(self, input_instance: T) -> dict:
        if not hasattr(input_instance, "id") or not input_instance.id:
            input_instance.id = str(uuid.uuid4())
        
        self.write(input_instance)
        self.run()
        output = self.read(input_instance.id)
        
        record = {
            "id": input_instance.id,
            "timestamp": datetime.datetime.now().isoformat(),
            "input": input_instance,
            "output": output,
            "status": "success"
        }
        self.history.append(record)
        return record

    def save_history(self, file_path: str):
        with open(file_path, "a") as f:
            for record in self.history:
                f.write(json.dumps({
                    "id": record["id"],
                    "timestamp": record["timestamp"],
                    "input": to_dict(record["input"]),
                    "output": to_dict(record["output"]),
                    "status": record["status"]
                }) + "\n")