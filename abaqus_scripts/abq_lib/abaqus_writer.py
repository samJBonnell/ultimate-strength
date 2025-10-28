import sys
import json
import gzip

from us_lib.abq_model.classes import ModelClass

# Handle type differences between Py2 and Py3
if sys.version_info[0] < 3:
    text_type = unicode
else:
    text_type = str

def clean_unicode(obj):
    """Recursively convert unicode to str in nested data (dicts, lists, tuples)."""
    if isinstance(obj, dict):
        return {clean_unicode(k): clean_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_unicode(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_unicode(i) for i in obj)
    elif isinstance(obj, text_type) and sys.version_info[0] < 3:
        return str(obj)
    else:
        return obj

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
    
def load_input(file_path, index=-1):
    """
    Load a ModelClass input from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        index: Line index to load (0-based). Use -1 for last line (default)
    
    Returns:
        ModelClass instance
    """
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError("No valid input found in file.")
    
    if index < -1 or index >= len(lines):
        raise ValueError("Index {} out of range. File has {} entries.".format(index, len(lines)))
    
    target_line = lines[index]
    
    data = json.loads(target_line)
    data = clean_unicode(data)
    return ModelClass.from_dict(data)


def load_last_input(file_path):
    """Load the last ModelInput from a JSONL file (convenience wrapper)"""
    return load_input(file_path, index=-1)