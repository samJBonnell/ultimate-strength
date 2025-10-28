# Import package code
import json
import gzip
import random
from tqdm import tqdm
from pathlib import Path
from typing import Generator, List, Optional, Set, Tuple

# Import personal code
from us_lib.abq_model.classes import ModelClass
from us_lib.abq_model.output import ModelOutput
from us_lib.data.record import Record

def stream_records(input_path: Path, output_path: Path, filter_ids: Optional[Set[str]] = None) -> Generator[Record, None, None]:
    """
    Stream records from input and output files, optionally filtering by IDs.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file  
        filter_ids: Optional set of IDs to filter by
        
    Yields:
        Record objects containing matched input/output pairs
    """
    input_map = {}
   
    # Read all inputs first
    with input_path.open("r") as f_in:
        for line in tqdm(f_in, desc="Reading inputs"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Use job_name as ID, or fallback to model_name
                id_ = data.get("job_name") or data.get("model_name")
                if (filter_ids is None) or (id_ in filter_ids):
                    model_input = ModelClass.from_dict(data)
                    input_map[id_] = model_input
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping invalid input record: {e}")
                continue
   
    # Stream outputs and match with inputs
    with output_path.open("r") as f_out:
        for line in tqdm(f_out, desc="Processing outputs"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                id_ = data.get("job_name") or data.get("model_name")
                if id_ in input_map:
                    output = ModelOutput.from_dict(data)
                    yield Record(input_map[id_], output)
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping invalid output record: {e}")
                continue

def load_records(input_path: Path, output_path: Path) -> List[Record]:
    """
    Load all records from input and output files.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        
    Returns:
        List of Record objects
    """
    return list(stream_records(input_path, output_path))

def load_random_records(input_path: Path, output_path: Path, n: int) -> List[Record]:
    """
    Load n random records from input and output files.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        n: Number of records to load
        
    Returns:
        List of Record objects
    """
    records = list(stream_records(input_path, output_path))
    if len(records) <= n:
        return records
    return random.sample(records, n)

def get_record_count(input_path: Path, output_path: Path) -> Tuple[int, int, int]:
    """
    Get counts of input records, output records, and matched records.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        
    Returns:
        Tuple of (input_count, output_count, matched_count)
    """
    input_ids = set()
    output_ids = set()
   
    # Count inputs
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    input_ids.add(data["job_name"])
                except (KeyError, json.JSONDecodeError):
                    continue
   
    # Count outputs
    with output_path.open("r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    output_ids.add(data.get("job_name"))
                except (KeyError, json.JSONDecodeError):
                    continue
   
    matched_count = len(input_ids & output_ids)
    return len(input_ids), len(output_ids), matched_count