from typing import Generator, List, Optional, Set
from dataclasses import dataclass
import json
import random
from pathlib import Path
from tqdm import tqdm
from utils.IO_utils import PanelInput, PanelOutput

@dataclass
class Record:
    input: PanelInput
    output: PanelOutput

def load_random_records(input_path: Path, output_path: Path, n: int) -> List[Record]:
    """Load n random records from input and output files."""
    # Load all inputs once
    all_inputs = {}
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_inputs[data["id"]] = data
    
    all_ids = list(all_inputs.keys())
    if len(all_ids) < n:
        raise ValueError(f"Not enough records. Requested {n}, but only {len(all_ids)} available.")
    
    sampled_ids = set(random.sample(all_ids, n))
    
    # Reuse stream_records with filtered IDs
    return list(stream_records(input_path, output_path, filter_ids=sampled_ids))

def stream_records(input_path: Path, output_path: Path, filter_ids: Optional[Set[str]] = None) -> Generator[Record, None, None]:
    """Stream records from input and output files, optionally filtering by IDs."""
    input_map = {}
    
    # Read all inputs first
    with input_path.open("r") as f_in:
        for line in tqdm(f_in, desc="Reading input.jsonl"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                id_ = data["id"]
                if (filter_ids is None) or (id_ in filter_ids):
                    # Create PanelInput using explicit constructor
                    panel_input = PanelInput(
                        id=data["id"],
                        num_longitudinal=data["num_longitudinal"],
                        width=data["width"],
                        length=data["length"],
                        t_panel=data["t_panel"],
                        t_longitudinal_web=data["t_longitudinal_web"],
                        t_longitudinal_flange=data["t_longitudinal_flange"],
                        h_longitudinal_web=data["h_longitudinal_web"],
                        w_longitudinal_flange=data["w_longitudinal_flange"],
                        axial_force=data["axial_force"],
                        mesh_plate=data["mesh_plate"],
                        mesh_longitudinal_web=data["mesh_longitudinal_web"],
                        mesh_longitudinal_flange=data["mesh_longitudinal_flange"]
                    )
                    input_map[id_] = panel_input
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping invalid input record: {e}")
                continue
    
    # Stream outputs and match with inputs
    with output_path.open("r") as f_out:
        for line in tqdm(f_out, desc="Reading output.jsonl"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                id_ = data.get("id")
                if id_ in input_map:
                    output = PanelOutput.from_dict(data)
                    yield Record(input=input_map[id_], output=output)
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping invalid output record: {e}")
                continue

def load_all_records(input_path: Path, output_path: Path) -> List[Record]:
    """Load all available records from input and output files."""
    return list(stream_records(input_path, output_path))

def get_record_count(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """
    Get counts of input records, output records, and matched records.
    Returns: (input_count, output_count, matched_count)
    """
    input_ids = set()
    output_ids = set()
    
    # Count inputs
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    input_ids.add(data["id"])
                except (KeyError, json.JSONDecodeError):
                    continue
    
    # Count outputs
    with output_path.open("r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    output_ids.add(data.get("id"))
                except (KeyError, json.JSONDecodeError):
                    continue
    
    matched_count = len(input_ids & output_ids)
    return len(input_ids), len(output_ids), matched_count

# # Example usage:
# if __name__ == "__main__":
#     input_path = Path("input.jsonl")
#     output_path = Path("output.jsonl")
    
#     # Check record counts
#     input_count, output_count, matched_count = get_record_count(input_path, output_path)
#     print(f"Input records: {input_count}")
#     print(f"Output records: {output_count}")
#     print(f"Matched records: {matched_count}")
    
#     # Load 10 random records
#     try:
#         random_records = load_random_records(input_path, output_path, 10)
#         print(f"Loaded {len(random_records)} random records")
        
#         # Example: access first record
#         if random_records:
#             first_record = random_records[0]
#             print(f"First record ID: {first_record.input.id}")
#             print(f"Panel width: {first_record.input.width}")
#             print(f"Job name: {first_record.output.job_name}")
#     except ValueError as e:
#         print(f"Error: {e}")
    
#     # Stream all records (memory efficient for large datasets)
#     print("\nStreaming all records:")
#     record_count = 0
#     for record in stream_records(input_path, output_path):
#         record_count += 1
#         if record_count <= 3:  # Show first 3 records
#             print(f"Record {record_count}: ID={record.input.id}, Width={record.input.width}")
    
#     print(f"Total records streamed: {record_count}")