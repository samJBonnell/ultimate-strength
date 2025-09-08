from typing import Generator
from dataclasses import dataclass
from typing import List
import json
import random
from pathlib import Path
from tqdm import tqdm

from models.IO_buckling import PanelInput, PanelOutput

@dataclass
class Record:
    input: PanelInput
    output: PanelOutput

def load_random_records(input_path: Path, output_path: Path, n: int) -> List[Record]:
    # Load all inputs once
    all_inputs = {}
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_inputs[data["id"]] = data
    all_ids = list(all_inputs.keys())
    if len(all_ids) < n:
        raise ValueError("Not enough records.")

    sampled_ids = set(random.sample(all_ids, n))
    # Reuse stream_records with filtered IDs
    return list(stream_records(input_path, output_path, filter_ids=sampled_ids))

def stream_records(input_path: Path, output_path: Path, filter_ids: set = None) -> Generator[Record, None, None]:
    input_map = {}
    with input_path.open("r") as f_in:
        for line in tqdm(f_in, desc="Reading input.jsonl"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                id_ = data["id"]
                if (filter_ids is None) or (id_ in filter_ids):
                    input_map[id_] = PanelInput(**data)
            except Exception:
                continue

    with output_path.open("r") as f_out:
        for line in tqdm(f_out, desc="Reading output.jsonl"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                id_ = data.get("id")
                if id_ in input_map:
                    yield Record(input=input_map[id_], output=PanelOutput.from_dict(data))
            except Exception:
                continue