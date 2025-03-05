"""
Module for reading inputs.
They should be jsonl files with the format
{"text": "<bos_token>text to be completed", "id": "some unique id (ex. uuid)"}
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Generator, List, Optional, Set, TextIO
from tqdm import tqdm


@dataclass(frozen=True)
class Batch:
    """
    A batch of inputs.
    """

    texts: List[str]
    ids: List[str]


def read_input_batches(
    input_path: Path, batch_size: int, completed_ids: Optional[Set[str]] = None
) -> Generator[Batch, None, None]:
    """
    Read inputs from a jsonl file and yield them in batches. Skips any inputs with ids in completed_ids.
    """
    batch_texts = []
    batch_ids = []
    with open(input_path, "r") as f:
        for line in f:
            input = json.loads(line)
            if completed_ids and input["id"] in completed_ids:
                continue
            batch_texts.append(input["text"])
            batch_ids.append(input["id"])
            if len(batch_texts) == batch_size:
                yield Batch(batch_texts, batch_ids)
                batch_texts = []
                batch_ids = []
    if batch_texts:
        yield Batch(batch_texts, batch_ids)


def read_ids(input_path: Path) -> Set[str]:
    """
    Read ids from a jsonl file and return them in a set.
    """
    ids = set()
    with open(input_path, "r") as f:
        for line in tqdm(f, desc="Reading existing output ids"):
            input = json.loads(line)
            ids.add(input["id"])
    return ids


def write_outputs(output_file: TextIO, batch: Batch, outputs: List[str]) -> None:
    """
    Write outputs to a jsonl file.
    """
    for (
        output,
        input_text,
        _id,
    ) in zip(outputs, batch.texts, batch.ids):
        output_file.write(
            json.dumps({"input_text": input_text, "id": _id, "output": output}) + "\n"
        )
