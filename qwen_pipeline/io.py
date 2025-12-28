"""I/O helpers for the Qwen pipeline."""

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from utils import get_test_question, load_csv_data

from .types import RetrievalRecord


def iter_examples(test_file: str) -> Iterator[Tuple[int, Dict[str, str]]]:
    rows, header = load_csv_data(test_file)
    for idx in range(len(rows)):
        yield idx, get_test_question(idx, rows, header)


def load_retrieval_results(path: str) -> Dict[str, Dict]:
    with open(path, "r") as f:
        return json.load(f)


def build_retrieval_record(
    example: Dict[str, str],
    data_id: str,
    blob: Dict,
    image_path: Optional[str],
    candidate_ids: Sequence[str],
) -> RetrievalRecord:
    dataset = example.get("dataset_name") or "unknown"
    image_id = example.get("dataset_image_ids", "")
    candidate_urls: List[str] = blob.get("initial_retrieved_entries") or blob.get("retrieved_entries", [])
    reranked_urls: List[str] = blob.get("retrieved_entries", []) or candidate_urls
    reranked_sections: List[str] = blob.get("reranked_sections", [])
    return RetrievalRecord(
        data_id=data_id,
        question=example["question"],
        image_path=image_path,
        dataset_name=dataset,
        candidate_urls=candidate_urls,
        reranked_urls=reranked_urls,
        reranked_sections=reranked_sections,
        meta={
            "retrieval_similarities": blob.get("retrieval_similarities", []),
            "initial_retrieved_entries": blob.get("initial_retrieved_entries", []),
            "retrieved_entries": blob.get("retrieved_entries", []),
            "image_id": image_id,
            "original_data_id": example.get("data_id"),
            "candidate_data_ids": list(candidate_ids),
        },
    )


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def load_metadata(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
