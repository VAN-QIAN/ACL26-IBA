"""Utility to add identification probability scores to existing metadata.

This script replays the new Qwen identification scoring pass without
regenerating metadata from scratch. It loads an existing metadata JSONL,
runs :meth:`QwenVLModel.score_candidates` on the ranked entities, and writes
an updated JSONL containing the probability fields plus optional section
score weightings. If desired, retrieval similarities from the initial
retrieval step can be included in the scoring prompt through a command-line
flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Local import to avoid depending on the external package name.
from ..pipeline import QwenVLModel  # noqa: F401  (re-exported wrapper)


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def _build_probability_maps(row: Dict, probabilities: List[Optional[float]]) -> Dict[str, float]:
    """Create a URL -> probability lookup aligned with ranked results."""
    lookup: Dict[str, float] = {}
    ranked_urls = row.get("identification_ranked_urls") or []
    for url, prob in zip(ranked_urls, probabilities):
        if prob is None:
            continue
        try:
            lookup[url] = float(prob)
        except (TypeError, ValueError):
            continue
    return lookup


def rescore_metadata(
    metadata_rows: List[Dict],
    scorer: QwenVLModel,
    *,
    score_top_k: int,
    max_new_tokens: int,
    temperature: float,
    use_similarity: bool,
) -> Dict[str, int]:
    processed = 0
    skipped = 0
    failures = 0
    for row in metadata_rows:
        ranked_titles: List[str] = row.get("identification_ranked_titles") or []
        image_path = row.get("image_path")
        if not ranked_titles or not image_path:
            skipped += 1
            continue
        top_k = min(max(1, score_top_k), len(ranked_titles))
        candidate_titles = ranked_titles[:top_k]
        candidate_similarities: Optional[List[Optional[float]]] = None
        if use_similarity:
            similarity_lookup: Dict[str, float] = {}
            retrieval_meta = row.get("retrieval_meta")
            if isinstance(retrieval_meta, dict):
                raw_similarities = retrieval_meta.get("retrieval_similarities") or []
                candidate_urls = row.get("candidate_urls") or []
                for url, sim in zip(candidate_urls, raw_similarities):
                    try:
                        similarity_lookup[url] = float(sim)
                    except (TypeError, ValueError):
                        continue
            if similarity_lookup:
                ranked_urls = row.get("identification_ranked_urls") or []
                candidate_similarities = [
                    similarity_lookup.get(url) for url in ranked_urls[:top_k]
                ]

        try:
            scores = scorer.score_candidates(
                image_path=image_path,
                candidate_titles=candidate_titles,
                question=row.get("question"),
                candidate_similarities=candidate_similarities,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            print(
                f"[warn] Failed to score data_id={row.get('data_id')} ({exc}). Leaving row unchanged.",
                file=sys.stderr,
            )
            continue

        probability_map = {candidate.title: candidate.probability for candidate in scores.candidates}
        ranked_probabilities: List[Optional[float]] = []
        for title in ranked_titles:
            ranked_probabilities.append(probability_map.get(title))
        row["identification_ranked_probabilities"] = ranked_probabilities
        row["identification_none_probability"] = scores.none_probability

        # Persist details alongside retrieval metadata for reproducibility.
        detail_records = [
            {
                "title": candidate.title,
                "probability": candidate.probability,
                "ordered_rank": index,
            }
            for index, candidate in enumerate(scores.candidates)
        ]
        retrieval_meta = row.get("retrieval_meta")
        if isinstance(retrieval_meta, dict):
            retrieval_meta["identification_scores"] = detail_records
            retrieval_meta["identification_none_probability"] = scores.none_probability
            retrieval_meta["identification_scores_raw_response"] = scores.raw_response
        else:
            row["retrieval_meta"] = {
                "identification_scores": detail_records,
                "identification_none_probability": scores.none_probability,
                "identification_scores_raw_response": scores.raw_response,
            }

        # Store weighting helper for downstream reranking (do not mutate existing ordering).
        probability_by_url = _build_probability_maps(row, ranked_probabilities)
        base_section_scores: Optional[List[float]] = row.get("section_scores")
        section_entity_urls: Optional[List[str]] = row.get("section_entity_urls")
        if (
            isinstance(base_section_scores, list)
            and isinstance(section_entity_urls, list)
            and len(base_section_scores) == len(section_entity_urls)
        ):
            weighted_scores: List[Optional[float]] = []
            for score, url in zip(base_section_scores, section_entity_urls):
                probability = probability_by_url.get(url)
                if probability is None:
                    weighted_scores.append(None if score is None else float(score))
                else:
                    try:
                        weighted_scores.append(float(score) * float(probability))
                    except (TypeError, ValueError):
                        weighted_scores.append(None)
            row["section_scores_weighted_by_identification"] = weighted_scores
        processed += 1
    return {"processed": processed, "skipped": skipped, "failed": failures}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Qwen identification scoring on existing metadata JSONL."
    )
    parser.add_argument("--metadata_path", type=str, required=True, help="Input metadata JSONL.")
    parser.add_argument("--output_path", type=str, required=True, help="Destination for updated JSONL.")
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen_device", type=str, default="cuda:0")
    parser.add_argument(
        "--score_top_k",
        type=int,
        default=3,
        help="Number of ranked options to score (defaults to 3).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--use_retrieval_similarity",
        action="store_true",
        help="Include retrieval similarities in the scoring prompt.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    output_path = Path(args.output_path)
    rows = _load_jsonl(metadata_path)

    scorer = QwenVLModel(
        model_name=args.qwen_model_name,
        device=args.qwen_device,
    )
    stats = rescore_metadata(
        rows,
        scorer,
        score_top_k=args.score_top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_similarity=args.use_retrieval_similarity,
    )
    _write_jsonl(output_path, rows)
    print(
        f"Scoring complete | processed={stats['processed']} | skipped={stats['skipped']} | failed={stats['failed']} "
        f"-> {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
