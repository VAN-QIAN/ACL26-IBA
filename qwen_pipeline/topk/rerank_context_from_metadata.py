"""Recompute context sections for existing metadata using weighted fusion.

This utility reorders stored sections in a metadata JSONL (produced by the
top-k prepare stage) by combining three signals:

1. Section reranker score (EchoSight/BGE etc.)
2. Initial retrieval similarity
3. Identification probability (from Qwen identification rescoring)

The user can control the weights and the fusion mode for the identification
probability. The selected section text and related metadata fields are updated
before writing a new JSONL file.
"""

from __future__ import annotations

import math
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def _safe_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_index(values: Sequence[Optional[float]]) -> Optional[int]:
    best_value = float("-inf")
    best_idx: Optional[int] = None
    for idx, value in enumerate(values):
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_idx = idx
    return best_idx


def _normalize_scores(scores: Sequence[Optional[float]]) -> List[Optional[float]]:
    valid = [
        float(score)
        for score in scores
        if score is not None and isinstance(score, (int, float)) and math.isfinite(score)
    ]
    if not valid:
        return [None] * len(scores)
    min_val = min(valid)
    max_val = max(valid)
    if math.isclose(min_val, max_val):
        return [1.0 if score is not None else None for score in scores]
    span = max_val - min_val
    normalized: List[Optional[float]] = []
    for score in scores:
        if score is None or not isinstance(score, (int, float)) or not math.isfinite(score):
            normalized.append(None)
            continue
        value = (float(score) - min_val) / span
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        normalized.append(value)
    return normalized


def rerank_row(
    row: Dict,
    *,
    section_weight: float,
    retrieval_weight: float,
    identification_weight: float,
    identification_mode: str,
    section_score_field: str,
) -> Tuple[bool, Optional[int], List[Optional[float]]]:
    sections: List[str] = (
        row.get("qwen_reranked_sections")
        or row.get("reranked_sections")
        or []
    )
    if not sections:
        return False, None, []

    retrieval_meta = row.get("retrieval_meta") or {}
    if not isinstance(retrieval_meta, dict):
        retrieval_meta = {}

    section_entity_urls = list(
        retrieval_meta.get("section_entity_urls")
        or row.get("section_entity_urls")
        or []
    )

    ranked_urls = row.get("identification_ranked_urls") or []
    ranked_probs = row.get("identification_ranked_probabilities") or []
    probability_map = {}
    for url, prob in zip(ranked_urls, ranked_probs):
        val = _safe_float(prob)
        if url and val is not None:
            probability_map[url] = val

    section_scores = (
        retrieval_meta.get(section_score_field)
        or row.get(section_score_field)
    )
    score_field_used = section_score_field
    if not section_scores:
        fallback_order = [
            "section_scores_unweighted",
            "section_scores_echosight_raw",
            "section_scores",
        ]
        for field in fallback_order:
            candidate = retrieval_meta.get(field) or row.get(field)
            if candidate:
                section_scores = candidate
                score_field_used = field
                break
    base_scores: List[Optional[float]] = [
        _safe_float(score) for score in (section_scores or [])
    ]
    if len(base_scores) < len(sections):
        base_scores.extend([None] * (len(sections) - len(base_scores)))

    normalized_scores: Optional[List[Optional[float]]] = None
    normalized_field: Optional[str] = None
    if isinstance(section_scores, (list, tuple)):
        normalized_scores = _normalize_scores(list(section_scores))
        if len(normalized_scores) < len(sections):
            normalized_scores.extend([None] * (len(sections) - len(normalized_scores)))
        normalized_field = f"{score_field_used}_normalized"

    retrieval_similarities = retrieval_meta.get("section_retrieval_similarities") or row.get("section_retrieval_similarities")
    retrieval_values: List[Optional[float]] = [
        _safe_float(value) for value in (retrieval_similarities or [])
    ]
    if len(retrieval_values) < len(sections):
        retrieval_values.extend([None] * (len(sections) - len(retrieval_values)))

    identification_source = (
        retrieval_meta.get("section_entity_probabilities")
        or row.get("section_entity_probabilities")
        or []
    )
    identification_values: List[Optional[float]] = [
        _safe_float(value) for value in identification_source
    ]
    if len(identification_values) < len(sections):
        identification_values.extend([None] * (len(sections) - len(identification_values)))

    final_scores: List[Optional[float]] = []
    section_components: List[Optional[float]] = []
    for idx in range(len(sections)):
        base = base_scores[idx] if idx < len(base_scores) else None
        retrieval_value = retrieval_values[idx] if idx < len(retrieval_values) else None
        identification_value = identification_values[idx] if idx < len(identification_values) else None
        if identification_value is None and idx < len(section_entity_urls):
            identification_value = probability_map.get(section_entity_urls[idx])
        if idx < len(identification_values):
            identification_values[idx] = identification_value
        else:
            identification_values.append(identification_value)

        section_component = base
        if identification_mode == "multiply":
            if section_component is not None and identification_value is not None:
                section_component = section_component * identification_value

        weighted_sum = 0.0
        weight_applied = False

        if section_component is not None and section_weight != 0.0:
            weighted_sum += section_weight * section_component
            weight_applied = True
        if retrieval_value is not None and retrieval_weight != 0.0:
            weighted_sum += retrieval_weight * retrieval_value
            weight_applied = True
        if identification_value is not None and identification_weight != 0.0:
            if identification_mode == "add":
                weighted_sum += identification_weight * identification_value
            else:
                weighted_sum += identification_weight * identification_value
            weight_applied = True

        if weight_applied:
            final_scores.append(weighted_sum)
        else:
            final_scores.append(section_component)
        section_components.append(section_component)

    order = sorted(
        range(len(sections)),
        key=lambda idx: final_scores[idx] if final_scores[idx] is not None else float("-inf"),
        reverse=True,
    )
    reordered_sections = [sections[idx] for idx in order]
    final_scores = [final_scores[idx] for idx in order]
    section_components = [section_components[idx] for idx in order]
    retrieval_values = [retrieval_values[idx] for idx in order]
    identification_values = [identification_values[idx] for idx in order]
    if normalized_scores is not None:
        normalized_scores = [
            normalized_scores[idx] if idx < len(normalized_scores) else None
            for idx in order
        ]

    best_idx = _best_index(final_scores)
    if best_idx is None:
        return False, None, []

    section_titles = retrieval_meta.get("section_titles") or row.get("section_titles") or []
    entity_urls = retrieval_meta.get("section_entity_urls") or row.get("section_entity_urls") or []
    entity_titles = retrieval_meta.get("section_entity_titles") or row.get("section_entity_titles") or []
    entity_ranks = retrieval_meta.get("section_entity_ranks") or row.get("section_entity_ranks") or []

    section_titles = [section_titles[idx] for idx in order] if section_titles else []
    entity_urls = [entity_urls[idx] for idx in order] if entity_urls else []
    entity_titles = [entity_titles[idx] for idx in order] if entity_titles else []
    entity_ranks = [entity_ranks[idx] for idx in order] if entity_ranks else []

    row["context_mode"] = "section"
    row["context_text"] = reordered_sections[best_idx]
    if best_idx < len(section_titles):
        row["context_section_title"] = section_titles[best_idx]
    if best_idx < len(entity_urls):
        row["context_source_url"] = entity_urls[best_idx]
    if best_idx < len(entity_titles):
        row["selected_title"] = entity_titles[best_idx]
    if best_idx < len(entity_ranks):
        row["context_source_rank"] = entity_ranks[best_idx]

    # Update ordering stored in metadata for downstream consumers.
    row["qwen_reranked_sections"] = reordered_sections
    row["section_titles"] = section_titles
    row["section_entity_urls"] = entity_urls
    row["section_entity_titles"] = entity_titles
    row["section_entity_ranks"] = entity_ranks
    row["section_scores"] = final_scores
    row["section_scores_section_component"] = section_components
    row["section_entity_probabilities"] = identification_values
    if normalized_field and normalized_scores is not None:
        row[normalized_field] = normalized_scores

    retrieval_meta["section_titles"] = section_titles
    retrieval_meta["section_entity_urls"] = entity_urls
    retrieval_meta["section_entity_titles"] = entity_titles
    retrieval_meta["section_entity_ranks"] = entity_ranks
    retrieval_meta["section_retrieval_similarities"] = retrieval_values
    retrieval_meta["section_entity_probabilities"] = identification_values
    retrieval_meta["section_scores_fused"] = final_scores
    retrieval_meta["section_scores_source"] = section_score_field
    if normalized_field and normalized_scores is not None:
        retrieval_meta[normalized_field] = normalized_scores
        sources = retrieval_meta.setdefault("section_scores_sources", {})
        base_source = sources.get(score_field_used)
        backend = base_source.get("backend") if isinstance(base_source, dict) else None
        model = base_source.get("model") if isinstance(base_source, dict) else None
        sources[normalized_field] = {
            "backend": backend,
            "model": model,
            "normalized": True,
        }
    row["retrieval_meta"] = retrieval_meta

    row.setdefault("metadata_rerank_details", {})
    if isinstance(row["metadata_rerank_details"], dict):
        row["metadata_rerank_details"]["fusion_weights"] = {
            "section": section_weight,
            "retrieval": retrieval_weight,
            "identification": identification_weight,
            "mode": identification_mode,
        }
        row["metadata_rerank_details"]["fusion_scores"] = final_scores
        row["metadata_rerank_details"]["section_components"] = section_components
        row["metadata_rerank_details"]["retrieval_components"] = retrieval_values
        row["metadata_rerank_details"]["identification_components"] = identification_values

    return True, best_idx, final_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute metadata context sections using weighted fusion of reranker, retrieval, and identification scores.",
    )
    parser.add_argument("--metadata_path", type=str, required=True, help="Input metadata JSONL.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to write updated metadata JSONL.")
    parser.add_argument("--section_score_weight", type=float, default=1.0)
    parser.add_argument("--retrieval_similarity_weight", type=float, default=0.0)
    parser.add_argument("--identification_probability_weight", type=float, default=0.0)
    parser.add_argument("--identification_score_mode", choices=["multiply", "add"], default="multiply")
    parser.add_argument(
        "--section_score_field",
        type=str,
        default="section_scores_echosight_raw",
        help="Metadata field to use as the base section score (default: section_scores_echosight_raw).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    output_path = Path(args.output_path)

    rows = _load_jsonl(metadata_path)

    processed = 0
    failures = 0
    scores_snapshot: Optional[List[Optional[float]]] = None

    for row in rows:
        success, _, scores = rerank_row(
            row,
            section_weight=args.section_score_weight,
            retrieval_weight=args.retrieval_similarity_weight,
            identification_weight=args.identification_probability_weight,
            identification_mode=args.identification_score_mode,
            section_score_field=args.section_score_field,
        )
        if success:
            processed += 1
            scores_snapshot = scores
        else:
            failures += 1

    _write_jsonl(output_path, rows)

    print(
        f"Rerank complete | processed={processed} | skipped={failures} -> {output_path}",
    )
    if scores_snapshot:
        print(f"Sample final scores: {scores_snapshot[:5]}")


if __name__ == "__main__":
    main()
