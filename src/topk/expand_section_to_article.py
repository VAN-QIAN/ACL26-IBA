"""Utilities for expanding section-level metadata into full articles."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from model.answer_generator import reconstruct_wiki_article
from model.retriever import WikipediaKnowledgeBaseEntry

from .io import load_metadata, write_jsonl

LOGGER = logging.getLogger(__name__)


def _normalise_url(url: str) -> str:
    """Normalise Wikipedia URLs so different spellings map to the same entry."""

    return url.strip().rstrip("/")


def _build_kb_index(knowledge_base_path: str) -> Dict[str, WikipediaKnowledgeBaseEntry]:
    """Load the knowledge base JSON and index entries by canonical URL."""

    with open(knowledge_base_path, "r") as handle:
        raw_entries = json.load(handle)

    index: Dict[str, WikipediaKnowledgeBaseEntry] = {}
    for raw_key, payload in raw_entries.items():
        entry = WikipediaKnowledgeBaseEntry(payload)
        candidates = {
            raw_key,
            payload.get("url"),
            payload.get("wikipedia_url"),
            entry.url,
        }
        if entry.title:
            wiki_url = f"https://en.wikipedia.org/wiki/{entry.title.replace(' ', '_')}"
            candidates.add(wiki_url)
        for candidate in candidates:
            if not candidate:
                continue
            index.setdefault(_normalise_url(candidate), entry)
    LOGGER.info("Indexed %d knowledge base entries from %s", len(index), knowledge_base_path)
    return index


def expand_section_to_article(
    metadata_path: str,
    knowledge_base_path: str,
    output_path: Optional[str] = None,
    preserve_section_text: bool = True,
) -> Tuple[int, int]:
    """Expand section-level contexts in metadata to full articles.

    Args:
        metadata_path: Path to the existing metadata JSONL file.
        knowledge_base_path: Path to the knowledge base JSON used during preparation.
        output_path: Destination for the expanded metadata JSONL. Defaults to
            ``<metadata_path>.expanded.jsonl`` when omitted.
        preserve_section_text: Whether to keep the original section snippet under
            ``context_section_text`` for downstream inspection.

    Returns:
        A tuple ``(expanded, skipped)`` indicating how many rows were expanded and
        how many section rows were skipped due to a missing knowledge base entry.
    """

    kb_index = _build_kb_index(knowledge_base_path)
    rows = load_metadata(metadata_path)
    expanded = 0
    skipped = 0
    updated_rows: List[Dict] = []

    for row in rows:
        mode = row.get("context_mode")
        url = row.get("context_source_url") or row.get("selected_url")
        new_row = dict(row)
        if mode == "section" and url:
            entry = kb_index.get(_normalise_url(url))
            if entry is None:
                LOGGER.warning("No knowledge base entry found for %s", url)
                skipped += 1
            else:
                if preserve_section_text and row.get("context_text"):
                    new_row.setdefault("context_section_text", row["context_text"])
                new_row["context_text"] = reconstruct_wiki_article(entry)
                new_row["context_expanded_from"] = "section"
                expanded += 1
        updated_rows.append(new_row)

    destination = output_path or f"{metadata_path}.expanded.jsonl"
    write_jsonl(destination, updated_rows)
    LOGGER.info(
        "Expanded %d contexts to articles (%d missing) -> %s",
        expanded,
        skipped,
        destination,
    )
    return expanded, skipped


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand section contexts to full articles")
    parser.add_argument("--metadata", required=True, help="Input metadata JSONL file")
    parser.add_argument("--knowledge_base", required=True, help="Knowledge base JSON path")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the expanded metadata (defaults to <metadata>.expanded.jsonl)",
    )
    parser.add_argument(
        "--drop-section-text",
        action="store_true",
        help="Do not preserve the original section text under context_section_text",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> Tuple[int, int]:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    return expand_section_to_article(
        metadata_path=args.metadata,
        knowledge_base_path=args.knowledge_base,
        output_path=args.output,
        preserve_section_text=not args.drop_section_text,
    )


if __name__ == "__main__":
    main()
