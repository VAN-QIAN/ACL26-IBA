"""EchoSight-specific variant of the Qwen VQA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from model.answer_generator import reconstruct_wiki_article, reconstruct_wiki_sections

from .context import extract_title_from_section
from .context_echosight import EchoSightSectionReranker
from .io import build_retrieval_record, iter_examples, load_metadata, load_retrieval_results, write_jsonl
from .pipeline import PipelineConfig, QwenVQAPipeline
from .types import ContextRecord, IdentificationRecord, MetadataRecord, RetrievalRecord


@dataclass
class EchoSightPipelineConfig(PipelineConfig):
    """Configuration for EchoSight-enabled pipeline runs."""

    echosight_checkpoint: Optional[str] = None
    echosight_batch_size: int = 64


class EchoSightVQAPipeline(QwenVQAPipeline):
    """Pipeline that reranks sections with the EchoSight reranker after entity selection."""

    config: EchoSightPipelineConfig

    def __init__(self, config: EchoSightPipelineConfig) -> None:
        super().__init__(config)
        checkpoint = config.echosight_checkpoint or config.section_reranker
        self.section_selector = EchoSightSectionReranker(
            checkpoint_path=checkpoint,
            device=config.qwen_device,
            batch_size=config.echosight_batch_size,
        )
        if not self.section_selector.available:
            self.logger.warning(
                "EchoSight reranker unavailable; falling back to baseline section selection."
            )

    def _selected_entry_sections(
        self,
        record: RetrievalRecord,
        identification: IdentificationRecord,
    ) -> Tuple[List[str], List[float]]:
        entry = self.kb_by_url.get(identification.selected_url)
        if entry is None:
            return record.reranked_sections, []
        sections = reconstruct_wiki_sections(entry)
        if not sections:
            return record.reranked_sections, []
        candidates = sections
        reranked, scores = self.section_selector.rerank(
            question=record.question,
            sections=candidates,
            image_path=record.image_path,
        )
        if not reranked:
            return list(candidates), scores
        return reranked, scores

    def _build_context(
        self,
        record: RetrievalRecord,
        identification: IdentificationRecord,
    ) -> ContextRecord:
        entry = self.kb_by_url[identification.selected_url]
        if self.config.context_mode == "section":
            sections = self._sections_for_entry(entry, record)
            pool = sections
            if not pool:
                article = reconstruct_wiki_article(entry)
                return ContextRecord(
                    data_id=record.data_id,
                    mode="article",
                    text=article,
                    source_url=identification.selected_url,
                )
            best_index, _ = self.section_selector.pick_best(
                record.question,
                pool,
                record.image_path,
            )
            if best_index < 0 or best_index >= len(pool):
                best_index = 0
            chosen = pool[best_index]
            return ContextRecord(
                data_id=record.data_id,
                mode="section",
                text=chosen,
                source_url=identification.selected_url,
                section_title=extract_title_from_section(chosen),
            )
        article = reconstruct_wiki_article(entry)
        return ContextRecord(
            data_id=record.data_id,
            mode="article",
            text=article,
            source_url=identification.selected_url,
        )

    def prepare_metadata(self, metadata_path: str) -> List[MetadataRecord]:
        if not self.kb_by_url:
            raise ValueError("Knowledge base must be loaded to prepare metadata.")
        retrieval_blob = load_retrieval_results(self.config.retrieval_results)
        records: List[MetadataRecord] = []
        for idx, example in iter_examples(self.config.test_file):
            candidate_ids = self._candidate_data_ids(idx, example)
            matched_id = next((cid for cid in candidate_ids if cid in retrieval_blob), None)
            if matched_id is None:
                self.logger.warning(
                    "Skipping example %s: no retrieval results for candidates %s",
                    example.get("data_id", candidate_ids[0] if candidate_ids else idx),
                    candidate_ids,
                )
                continue
            image_path = self._resolve_image_path(
                dataset_name=example.get("dataset_name", ""),
                image_id=str(example.get("dataset_image_ids", "")),
            )
            if image_path is None:
                self.logger.warning("Skipping %s: image not found", matched_id)
                continue
            retrieval_record = build_retrieval_record(
                example,
                matched_id,
                retrieval_blob[matched_id],
                image_path=image_path,
                candidate_ids=candidate_ids,
            )
            try:
                identification = self._run_identification(retrieval_record)
            except ValueError:
                self.logger.warning("Skipping %s due to missing knowledge base entries", matched_id)
                continue
            if identification.fallback_reason:
                self.logger.warning(
                    "Skipping %s: identification fallback (%s)",
                    matched_id,
                    identification.fallback_reason,
                )
                continue
            reranked_sections, _ = self._selected_entry_sections(retrieval_record, identification)
            retrieval_record.reranked_sections = reranked_sections
            context = self._build_context(retrieval_record, identification)
            self._log_prepare_step(retrieval_record, identification, context)
            records.append(MetadataRecord(retrieval_record, identification, context))
        write_jsonl(metadata_path, (record.to_dict() for record in records))
        self.logger.info("Prepared metadata for %d examples -> %s", len(records), metadata_path)
        return records

    def answer_from_metadata(
        self,
        metadata_path: str,
        output_path: str,
        use_image: bool = True,
        dataset_name: Optional[str] = None,
    ) -> None:
        rows = load_metadata(metadata_path)
        outputs: List[dict] = []
        for row in rows:
            question = row["question"]
            context_text = row.get("context_text")
            effective_row: Dict[str, object] = dict(row)
            effective_row["question"] = question
            if dataset_name is not None:
                effective_row["dataset_name"] = dataset_name
            if context_text is not None:
                effective_row["context_text"] = context_text
            section_title = row.get("context_section_title")
            section_index: Optional[int] = None
            section_scores: Optional[List[float]] = None
            section_source = "metadata"
            if row.get("context_mode") == "section":
                sections: List[str] = row.get("reranked_sections", []) or []
                if (
                    self.config.answer_rerank_sections
                    and sections
                    and self.section_selector.available
                ):
                    best_index, scores = self.section_selector.pick_best(
                        question or "",
                        sections,
                        row.get("image_path"),
                    )
                    if 0 <= best_index < len(sections):
                        context_text = sections[best_index]
                        section_title = extract_title_from_section(context_text)
                        section_index = best_index
                        section_scores = scores
                        section_source = "answer_reranker"
                else:
                    if sections and context_text in sections:
                        section_index = sections.index(context_text)
                if context_text is not None:
                    effective_row["context_text"] = context_text
                if section_index is not None:
                    effective_row["selected_section_index"] = section_index
                if section_title is not None:
                    effective_row["context_section_title"] = section_title
            result = self._answer_single(
                question=question,
                context_text=context_text,
                image_path=row.get("image_path") if use_image else None,
                use_image=use_image,
                metadata_row=effective_row,
            )
            output_row = {
                "data_id": row["data_id"],
                "prediction": result.answer,
                "raw_response": result.raw_response,
                "context_mode": row.get("context_mode"),
                "selected_url": row.get("selected_url"),
                "selected_title": row.get("selected_title"),
                "use_image": use_image,
            }
            if row.get("context_mode") == "section":
                output_row.update(
                    {
                        "selected_section_text": context_text,
                        "selected_section_title": section_title,
                        "selected_section_index": section_index,
                        "selected_section_source": section_source,
                    }
                )
                if section_scores is not None:
                    output_row["section_scores"] = section_scores
            context_ref = section_title or row.get("context_source_url") or row.get("selected_url")
            self._log_answer_step(
                data_id=row["data_id"],
                question=question,
                selected_entity=row.get("selected_title"),
                context_mode=row.get("context_mode"),
                context_ref=context_ref,
                context_text=context_text,
                answer=result.answer,
                raw_response=result.raw_response,
                section_source=section_source if row.get("context_mode") == "section" else None,
                section_index=section_index,
                section_scores=section_scores,
            )
            outputs.append(output_row)
        write_jsonl(output_path, outputs)
        self.logger.info("Generated answers for %d examples -> %s", len(outputs), output_path)
