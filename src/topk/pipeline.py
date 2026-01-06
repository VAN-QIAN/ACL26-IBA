"""Top-K aware variant of the Qwen VQA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from model.answer_generator import reconstruct_wiki_article, reconstruct_wiki_sections
from model.retriever import WikipediaKnowledgeBaseEntry

from ..context import SectionReranker, extract_title_from_section
from ..context_echosight import EchoSightSectionReranker
from ..io import build_retrieval_record, iter_examples, load_metadata, load_retrieval_results, write_jsonl
from ..pipeline import PipelineConfig, QwenVQAPipeline
from ..types import ContextRecord, IdentificationRecord, MetadataRecord, RetrievalRecord


@dataclass
class TopKPipelineConfig(PipelineConfig):
    """Configuration for the entity top-k selection pipeline."""

    identification_select_top: int = 3
    identification_score_top_k: int = 3
    entity_top_k: int = 3
    section_reranker_backend: str = "auto"  # values: auto, bge, echosight
    echosight_checkpoint: Optional[str] = None
    echosight_batch_size: int = 64
    echosight_retrieval_alpha: float = 0.0
    section_score_weight: float = 1.0
    retrieval_similarity_weight: float = 0.0
    identification_probability_weight: float = 0.0
    identification_score_mode: str = "multiply"  # values: multiply, add
    section_score_source: str = "blended"  # values: blended, raw


@dataclass
class _EntityCandidate:
    url: str
    title: str
    entry: WikipediaKnowledgeBaseEntry
    rank: Optional[int]
    retrieval_similarity: Optional[float] = None
    probability: Optional[float] = None


@dataclass
class _SectionCandidate:
    text: str
    entity_url: str
    entity_title: str
    entity_rank: Optional[int]
    section_title: Optional[str]
    echosight_score: Optional[float] = None
    retrieval_similarity: Optional[float] = None
    score: Optional[float] = None
    entity_probability: Optional[float] = None
    blended_score: Optional[float] = None


class TopKQwenPipeline(QwenVQAPipeline):
    """Pipeline that reranks sections across multiple identified entities."""

    config: TopKPipelineConfig

    def __init__(self, config: TopKPipelineConfig) -> None:
        super().__init__(config)
        backend = (config.section_reranker_backend or "auto").lower()
        if backend not in {"auto", "bge", "echosight"}:
            self.logger.warning(
                "Unknown section_reranker_backend '%s'; falling back to 'auto'.",
                backend,
            )
            backend = "auto"
        if backend == "auto":
            backend = "echosight" if config.echosight_checkpoint else "bge"
        self.section_backend = backend
        if backend == "echosight":
            checkpoint = config.echosight_checkpoint or config.section_reranker
            self.section_selector = EchoSightSectionReranker(
                checkpoint_path=checkpoint,
                device=config.qwen_device,
                batch_size=config.echosight_batch_size,
            )
            if not self.section_selector.available:
                self.logger.warning(
                    "EchoSight reranker unavailable; reverting to BGE reranker backend.",
                )
                self.section_backend = "bge"
                self.section_selector = SectionReranker(config.section_reranker, config.qwen_device)
            elif config.echosight_retrieval_alpha > 0:
                self.logger.info(
                    "EchoSight retrieval blending enabled (alpha=%.3f)",
                    config.echosight_retrieval_alpha,
                )
        else:
            self.section_selector = SectionReranker(config.section_reranker, config.qwen_device)
            if config.echosight_retrieval_alpha > 0:
                self.logger.warning(
                    "echosight_retrieval_alpha=%.3f ignored because backend=%s",
                    config.echosight_retrieval_alpha,
                    self.section_backend,
                )
        mode = str(getattr(self.config, "identification_score_mode", "multiply")).lower()
        if mode not in {"multiply", "add"}:
            self.logger.warning(
                "Unknown identification_score_mode '%s'; defaulting to 'multiply'.",
                mode,
            )
            mode = "multiply"
        self.config.identification_score_mode = mode
        score_source = str(getattr(self.config, "section_score_source", "blended")).lower()
        if score_source not in {"blended", "raw"}:
            self.logger.warning(
                "Unknown section_score_source '%s'; defaulting to 'blended'.",
                score_source,
            )
            score_source = "blended"
        self.config.section_score_source = score_source

        self.logger.info(
            "TopK pipeline ready | entities_top=%d | reranker_backend=%s",
            config.entity_top_k,
            self.section_backend,
        )
        self._last_section_components = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieval_similarity_lookup(self, retrieval: RetrievalRecord) -> Dict[str, float]:
        similarities = retrieval.meta.get("retrieval_similarities") or []
        lookup: Dict[str, float] = {}
        for url, score in zip(retrieval.candidate_urls, similarities):
            if url in lookup:
                continue
            try:
                lookup[url] = float(score)
            except (TypeError, ValueError):
                continue
        return lookup

    def _entity_candidates(
        self,
        retrieval: RetrievalRecord,
        identification: IdentificationRecord,
        retrieval_similarity: Dict[str, float],
    ) -> List[_EntityCandidate]:
        desired = max(1, self.config.entity_top_k)
        ranked_urls = identification.ranked_urls or []
        rank_lookup = {url: idx for idx, url in enumerate(ranked_urls)}
        probability_lookup: Dict[str, float] = {}
        ranked_probabilities = identification.ranked_probabilities or []
        for url, prob in zip(ranked_urls, ranked_probabilities):
            if prob is None:
                continue
            try:
                probability_lookup[url] = float(prob)
            except (TypeError, ValueError):
                continue
        ordered_urls: List[str] = []
        for url in ranked_urls:
            if url not in ordered_urls:
                ordered_urls.append(url)
        if identification.selected_url and identification.selected_url not in ordered_urls:
            ordered_urls.append(identification.selected_url)
        for url in retrieval.reranked_urls:
            if len(ordered_urls) >= desired:
                break
            if url not in ordered_urls:
                ordered_urls.append(url)
        candidates: List[_EntityCandidate] = []
        for url in ordered_urls[:desired]:
            entry = self.kb_by_url.get(url)
            if entry is None:
                continue
            candidates.append(
                _EntityCandidate(
                    url=url,
                    title=entry.title,
                    entry=entry,
                    rank=rank_lookup.get(url),
                    retrieval_similarity=retrieval_similarity.get(url),
                    probability=probability_lookup.get(url),
                )
            )
        return candidates

    def _collect_sections(self, entities: Sequence[_EntityCandidate]) -> List[_SectionCandidate]:
        pool: List[_SectionCandidate] = []
        for entity in entities:
            sections = reconstruct_wiki_sections(entity.entry)
            if not sections:
                article = reconstruct_wiki_article(entity.entry)
                pool.append(
                    _SectionCandidate(
                        text=article,
                        entity_url=entity.url,
                        entity_title=entity.title,
                        entity_rank=entity.rank,
                        section_title=None,
                        retrieval_similarity=entity.retrieval_similarity,
                        entity_probability=entity.probability,
                    )
                )
                continue
            for section in sections:
                pool.append(
                    _SectionCandidate(
                        text=section,
                        entity_url=entity.url,
                        entity_title=entity.title,
                        entity_rank=entity.rank,
                        section_title=extract_title_from_section(section),
                        retrieval_similarity=entity.retrieval_similarity,
                        entity_probability=entity.probability,
                    )
                )
        return pool

    def _order_sections(
        self,
        question: str,
        sections: Sequence[_SectionCandidate],
        image_path: Optional[str],
    ) -> Tuple[List[_SectionCandidate], List[Optional[float]], List[Optional[float]]]:
        if not sections:
            return [], [], []
        texts = [candidate.text for candidate in sections]
        if self.section_backend == "echosight":
            best_idx, raw_scores = self.section_selector.pick_best(question, texts, image_path)
        else:
            best_idx, raw_scores = self.section_selector.pick_best(question, texts)

        scores: List[float] = list(raw_scores) if raw_scores else []
        blended_scores = self._blend_echosight_scores(sections, scores) if scores else []
        score_source = getattr(self.config, "section_score_source", "blended")

        section_weight = float(getattr(self.config, "section_score_weight", 1.0))
        retrieval_weight = float(getattr(self.config, "retrieval_similarity_weight", 0.0))
        identification_weight = float(getattr(self.config, "identification_probability_weight", 0.0))
        identification_mode = str(getattr(self.config, "identification_score_mode", "multiply")).lower()

        base_scores: List[Optional[float]] = []
        section_components: List[Optional[float]] = []
        retrieval_components: List[Optional[float]] = []
        identification_components: List[Optional[float]] = []
        final_scores: List[Optional[float]] = []

        for idx, candidate in enumerate(sections):
            raw_score = scores[idx] if scores and idx < len(scores) else None
            blended_score = blended_scores[idx] if blended_scores and idx < len(blended_scores) else None
            if score_source == "raw" and raw_score is not None:
                base_score = raw_score
            else:
                base_score = blended_score
            candidate.blended_score = base_score
            candidate.echosight_score = raw_score
            identification_value = None
            if candidate.entity_probability is not None:
                try:
                    identification_value = float(candidate.entity_probability)
                except (TypeError, ValueError):
                    identification_value = None
            retrieval_value = None
            if candidate.retrieval_similarity is not None:
                try:
                    retrieval_value = float(candidate.retrieval_similarity)
                except (TypeError, ValueError):
                    retrieval_value = None

            section_component = base_score
            if identification_mode == "multiply" and section_component is not None and identification_value is not None:
                section_component = section_component * identification_value

            weighted_sum = 0.0
            weight_applied = False
            final_score: Optional[float] = None

            if section_component is not None and section_weight != 0.0:
                weighted_sum += section_weight * section_component
                weight_applied = True
            if retrieval_value is not None and retrieval_weight != 0.0:
                weighted_sum += retrieval_weight * retrieval_value
                weight_applied = True
            if identification_value is not None and identification_weight != 0.0:
                weighted_sum += identification_weight * identification_value
                weight_applied = True

            if weight_applied:
                final_score = weighted_sum
            else:
                final_score = section_component

            base_scores.append(base_score)
            section_components.append(section_component)
            retrieval_components.append(retrieval_value)
            identification_components.append(identification_value)
            final_scores.append(final_score)

        def _score_value(values: Sequence[Optional[float]], idx: int) -> float:
            if not values or idx >= len(values):
                return float("-inf")
            value = values[idx]
            return float(value) if value is not None else float("-inf")

        if any(value is not None for value in final_scores):
            order = sorted(range(len(texts)), key=lambda idx: _score_value(final_scores, idx), reverse=True)
        elif any(value is not None for value in base_scores):
            order = sorted(range(len(texts)), key=lambda idx: _score_value(base_scores, idx), reverse=True)
        elif best_idx >= 0:
            order = [best_idx] + [idx for idx in range(len(texts)) if idx != best_idx]
        else:
            order = list(range(len(texts)))
        ordered: List[_SectionCandidate] = []
        ordered_scores: List[Optional[float]] = []
        ordered_raw_scores: List[Optional[float]] = []
        ordered_section_components: List[Optional[float]] = []
        ordered_retrieval_components: List[Optional[float]] = []
        ordered_identification_components: List[Optional[float]] = []
        for idx in order:
            final_score = final_scores[idx] if idx < len(final_scores) else None
            base_score = base_scores[idx] if idx < len(base_scores) else None
            raw_score = scores[idx] if scores and idx < len(scores) else None
            candidate = sections[idx]
            ordered.append(
                _SectionCandidate(
                    text=candidate.text,
                    entity_url=candidate.entity_url,
                    entity_title=candidate.entity_title,
                    entity_rank=candidate.entity_rank,
                    section_title=candidate.section_title,
                    echosight_score=raw_score,
                    retrieval_similarity=candidate.retrieval_similarity,
                    score=final_score,
                    entity_probability=candidate.entity_probability,
                    blended_score=base_score,
                )
            )
            ordered_scores.append(final_score)
            ordered_raw_scores.append(raw_score)
            ordered_section_components.append(section_components[idx])
            ordered_retrieval_components.append(retrieval_components[idx])
            ordered_identification_components.append(identification_components[idx])

        # Attach component traces to the original sequence order for metadata.
        self._last_section_components = {
            "section": ordered_section_components,
            "retrieval": ordered_retrieval_components,
            "identification": ordered_identification_components,
            "weights": {
                "section": section_weight,
                "retrieval": retrieval_weight,
                "identification": identification_weight,
                "mode": identification_mode,
            },
        }

        return ordered, ordered_scores, ordered_raw_scores

    def _pick_best_sections(
        self,
        question: str,
        sections: Sequence[str],
        image_path: Optional[str],
    ) -> Tuple[int, List[float]]:
        if self.section_backend == "echosight":
            return self.section_selector.pick_best(question, list(sections), image_path)
        return self.section_selector.pick_best(question, list(sections))

    def _blend_echosight_scores(
        self,
        sections: Sequence[_SectionCandidate],
        raw_scores: List[float],
    ) -> List[float]:
        if not raw_scores:
            return raw_scores
        if self.section_backend != "echosight":
            return raw_scores
        alpha = max(0.0, min(1.0, float(getattr(self.config, "echosight_retrieval_alpha", 0.0))))
        if alpha <= 0.0:
            return raw_scores
        retrieval_weight = alpha
        reranker_weight = 1.0 - alpha
        blended: List[float] = []
        for idx, score in enumerate(raw_scores):
            candidate = sections[idx]
            retrieval_component = 0.0
            if candidate.retrieval_similarity is not None:
                try:
                    retrieval_component = float(candidate.retrieval_similarity)
                except (TypeError, ValueError):
                    retrieval_component = 0.0
            blended.append(retrieval_weight * retrieval_component + reranker_weight * score)
        return blended

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

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
            similarity_lookup = self._retrieval_similarity_lookup(retrieval_record)
            score_top_k = max(0, int(getattr(self.config, "identification_score_top_k", 0)))
            if score_top_k > 0:
                top_titles = identification.ranked_titles[:score_top_k] or []
                if top_titles:
                    try:
                        candidate_similarities = [
                            similarity_lookup.get(url)
                            for url in identification.ranked_urls[:score_top_k]
                        ]
                        scores = self.qwen.score_candidates(
                            image_path=retrieval_record.image_path,
                            candidate_titles=top_titles,
                            question=retrieval_record.question,
                            candidate_similarities=candidate_similarities,
                            max_new_tokens=self.config.identification_max_new_tokens,
                            temperature=self.config.identification_temperature,
                        )
                    except Exception as exc:
                        self.logger.warning(
                            "Failed to score identification candidates for %s: %s",
                            matched_id,
                            exc,
                        )
                    else:
                        probability_map = {score.title: score.probability for score in scores.candidates}
                        identification.ranked_probabilities = [
                            probability_map.get(title) if title in probability_map else None
                            for title in identification.ranked_titles
                        ]
                        identification.none_probability = scores.none_probability
                        retrieval_record.meta["identification_scores"] = [
                            {
                                "title": candidate.title,
                                "probability": candidate.probability,
                                "ordered_rank": idx,
                            }
                            for idx, candidate in enumerate(scores.candidates)
                        ]
                        retrieval_record.meta["identification_none_probability"] = scores.none_probability
                        retrieval_record.meta["identification_scores_raw_response"] = scores.raw_response
            ground_truth_url = (
                example.get("wikipedia_url")
                or example.get("ground_truth_url")
                or example.get("gold_wikipedia_url")
            )
            if ground_truth_url:
                ranked_urls = identification.ranked_urls or []
                in_topk = ground_truth_url in ranked_urls or ground_truth_url == identification.selected_url
                identification.ground_truth_url = ground_truth_url
                identification.ground_truth_in_topk = in_topk
                retrieval_record.meta.setdefault("ground_truth_url", ground_truth_url)
                retrieval_record.meta["ground_truth_entity_in_topk_identification"] = in_topk
            retrieval_record.meta[
                "identification_prompt_includes_similarity"
            ] = self.config.identification_include_similarity
            entities = self._entity_candidates(
                retrieval_record,
                identification,
                similarity_lookup,
            )
            if not entities:
                self.logger.warning("No entity candidates survived for %s", matched_id)
                continue
            sections = self._collect_sections(entities)
            if not sections:
                self.logger.warning("No sections available after expansion for %s", matched_id)
                continue
            ordered_sections, ordered_scores, ordered_raw_scores = self._order_sections(
                retrieval_record.question,
                sections,
                retrieval_record.image_path,
            )
            if not ordered_sections:
                ordered_sections = sections
                ordered_scores = [candidate.score for candidate in sections]
                ordered_raw_scores = [candidate.echosight_score for candidate in sections]
            best_section = ordered_sections[0]
            mode = "section" if best_section.section_title else "article"
            retrieval_record.reranked_sections = [candidate.text for candidate in ordered_sections]
            retrieval_record.meta["section_entity_urls"] = [candidate.entity_url for candidate in ordered_sections]
            retrieval_record.meta["section_entity_titles"] = [candidate.entity_title for candidate in ordered_sections]
            retrieval_record.meta["section_entity_ranks"] = [candidate.entity_rank for candidate in ordered_sections]
            retrieval_record.meta["section_entity_probabilities"] = [
                candidate.entity_probability for candidate in ordered_sections
            ]
            retrieval_record.meta["section_titles"] = [candidate.section_title for candidate in ordered_sections]
            retrieval_record.meta["section_scores"] = ordered_scores
            retrieval_record.meta["section_scores_unweighted"] = [
                candidate.blended_score for candidate in ordered_sections
            ]
            components_payload = getattr(self, "_last_section_components", None)
            if isinstance(components_payload, dict):
                retrieval_record.meta["section_scores_components"] = {
                    "section": components_payload.get("section"),
                    "retrieval": components_payload.get("retrieval"),
                    "identification": components_payload.get("identification"),
                    "weights": components_payload.get("weights"),
                }
                retrieval_record.meta["section_scores_weighting"] = components_payload.get("weights")
            if ordered_raw_scores:
                retrieval_record.meta["section_scores_echosight_raw"] = ordered_raw_scores
            retrieval_record.meta["section_scores_source"] = getattr(self.config, "section_score_source", "blended")
            if self.section_backend == "echosight":
                retrieval_record.meta["section_scores_mode"] = (
                    "echosight_blended"
                    if getattr(self.config, "echosight_retrieval_alpha", 0.0) > 0.0
                    else "echosight"
                )
                retrieval_record.meta["section_retrieval_similarities"] = [
                    candidate.retrieval_similarity for candidate in ordered_sections
                ]
                retrieval_record.meta["section_blend_alpha"] = getattr(
                    self.config, "echosight_retrieval_alpha", 0.0
                )
            retrieval_record.meta["section_reranker_backend"] = self.section_backend
            context = ContextRecord(
                data_id=retrieval_record.data_id,
                mode=mode,
                text=best_section.text,
                source_url=best_section.entity_url,
                section_title=best_section.section_title,
                source_rank=best_section.entity_rank,
                section_score=best_section.score,
                source_probability=best_section.entity_probability,
            )
            self._last_section_components = None
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
        outputs: List[Dict[str, Optional[str]]] = []
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
                    and getattr(self.section_selector, "model", None) is not None
                ):
                    best_index, scores = self._pick_best_sections(
                        question or "",
                        sections,
                        row.get("image_path") if use_image else None,
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


__all__ = ["TopKPipelineConfig", "TopKQwenPipeline"]
