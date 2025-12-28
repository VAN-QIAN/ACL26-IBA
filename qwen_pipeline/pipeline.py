"""Core orchestration for the Qwen-2.5-VL VQA workflow."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from aligned_answer_generator_w_qwen_metadata import (
    AlignedAnswerGenerator,
    GeneratorResponse,
    build_aligned_generator,
)

from model.retriever import WikipediaKnowledgeBase, WikipediaKnowledgeBaseEntry
from model.answer_generator import reconstruct_wiki_article, reconstruct_wiki_sections
from utils import get_image as legacy_get_image
from utils.test_utils_modified import get_infoseek_image_path, load_infoseek_mappings

from .context import SectionReranker, extract_title_from_section
from .io import build_retrieval_record, iter_examples, load_metadata, load_retrieval_results, write_jsonl
from .types import ContextRecord, IdentificationRecord, MetadataRecord, RetrievalRecord

_QWEN_SPEC = importlib_util.spec_from_file_location(
    "model.qwen_vl",
    Path(__file__).resolve().parent.parent / "model" / "Qwen-vl.py",
)
if _QWEN_SPEC is None or _QWEN_SPEC.loader is None:
    raise ImportError("Unable to load model/Qwen-vl.py")
_qwen_module = importlib_util.module_from_spec(_QWEN_SPEC)
sys.modules[_QWEN_SPEC.name] = _qwen_module
_QWEN_SPEC.loader.exec_module(_qwen_module)
QwenVLModel = _qwen_module.QwenVLModel
AnswerResult = _qwen_module.AnswerResult


@dataclass
class PipelineConfig:
    test_file: str
    retrieval_results: str
    knowledge_base: str
    qwen_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_device: str = "cuda:0"
    identification_top_k: int = 5
    identification_select_top: int = 1
    identification_score_top_k: int = 0
    identification_temperature: float = 0.0
    identification_max_new_tokens: int = 256
    identification_include_similarity: bool = False
    context_mode: str = "article"
    section_reranker: Optional[str] = None
    # Deprecated: kept for backward compatibility. Non-positive values mean "use all sections".
    section_pool_size: int = 0
    use_reranked_sections_first: bool = True
    answer_temperature: float = 0.0
    answer_max_new_tokens: int = 128
    require_reasoning: bool = False
    answer_backend: str = "qwen"
    answer_backend_device: Optional[str] = None
    answer_backend_model_path: Optional[str] = None
    inat_mapping_path: Optional[str] = "$ROOT_PATH/EchoSight/images/val_id2name.json"
    answer_rerank_sections: bool = False
    evqa_landmark_root: Optional[str] = "$ROOT_PATH/EchoSight/E-VQA/landmark"
    log_file: Optional[str] = None
    log_level: str = "INFO"

class QwenVQAPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("qwen_pipeline")
        level = getattr(logging, config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        stream_attached = any(
            isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout
            for handler in self.logger.handlers
        )
        if not stream_attached:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(stream_handler)

        if config.log_file:
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_log = log_path.resolve()
            file_attached = any(
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename).resolve() == resolved_log
                for handler in self.logger.handlers
            )
            if not file_attached:
                file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                self.logger.addHandler(file_handler)

        if logging.getLogger().level > level:
            logging.getLogger().setLevel(level)

        self.logger.propagate = False
        self.logger.info(
            "Initializing Qwen pipeline | context_mode=%s | answer_rerank=%s | qwen_model=%s",
            config.context_mode,
            config.answer_rerank_sections,
            config.qwen_model_name,
        )
        if config.section_reranker:
            self.logger.info("Using section reranker %s", config.section_reranker)

        self.answer_backend = config.answer_backend.lower()
        self.aligned_generator: Optional[AlignedAnswerGenerator] = None
        if self.answer_backend == "qwen":
            self.logger.info("Using Qwen as the answer generator")
            self.qwen = QwenVLModel(
                model_name=config.qwen_model_name,
                device=config.qwen_device,
            )
        self.section_selector = SectionReranker(config.section_reranker, config.qwen_device)
        self.kb_by_url: Dict[str, WikipediaKnowledgeBaseEntry] = {}
        self._inat_id2name: Optional[Dict[str, str]] = None
        self._image_id_to_data_id: Optional[Dict[str, str]] = None
        self._data_id_to_entity_id: Optional[Dict[str, str]] = None
        if config.knowledge_base:
            kb = WikipediaKnowledgeBase(config.knowledge_base)
            kb.load_knowledge_base()
            self.kb_by_url = {entry.url: entry for entry in kb.knowledge_base}
        if self.answer_backend != "qwen":
            backend_device = config.answer_backend_device or config.qwen_device
            self.logger.info(
                "Initializing aligned answer generator '%s' (device=%s)",
                self.answer_backend,
                backend_device,
            )
            self.aligned_generator = build_aligned_generator(
                self.answer_backend,
                kb_by_url=self.kb_by_url,
                device=backend_device,
                model_path=config.answer_backend_model_path,
                max_new_tokens=config.answer_max_new_tokens,
                temperature=config.answer_temperature,
            )
        if config.inat_mapping_path and Path(config.inat_mapping_path).exists():
            with open(config.inat_mapping_path, "r") as f:
                self._inat_id2name = json.load(f)
        self._image_id_to_data_id, self._data_id_to_entity_id = load_infoseek_mappings()

    def _candidate_entries(self, urls: List[str]) -> Tuple[List[WikipediaKnowledgeBaseEntry], List[str], List[str]]:
        entries: List[WikipediaKnowledgeBaseEntry] = []
        titles: List[str] = []
        kept_urls: List[str] = []
        for url in urls:
            entry = self.kb_by_url.get(url)
            if entry is None:
                continue
            entries.append(entry)
            titles.append(entry.title)
            kept_urls.append(url)
        return entries, titles, kept_urls

    @staticmethod
    def _collect_candidate_similarities(
        record: RetrievalRecord,
        ordered_urls: Sequence[str],
    ) -> List[Optional[float]]:
        similarities = record.meta.get("retrieval_similarities") or []
        if not similarities:
            return [None] * len(ordered_urls)
        buckets: Dict[str, List[Optional[float]]] = {}
        source_urls = record.candidate_urls or []
        for url, similarity in zip(source_urls, similarities):
            buckets.setdefault(url, []).append(similarity)
        aligned: List[Optional[float]] = []
        for url in ordered_urls:
            slot = buckets.get(url)
            if slot:
                aligned.append(slot.pop(0))
            else:
                aligned.append(None)
        return aligned

    def _run_identification(self, record: RetrievalRecord) -> IdentificationRecord:
        candidate_pool = record.reranked_urls or record.candidate_urls
        candidate_pool = candidate_pool[: self.config.identification_top_k]
        # print("Candidate pool:", candidate_pool)
        entries, titles, kept_urls = self._candidate_entries(candidate_pool)
        if not entries:
            self.logger.error("No matching knowledge base entries found for %s", record.data_id)
            raise ValueError(f"No matching knowledge base entries found for {record.data_id}.")
        rank_top = max(1, min(self.config.identification_select_top, len(entries)))
        # print("Running Qwen identification... with entries:", titles)
        candidate_similarities: Optional[List[Optional[float]]] = None
        if self.config.identification_include_similarity:
            aligned_sims = self._collect_candidate_similarities(record, kept_urls)
            if any(sim is not None for sim in aligned_sims):
                candidate_similarities = aligned_sims
        qwen_result = self.qwen.identify(
            image_path=record.image_path,
            candidate_titles=titles,
            question=record.question,
            max_new_tokens=self.config.identification_max_new_tokens,
            temperature=self.config.identification_temperature,
            top_k=rank_top,
            candidate_similarities=candidate_similarities,
        )
        fallback_reason = None
        ranked_pairs = [
            (label, idx)
            for label, idx in zip(qwen_result.ranked_labels, qwen_result.ranked_indices)
            if 0 <= idx < len(entries)
        ]
        ranked_pairs = ranked_pairs[:rank_top]
        if not ranked_pairs and 0 <= qwen_result.selected_index < len(entries):
            fallback_label = chr(ord("A") + qwen_result.selected_index)
            ranked_pairs = [(fallback_label, qwen_result.selected_index)]
        ranked_labels = [label for label, _ in ranked_pairs]
        ranked_indices = [idx for _, idx in ranked_pairs]
        ranked_urls = [kept_urls[idx] for idx in ranked_indices]
        ranked_titles = [entries[idx].title for idx in ranked_indices]
        if 0 <= qwen_result.selected_index < len(entries):
            chosen_index = qwen_result.selected_index
            selected_entry = entries[chosen_index]
            selected_url = kept_urls[chosen_index]
        else:
            chosen_index = 0
            selected_entry = entries[chosen_index]
            selected_url = kept_urls[chosen_index]
            fallback_reason = "qwen_no_selection"
            if not ranked_indices:
                ranked_indices = [0]
                ranked_urls = [selected_url]
                ranked_titles = [selected_entry.title]
                ranked_labels = ranked_labels or ["A"]
        return IdentificationRecord(
            data_id=record.data_id,
            selected_url=selected_url,
            selected_title=selected_entry.title,
            selected_index=chosen_index,
            raw_response=qwen_result.raw_response,
            fallback_reason=fallback_reason,
            options=titles,
            ranked_urls=ranked_urls,
            ranked_titles=ranked_titles,
            ranked_indices=ranked_indices,
            ranked_labels=ranked_labels,
            matched_by=qwen_result.matched_by,
        )

    def _sections_for_entry(self, entry: WikipediaKnowledgeBaseEntry, record: RetrievalRecord) -> List[str]:
        if self.config.use_reranked_sections_first:
            filtered = []
            for section in record.reranked_sections:
                title = extract_title_from_section(section)
                if title and title.lower() == entry.title.lower():
                    filtered.append(section)
            if filtered:
                return filtered
        sections = reconstruct_wiki_sections(entry)
        return sections

    @staticmethod
    def _truncate(text: Optional[str], limit: int = 160) -> Optional[str]:
        if text is None:
            return None
        trimmed = text.strip()
        if len(trimmed) <= limit:
            return trimmed
        return trimmed[: max(0, limit - 3)].rstrip() + "..."

    def _resolve_image_path(self, dataset_name: str, image_id: str) -> Optional[str]:
        try:
            if dataset_name == "inaturalist":
                if not self._inat_id2name:
                    self.logger.warning("Inaturalist mapping unavailable; cannot locate image")
                    return None
                path = legacy_get_image(image_id, dataset_name, self._inat_id2name)
            else:
                path = legacy_get_image(image_id, dataset_name)
        except (FileNotFoundError, ValueError, KeyError):
            path = None
        if path:
            return path
        if (
            dataset_name == "infoseek"
            and self._image_id_to_data_id is not None
            and self._data_id_to_entity_id is not None
        ):
            return get_infoseek_image_path(
                image_id,
                self._image_id_to_data_id,
                self._data_id_to_entity_id,
            )
        if dataset_name in {"landmarks", "landmark"}:
            evqa_path = self._resolve_evqa_landmark_image(image_id)
            if evqa_path:
                return evqa_path
        return None

    def _resolve_evqa_landmark_image(self, image_id: str) -> Optional[str]:
        if not image_id:
            return None
        root = self.config.evqa_landmark_root
        if not root:
            return None
        base = Path(root)
        if not base.exists():
            return None
        parts = [ch for ch in image_id[:3] if ch]
        search_dirs: List[Path] = []
        if parts:
            search_dirs.append(base.joinpath(*parts))
        search_dirs.append(base)
        unique_dirs: List[Path] = []
        seen_dirs = set()
        for directory in search_dirs:
            key = str(directory)
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            unique_dirs.append(directory)
        for directory in unique_dirs:
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = directory / f"{image_id}{ext}"
                if candidate.exists():
                    return str(candidate)
        return None

    def _candidate_data_ids(self, index: int, example: Dict[str, str]) -> List[str]:
        dataset_name = example.get("dataset_name", "")
        image_id = str(example.get("dataset_image_ids", "")).strip()
        candidates: List[str] = []
        if example.get("data_id"):
            candidates.append(str(example["data_id"]))
        candidates.extend(
            [
                f"E-VQA_{index:08d}",
                f"E-VQA_{index}",
                f"{dataset_name}_{image_id}".rstrip("_"),
                image_id,
            ]
        )
        seen = set()
        return [cid for cid in candidates if cid and not (cid in seen or seen.add(cid))]

    def _build_context(self, record: RetrievalRecord, identification: IdentificationRecord) -> ContextRecord:
        entry = self.kb_by_url[identification.selected_url]
        source_probability: Optional[float] = None
        if identification.ranked_urls and identification.ranked_probabilities:
            lookup = {
                url: prob
                for url, prob in zip(identification.ranked_urls, identification.ranked_probabilities)
                if prob is not None
            }
            source_probability = lookup.get(identification.selected_url)
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
                    source_probability=source_probability,
                )
            best_index, _ = self.section_selector.pick_best(record.question, pool)
            if best_index < 0 or best_index >= len(pool):
                best_index = 0
            chosen = pool[best_index]
            return ContextRecord(
                data_id=record.data_id,
                mode="section",
                text=chosen,
                source_url=identification.selected_url,
                section_title=extract_title_from_section(chosen),
                source_probability=source_probability,
            )
        article = reconstruct_wiki_article(entry)
        return ContextRecord(
            data_id=record.data_id,
            mode="article",
            text=article,
            source_url=identification.selected_url,
            source_probability=source_probability,
        )

    def _log_prepare_step(
        self,
        record: RetrievalRecord,
        identification: IdentificationRecord,
        context: ContextRecord,
    ) -> None:
        options_preview: Optional[str] = None
        if identification.options:
            preview = [str(option) for option in identification.options[:3]]
            options_preview = ', '.join(preview)
            if len(identification.options) > 3:
                options_preview += ', ...'
        context_ref = context.section_title or context.source_url
        self.logger.info(
            'Prepare %s | question="%s" | selected_entity="%s" (idx=%s) | context_mode=%s | context_ref="%s"',
            record.data_id,
            self._truncate(record.question, 120),
            identification.selected_title,
            identification.selected_index,
            context.mode,
            self._truncate(context_ref, 120),
        )
        if identification.fallback_reason:
            self.logger.warning(
                'Fallback for %s: %s',
                record.data_id,
                identification.fallback_reason,
            )
        if options_preview:
            self.logger.debug(
                'Options %s: %s',
                record.data_id,
                options_preview,
            )
        self.logger.debug(
            'Context snippet %s: %s',
            record.data_id,
            self._truncate(context.text, 200),
        )

    def _log_answer_step(
        self,
        data_id: str,
        question: str,
        selected_entity: Optional[str],
        context_mode: Optional[str],
        context_ref: Optional[str],
        context_text: Optional[str],
        answer: str,
        raw_response: str,
        section_source: Optional[str] = None,
        section_index: Optional[int] = None,
        section_scores: Optional[List[float]] = None,
    ) -> None:
        self.logger.info(
            'Answer %s | question="%s" | entity="%s" | context_mode=%s | context_ref="%s" | answer="%s"',
            data_id,
            self._truncate(question, 120),
            selected_entity,
            context_mode,
            self._truncate(context_ref, 120),
            self._truncate(answer, 180),
        )
        if section_index is not None:
            self.logger.debug(
                'Section info %s: source=%s index=%s',
                data_id,
                section_source,
                section_index,
            )
        if section_scores:
            head_scores = ', '.join(f'{score:.3f}' for score in section_scores[:5])
            self.logger.debug(
                'Section scores %s: %s',
                data_id,
                head_scores,
            )
        self.logger.debug(
            'Raw response %s: %s',
            data_id,
            self._truncate(raw_response, 200),
        )
        if context_text:
            self.logger.debug(
                'Context used %s: %s',
                data_id,
                self._truncate(context_text, 200),
            )

    def prepare_metadata(self, metadata_path: str) -> List[MetadataRecord]:
        if not self.kb_by_url:
            raise ValueError('Knowledge base must be loaded to prepare metadata.')
        retrieval_blob = load_retrieval_results(self.config.retrieval_results)
        records: List[MetadataRecord] = []
        for idx, example in iter_examples(self.config.test_file):
            candidate_ids = self._candidate_data_ids(idx, example)
            matched_id = next((cid for cid in candidate_ids if cid in retrieval_blob), None)
            if matched_id is None:
                self.logger.warning(
                    "Skipping example %s: no retrieval results for candidates %s",
                    example.get('data_id', candidate_ids[0] if candidate_ids else idx),
                    candidate_ids,
                )
                continue
            image_path = self._resolve_image_path(
                dataset_name=example.get('dataset_name', ''),
                image_id=str(example.get('dataset_image_ids', '')),
            )
            if image_path is None:
                self.logger.warning('Skipping %s: image not found', matched_id)
                continue
            self.logger.debug(
                'Matched data_id %s from candidates %s',
                matched_id,
                candidate_ids,
            )
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
                self.logger.warning('Skipping %s due to missing knowledge base entries', matched_id)
                continue
            ground_truth_url = (
                example.get('wikipedia_url')
                or example.get('ground_truth_url')
                or example.get('gold_wikipedia_url')
            )
            if ground_truth_url:
                ranked_urls = identification.ranked_urls or []
                in_topk = ground_truth_url in ranked_urls or ground_truth_url == identification.selected_url
                identification.ground_truth_url = ground_truth_url
                identification.ground_truth_entity_in_topk = in_topk
                retrieval_record.meta.setdefault('ground_truth_url', ground_truth_url)
                retrieval_record.meta['ground_truth_entity_in_topk_identification'] = in_topk
            retrieval_record.meta[
                'identification_prompt_includes_similarity'
            ] = self.config.identification_include_similarity
            entry = self.kb_by_url.get(identification.selected_url)
            if entry is not None:
                filtered_sections = self._sections_for_entry(entry, retrieval_record)
                retrieval_record.reranked_sections = filtered_sections
            else:
                retrieval_record.reranked_sections = []
            context = self._build_context(retrieval_record, identification)
            self._log_prepare_step(retrieval_record, identification, context)
            records.append(MetadataRecord(retrieval_record, identification, context))
        write_jsonl(metadata_path, (record.to_dict() for record in records))
        self.logger.info('Prepared metadata for %d examples -> %s', len(records), metadata_path)
        return records

    def _answer_single(
        self,
        question: str,
        context_text: Optional[str],
        image_path: Optional[str],
        use_image: bool,
        metadata_row: Optional[Dict[str, object]] = None,
    ) -> AnswerResult:
        dataset_name: Optional[str] = None
        if metadata_row is not None:
            raw_dataset = metadata_row.get("dataset_name")
            if isinstance(raw_dataset, str) and raw_dataset.strip():
                dataset_name = raw_dataset.strip()
        if dataset_name is None and metadata_row is not None:
            retrieval_meta = metadata_row.get("retrieval_meta")
            if isinstance(retrieval_meta, dict):
                raw_dataset = retrieval_meta.get("dataset_name")
                if isinstance(raw_dataset, str) and raw_dataset.strip():
                    dataset_name = raw_dataset.strip()

        if self.answer_backend != "qwen":
            if self.aligned_generator is None:
                raise ValueError("Aligned answer generator is not initialized")
            effective_row: Dict[str, object]
            if metadata_row is None:
                effective_row = {
                    "question": question,
                    "context_text": context_text,
                    "context_mode": None,
                    "reranked_sections": [],
                    "image_path": image_path,
                }
            else:
                effective_row = dict(metadata_row)
                effective_row.setdefault("question", question)
                if context_text is not None:
                    effective_row["context_text"] = context_text
                if context_text is None and "context_text" not in effective_row:
                    effective_row["context_text"] = None
                effective_row.setdefault("context_mode", None)
                if image_path is not None:
                    effective_row["image_path"] = image_path
            if dataset_name is not None:
                effective_row.setdefault("dataset_name", dataset_name)
            generation_kwargs = {
                "max_new_tokens": self.config.answer_max_new_tokens,
                "temperature": self.config.answer_temperature,
            }
            response: GeneratorResponse = self.aligned_generator.answer_metadata(
                effective_row,
                generation_kwargs=generation_kwargs,
            )
            return AnswerResult(raw_response=response.raw_response, answer=response.answer)

        if use_image and image_path:
            return self.qwen.answer_question(
                image_path=image_path,
                question=question,
                context=context_text,
                require_reasoning=self.config.require_reasoning,
                max_new_tokens=self.config.answer_max_new_tokens,
                temperature=self.config.answer_temperature,
                return_full=True,
                dataset_name=dataset_name,
            )
        prepared_context = self.qwen._prepare_context(context_text)  # type: ignore[attr-defined]
        messages, images_payload = self.qwen._prepare_answer_chat(
            question=question,
            context_text=prepared_context,
            require_reasoning=self.config.require_reasoning,
            dataset_name=dataset_name,
            image=None,
        )
        raw = self.qwen._generate(
            messages=messages,
            images=images_payload,
            max_new_tokens=self.config.answer_max_new_tokens,
            temperature=self.config.answer_temperature,
        )
        answer = self.qwen._extract_answer(raw)
        return AnswerResult(raw_response=raw, answer=answer)

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
                    and self.section_selector.model is not None
                ):
                    best_index, scores = self.section_selector.pick_best(question or "", sections)
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

    def export_metadata(self, metadata_path: str, records: Iterable[MetadataRecord]) -> None:
        write_jsonl(metadata_path, (record.to_dict() for record in records))
