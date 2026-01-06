"""Structured records shared across the Qwen pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalRecord:
    data_id: str
    question: str
    image_path: str
    dataset_name: str
    candidate_urls: List[str]
    reranked_urls: List[str]
    reranked_sections: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentificationRecord:
    data_id: str
    selected_url: str
    selected_title: str
    selected_index: int
    raw_response: str
    fallback_reason: Optional[str] = None
    options: List[str] = field(default_factory=list)
    ranked_urls: List[str] = field(default_factory=list)
    ranked_titles: List[str] = field(default_factory=list)
    ranked_indices: List[int] = field(default_factory=list)
    ranked_labels: List[str] = field(default_factory=list)
    ranked_probabilities: List[Optional[float]] = field(default_factory=list)
    none_probability: Optional[float] = None
    matched_by: Optional[str] = None
    ground_truth_url: Optional[str] = None
    ground_truth_in_topk: Optional[bool] = None


@dataclass
class ContextRecord:
    data_id: str
    mode: str
    text: str
    source_url: str
    section_title: Optional[str] = None
    source_rank: Optional[int] = None
    section_score: Optional[float] = None
    source_probability: Optional[float] = None


@dataclass
class MetadataRecord:
    retrieval: RetrievalRecord
    identification: IdentificationRecord
    context: ContextRecord

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.retrieval.data_id,
            "question": self.retrieval.question,
            "image_path": self.retrieval.image_path,
            "dataset_name": self.retrieval.dataset_name,
            "candidate_urls": self.retrieval.candidate_urls,
            "reranked_urls": self.retrieval.reranked_urls,
            "qwen_reranked_sections": self.retrieval.reranked_sections,
            "retrieval_meta": self.retrieval.meta,
            "original_data_id": self.retrieval.meta.get("original_data_id"),
            "candidate_data_ids": self.retrieval.meta.get("candidate_data_ids"),
            "selected_url": self.identification.selected_url,
            "selected_title": self.identification.selected_title,
            "selected_index": self.identification.selected_index,
            "identification_raw": self.identification.raw_response,
            "fallback_reason": self.identification.fallback_reason,
            "identification_options": self.identification.options,
            "identification_ranked_urls": self.identification.ranked_urls,
            "identification_ranked_titles": self.identification.ranked_titles,
            "identification_ranked_indices": self.identification.ranked_indices,
            "identification_ranked_labels": self.identification.ranked_labels,
            "identification_ranked_probabilities": self.identification.ranked_probabilities,
            "identification_none_probability": self.identification.none_probability,
            "identification_matched_by": self.identification.matched_by,
            "ground_truth_url": self.identification.ground_truth_url,
            "ground_truth_entity_in_topk_identification": self.identification.ground_truth_in_topk,
            "context_mode": self.context.mode,
            "context_text": self.context.text,
            "context_source_url": self.context.source_url,
            "context_section_title": self.context.section_title,
            "context_source_rank": self.context.source_rank,
            "context_section_score": self.context.section_score,
            "context_source_probability": self.context.source_probability,
        }
