"""EchoSight-based section reranker integrated with the Qwen pipeline."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image


class EchoSightSectionReranker:
    """Section reranker that leverages the EchoSight BLIP-2 Q-Former model."""

    def __init__(
        self,
        checkpoint_path: Optional[str],
        device: str,
        batch_size: int = 64,
    ) -> None:
        self.logger = logging.getLogger("qwen_pipeline.echosight_reranker")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.batch_size = max(1, batch_size)
        self.model = None
        self.txt_processor = None
        self.image_transform = None

        requested_device = device or "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning(
                "CUDA requested (%s) but not available; falling back to CPU.", requested_device
            )
            requested_device = "cpu"
        try:
            self.device = torch.device(requested_device)
        except (TypeError, RuntimeError):
            self.logger.warning(
                "Invalid device %s; defaulting to CPU for EchoSight reranker.", requested_device
            )
            self.device = torch.device("cpu")
        self._load_model()

    @property
    def available(self) -> bool:
        return self.model is not None and self.txt_processor is not None and self.image_transform is not None

    def _load_model(self) -> None:
        if self.checkpoint_path is None:
            self.logger.info("No EchoSight checkpoint provided; section reranker disabled.")
            return
        if not self.checkpoint_path.exists():
            self.logger.warning(
                "EchoSight checkpoint %s not found; reranker disabled.",
                self.checkpoint_path,
            )
            return
        try:
            from lavis.models import load_model_and_preprocess
            from data_utils import targetpad_transform
        except ImportError as exc:  # pragma: no cover - dependency error surfaced at runtime
            self.logger.error("Failed to import EchoSight dependencies: %s", exc)
            return

        base_device = "cuda" if self.device.type == "cuda" else "cpu"
        try:
            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_reranker",
                model_type="pretrain",
                is_eval=True,
                device=base_device,
            )
        except Exception as exc:  # pragma: no cover - runtime error surface
            self.logger.error("Unable to load EchoSight reranker backbone: %s", exc)
            return

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as exc:  # pragma: no cover - runtime error surface
            self.logger.error("Unable to load EchoSight checkpoint %s: %s", self.checkpoint_path, exc)
            return

        missing_keys = model.load_state_dict(checkpoint, strict=False).missing_keys
        if missing_keys:
            self.logger.debug("EchoSight reranker missing keys: %s", missing_keys)

        model.to(self.device)
        if self.device.type == "cuda":
            model = model.half()
        model.eval()
        model.use_vanilla_qformer = True

        self.model = model
        self.txt_processor = txt_processors["eval"]
        self.image_transform = targetpad_transform(1.25, 224)
        self.logger.info(
            "Loaded EchoSight reranker from %s on %s (batch=%d)",
            self.checkpoint_path,
            self.device,
            self.batch_size,
        )

    def _prepare_image(self, image_path: Optional[str]) -> Optional[torch.Tensor]:
        if not image_path:
            return None
        path_obj = Path(image_path)
        if not path_obj.exists():
            self.logger.warning("Image path %s missing; skipping reranking.", image_path)
            return None
        if self.image_transform is None:
            return None
        with Image.open(path_obj) as img:
            rgb = img.convert("RGB")
            tensor = self.image_transform(rgb).unsqueeze(0)
        return tensor.to(self.device)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return contextlib.nullcontext()

    def _encode_sections(self, sections: Sequence[str]) -> Optional[torch.Tensor]:
        if self.model is None or self.txt_processor is None:
            return None
        processed = [self.txt_processor(section) for section in sections]
        embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            with self._autocast():
                for start in range(0, len(processed), self.batch_size):
                    batch = processed[start : start + self.batch_size]
                    features = self.model.extract_features(
                        {"text_input": batch},
                        mode="text",
                    )["text_embeds_proj"][:, 0, :]
                    embeddings.append(features)
        if not embeddings:
            return None
        return torch.cat(embeddings, dim=0).to(torch.float32)

    def _score_sections(
        self,
        question: str,
        sections: Sequence[str],
        image_path: Optional[str],
    ) -> List[float]:
        if not sections or not self.available:
            return []
        image_tensor = self._prepare_image(image_path)
        if image_tensor is None:
            return []
        assert self.model is not None  # for type checkers
        fusion_embs = None
        with torch.no_grad():
            with self._autocast():
                fusion = self.model.extract_features(
                    {"image": image_tensor, "text_input": question or ""},
                    mode="multimodal",
                )
                fusion_embs = fusion["multimodal_embeds"].to(torch.float32)
        section_embs = self._encode_sections(sections)
        if section_embs is None or fusion_embs is None:
            return []
        scores = torch.matmul(
            section_embs.unsqueeze(1).unsqueeze(1),
            fusion_embs.permute(0, 2, 1),
        ).squeeze()
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        if scores.ndim == 1:
            final_scores = scores
        else:
            final_scores = scores.max(dim=-1).values
        return final_scores.cpu().tolist()

    def rerank(
        self,
        question: str,
        sections: Sequence[str],
        image_path: Optional[str],
    ) -> Tuple[List[str], List[float]]:
        if not sections:
            return list(sections), []
        scores = self._score_sections(question, sections, image_path)
        if not scores:
            return list(sections), []
        order = sorted(range(len(sections)), key=lambda idx: scores[idx], reverse=True)
        reranked = [sections[idx] for idx in order]
        sorted_scores = [scores[idx] for idx in order]
        return reranked, sorted_scores

    def pick_best(
        self,
        question: str,
        sections: Sequence[str],
        image_path: Optional[str],
    ) -> Tuple[int, List[float]]:
        if not sections:
            return -1, []
        scores = self._score_sections(question, sections, image_path)
        if not scores:
            return 0, []
        best_index = int(torch.tensor(scores).argmax().item())
        return best_index, scores
