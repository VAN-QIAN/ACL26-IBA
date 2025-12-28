"""Qwen-2.5-VL integration for EchoSight.

This module wraps the Hugging Face implementation of Qwen/Qwen2.5-VL-7B-Instruct
so that the model can be used for both entity identification (image disambiguation)
and answer generation inside the EchoSight workflow.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .prompt_templates import PromptParts, build_prompt_parts

try:
    from .answer_generator import _adjust_prompt_length, reconstruct_wiki_article
    from .retriever import WikipediaKnowledgeBaseEntry
except Exception:  # pragma: no cover - allow execution outside package context
    try:
        from answer_generator import _adjust_prompt_length, reconstruct_wiki_article  # type: ignore
        from retriever import WikipediaKnowledgeBaseEntry  # type: ignore
    except Exception:  # pragma: no cover - final fallback shims
        WikipediaKnowledgeBaseEntry = None  # type: ignore

        def _adjust_prompt_length(prompt: str, desired_token_length: int) -> str:
            if not isinstance(prompt, str):
                return prompt
            if desired_token_length <= 0:
                return ""
            return prompt[:desired_token_length]

        def reconstruct_wiki_article(entry) -> str:
            if entry is None:
                return ""
            title = getattr(entry, "title", "")
            section_titles = getattr(entry, "section_titles", []) or []
            section_texts = getattr(entry, "section_texts", []) or []
            article = "# Wiki Article: " + str(title)
            for sec_title, sec_text in zip(section_titles, section_texts):
                article += "\n\n## Section Title: " + str(sec_title) + "\n" + str(sec_text or "")
            return article

try:
    _RESAMPLING_LANCZOS = Image.Resampling.LANCZOS  # Pillow>=9.1
except AttributeError:  # pragma: no cover - Pillow<9.1 compatibility
    _RESAMPLING_LANCZOS = Image.LANCZOS

LOGGER = logging.getLogger(__name__)

__all__ = [
    "QwenVLModel",
    "IdentificationResult",
    "IdentificationScores",
    "CandidateScore",
    "AnswerResult",
]


@dataclass
class IdentificationResult:
    """Structured output for identification requests."""

    raw_response: str
    selected_option: Optional[str]
    selected_index: int
    matched_by: Optional[str] = None
    ranked_labels: List[str] = field(default_factory=list)
    ranked_indices: List[int] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.selected_option is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "selected_option": self.selected_option,
            "selected_index": self.selected_index,
            "matched_by": self.matched_by,
            "ranked_labels": list(self.ranked_labels),
            "ranked_indices": list(self.ranked_indices),
        }


@dataclass
class AnswerResult:
    """Structured output for answer generation requests."""

    raw_response: str
    answer: str

    def as_dict(self) -> Dict[str, str]:
        return {"raw_response": self.raw_response, "answer": self.answer}


@dataclass
class CandidateScore:
    """Probability assigned to a single candidate entity."""

    index: int
    title: str
    probability: float


@dataclass
class IdentificationScores:
    """Structured probabilities for the identification ranking."""

    candidates: List[CandidateScore]
    none_probability: float
    raw_response: str


class QwenVLModel:
    """Thin wrapper around Qwen2.5-VL tailored for EchoSight."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        max_context_tokens: int = 1024,  # 4096
        max_image_side: Optional[int] = 2048,
        max_image_area: Optional[int] = 4_194_304,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU for Qwen-VL.")
            device = "cpu"
        self.model_name = model_name
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.max_image_side = max_image_side
        self.max_image_area = max_image_area
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self._load_model()

    def _load_model(self) -> None:
        LOGGER.info("Loading Qwen2.5-VL model %s on %s", self.model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        torch_dtype = (
            torch.bfloat16
            # if torch.cuda.is_available() and self.device.startswith("cuda")
            # else torch.float32
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as img:
            width, height = img.size
            scale_factors: List[float] = []
            if self.max_image_side is not None:
                longest = max(width, height)
                if longest > self.max_image_side:
                    scale_factors.append(self.max_image_side / float(longest))
            if self.max_image_area is not None:
                area = width * height
                if area > self.max_image_area and area > 0:
                    scale_factors.append(math.sqrt(self.max_image_area / float(area)))
            if scale_factors:
                scale = min(scale_factors)
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                if (new_width, new_height) != (width, height):
                    LOGGER.debug(
                        "Resizing image for Qwen input | path=%s | original=%dx%d | resized=%dx%d | scale=%.4f",
                        image_path,
                        width,
                        height,
                        new_width,
                        new_height,
                        scale,
                    )
                    img = img.resize((new_width, new_height), _RESAMPLING_LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            image = img.copy()
        return image

    def _prepare_context(self, context: Optional[Union[str, Sequence[str]]]) -> Optional[str]:
        if context is None:
            return None
        if isinstance(context, str):
            text = context.strip()
        else:
            text = "\n\n".join(str(item).strip() for item in context if str(item).strip())
        if not text:
            return None
        try:
            return _adjust_prompt_length(text, self.max_context_tokens)
        except Exception:
            LOGGER.debug("Context trimming failed; using raw text.", exc_info=True)
            return text

    def _format_identification_prompt(
        self,
        candidate_titles: Sequence[str],
        question: Optional[str],
        instruction: Optional[str],
        top_k: int,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> Tuple[List[str], str]:
        labels = [chr(ord("A") + idx) for idx in range(len(candidate_titles))]
        option_lines: List[str] = []
        for idx, (label, title) in enumerate(zip(labels, candidate_titles)):
            line = f"{label}. {title}"
            if candidate_similarities and idx < len(candidate_similarities):
                similarity = candidate_similarities[idx]
                if similarity is not None:
                    try:
                        line += f" (image similarity: {float(similarity):.3f})"
                    except (TypeError, ValueError):
                        pass
            option_lines.append(line)
        base_instruction = (
            "You are an expert visual entity recognizer. Look at the image and here are some potentially relevant options."
        )
        if instruction:
            base_instruction += f"\n{instruction.strip()}"
        if top_k <= 1:
            guidance = "Reply with 'Answer: <label>' where <label> is one of the option letters. "
        else:
            guidance = (
                f"Reply with 'Answer: <label1>, <label2>, ...' listing the top {top_k} option letters "
                "from most to least likely based on the image. "
            )
        # question_text = question #or "Which option matches the subject shown best in the image?"
        options_block = "\n".join(option_lines)
        prompt = (
            f"{base_instruction}\n"
            f"Options:\n{options_block}\n"
            # f"Question: {question_text}\n"
            f"{guidance}"
        )
        print("Identification prompt:", prompt)
        return labels, prompt

    def _format_scoring_prompt(
        self,
        candidate_titles: Sequence[str],
        question: Optional[str],
        instruction: Optional[str],
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> str:
        candidate_lines: List[str] = []
        for idx, title in enumerate(candidate_titles):
            line = f"{idx + 1}. {title}"
            if candidate_similarities and idx < len(candidate_similarities):
                similarity = candidate_similarities[idx]
                if similarity is not None:
                    try:
                        line += f" (retrieval similarity: {float(similarity):.3f})"
                    except (TypeError, ValueError):
                        pass
            candidate_lines.append(line)
        base_instruction = (
            "You are an expert visual entity recognizer. Examine the image carefully and estimate how likely "
            "each candidate entity corresponds to the subject. The candidates are listed in descending order "
            "based on a previous ranking."
        )
        if instruction:
            base_instruction += f"\n{instruction.strip()}"
        guidance = (
            "Respond ONLY with a JSON object using the keys "
            "'candidate_probabilities' (an array of probabilities matching the order of the candidates) "
            "and 'none_probability' (the probability that none of the candidates match). "
            "All probabilities must be numeric (not percentages) and should sum to 1 within 0.01."
        )
        question_text = question or "Which candidate best matches the subject in this image?"
        options_block = "\n".join(candidate_lines)
        prompt = (
            f"{base_instruction}\n"
            f"Question: {question_text}\n"
            f"Candidates (ordered):\n{options_block}\n"
            f"{guidance}"
        )
        return prompt

    def _build_prompt_parts(
        self,
        *,
        question: str,
        context_text: Optional[str],
        require_reasoning: bool,
        dataset_name: Optional[str] = None,
    ) -> PromptParts:
        return build_prompt_parts(
            dataset_name=dataset_name,
            question=question,
            context_text=context_text,
            require_reasoning=require_reasoning,
        )

    def _compose_messages(
        self,
        *,
        prompt_parts: PromptParts,
        image: Optional[Image.Image],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if prompt_parts.system:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt_parts.system}],
                }
            )
        user_content: List[Dict[str, Any]] = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": prompt_parts.user})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _prepare_answer_chat(
        self,
        *,
        question: str,
        context_text: Optional[str],
        require_reasoning: bool,
        dataset_name: Optional[str],
        image: Optional[Image.Image],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Image.Image]]]:
        prompt_parts = self._build_prompt_parts(
            question=question,
            context_text=context_text,
            require_reasoning=require_reasoning,
            dataset_name=dataset_name,
        )
        messages = self._compose_messages(prompt_parts=prompt_parts, image=image)
        images_payload = [image] if image is not None else None
        return messages, images_payload

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("QwenVLModel is not initialized.")
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: Dict[str, Any] = {"text": [text_prompt], "return_tensors": "pt"}
        if images is not None:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        tensor_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        pad_token_id = self.processor.tokenizer.pad_token_id
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = self.model.generate(**tensor_inputs, **generation_kwargs)
        attention_mask = tensor_inputs.get("attention_mask")
        if attention_mask is not None:
            input_length = int(attention_mask[0].sum().item())
        else:
            input_length = tensor_inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        decoded = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded.strip()

    @staticmethod
    def _parse_identification(
        response: str,
        labels: Sequence[str],
        candidate_titles: Sequence[str],
        top_k: int,
    ) -> Tuple[Optional[str], int, Optional[str], List[int], List[str]]:
        ranked_labels = QwenVLModel._extract_ranked_labels(response, labels)[:top_k]
        ranked_indices: List[int] = [labels.index(label) for label in ranked_labels]
        selected_option: Optional[str] = None
        selected_index = -1
        matched_by: Optional[str] = None
        if ranked_indices:
            selected_index = ranked_indices[0]
            selected_option = candidate_titles[selected_index]
            matched_by = "ranked_labels"
        normalized = response.lower()
        if "not sure" in normalized or "none of the" in normalized or "unknown" in normalized:
            return None, -1, None, ranked_indices, ranked_labels
        label_match = re.search(r"answer\s*[:：]\s*([A-Z])", response, re.IGNORECASE)
        if label_match:
            label = label_match.group(1).upper()
            if label in labels:
                idx = labels.index(label)
                if selected_index < 0:
                    selected_index = idx
                    selected_option = candidate_titles[idx]
                    matched_by = "label"
                if label not in ranked_labels:
                    ranked_labels.insert(0, label)
                    ranked_indices.insert(0, idx)
                return selected_option, selected_index, matched_by, ranked_indices, ranked_labels
        for idx, title in enumerate(candidate_titles):
            if title.lower() in normalized:
                if selected_index < 0:
                    selected_index = idx
                    selected_option = title
                    matched_by = "text"
                if labels[idx] not in ranked_labels:
                    ranked_labels.insert(0, labels[idx])
                    ranked_indices.insert(0, idx)
                return selected_option, selected_index, matched_by, ranked_indices, ranked_labels
        if selected_index >= 0:
            return selected_option, selected_index, matched_by, ranked_indices, ranked_labels
        return None, -1, None, ranked_indices, ranked_labels

    @staticmethod
    def _coerce_probability(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            is_percent = cleaned.endswith("%")
            if is_percent:
                cleaned = cleaned[:-1].strip()
            cleaned = cleaned.replace(",", "")
            try:
                numeric = float(cleaned)
            except ValueError:
                return None
            return numeric / 100.0 if is_percent else numeric
        return None

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
        # Attempt to parse progressively in case of trailing text.
        for end in range(len(candidate), 0, -1):
            snippet = candidate[:end]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
        return None

    def score_candidates(
        self,
        image_path: str,
        candidate_titles: Sequence[str],
        question: Optional[str] = None,
        instruction: Optional[str] = None,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> IdentificationScores:
        if not candidate_titles:
            raise ValueError("candidate_titles must not be empty.")
        limit = len(candidate_titles)
        image = self._load_image(image_path)
        prompt = self._format_scoring_prompt(
            candidate_titles,
            question,
            instruction,
            candidate_similarities=candidate_similarities,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        raw = self._generate(messages, images=[image], max_new_tokens=max_new_tokens, temperature=temperature)
        parsed = self._extract_first_json_object(raw)
        if not parsed:
            raise ValueError("Unable to parse probability JSON from Qwen response.")
        candidate_probs_raw = parsed.get("candidate_probabilities")
        values: List[Optional[float]] = []
        if isinstance(candidate_probs_raw, list):
            for item in candidate_probs_raw:
                if isinstance(item, dict):
                    value = item.get("probability") or item.get("score")
                else:
                    value = item
                values.append(self._coerce_probability(value))
        if not values:
            candidates_block = parsed.get("candidates")
            if isinstance(candidates_block, list):
                for entry in candidates_block:
                    value = None
                    if isinstance(entry, dict):
                        value = (
                            entry.get("probability")
                            or entry.get("score")
                            or entry.get("likelihood")
                        )
                    values.append(self._coerce_probability(value))
        # Fallback in case nothing was parsed.
        if not values:
            raise ValueError("Probability list missing in Qwen response.")
        # Ensure we have as many values as candidates (pad with None).
        if len(values) < limit:
            values.extend([None] * (limit - len(values)))
        candidate_probs: List[float] = []
        for idx in range(limit):
            prob = values[idx] if idx < len(values) else None
            if prob is None:
                candidate_probs.append(0.0)
            else:
                candidate_probs.append(max(0.0, float(prob)))
        none_raw = (
            parsed.get("none_probability")
            if "none_probability" in parsed
            else parsed.get("none_of_the_above_probability", parsed.get("other_probability"))
        )
        none_prob = self._coerce_probability(none_raw)
        if none_prob is None:
            none_prob = max(0.0, 1.0 - sum(candidate_probs))
        total = sum(candidate_probs) + none_prob
        if total > 0:
            candidate_probs = [prob / total for prob in candidate_probs]
            none_prob = none_prob / total
        candidates = [
            CandidateScore(index=idx, title=title, probability=candidate_probs[idx])
            for idx, title in enumerate(candidate_titles)
        ]
        return IdentificationScores(
            candidates=candidates,
            none_probability=max(0.0, min(1.0, none_prob)),
            raw_response=raw,
        )

    @staticmethod
    def _extract_ranked_labels(response: str, labels: Sequence[str]) -> List[str]:
        ordered: List[str] = []
        def extend_from_text(text: str) -> None:
            raw = re.findall(r"[A-Z]", text.upper())
            for label in raw:
                if label in labels and label not in ordered:
                    ordered.append(label)

        match_blocks = re.findall(
            r"(?:answer|top\s*\d*|choices?|options?)\s*[:：]\s*([A-Z\s,>/\-]+)",
            response,
            re.IGNORECASE,
        )
        for block in match_blocks:
            extend_from_text(block)
        if not ordered:
            bullet_matches = re.findall(r"^[\s\-•*]*([A-Z])\s*[).:-]", response, re.IGNORECASE | re.MULTILINE)
            for label in bullet_matches:
                upper = label.upper()
                if upper in labels and upper not in ordered:
                    ordered.append(upper)
        if not ordered:
            extend_from_text(response)
        return ordered

    @staticmethod
    def _extract_answer(response: str) -> str:
        match = re.search(r"answer\s*[:：]\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()

    def identify(
        self,
        image_path: str,
        candidate_titles: Sequence[str],
        question: Optional[str] = None,
        instruction: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 1,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> IdentificationResult:
        if not candidate_titles:
            raise ValueError("candidate_titles must not be empty.")
        # print(f"image_path: {image_path}")
        image = self._load_image(image_path)
        top_k = max(1, min(top_k, len(candidate_titles)))
        labels, prompt = self._format_identification_prompt(
            candidate_titles,
            question,
            instruction,
            top_k=top_k,
            candidate_similarities=candidate_similarities,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        raw = self._generate(messages, images=[image], max_new_tokens=max_new_tokens, temperature=temperature)
        selection, idx, matched_by, ranked_indices, ranked_labels = self._parse_identification(
            raw,
            labels,
            candidate_titles,
            top_k=top_k,
        )
        if ranked_indices and idx < 0:
            idx = ranked_indices[0]
            selection = candidate_titles[idx]
            matched_by = matched_by or "ranked_labels"
        if idx < 0 and candidate_titles:
            idx = 0
            selection = candidate_titles[0]
        return IdentificationResult(
            raw_response=raw,
            selected_option=selection,
            selected_index=idx,
            matched_by=matched_by,
            ranked_labels=ranked_labels,
            ranked_indices=ranked_indices,
        )

    def answer_question(
        self,
        image_path: str,
        question: str,
        context: Optional[Union[str, Sequence[str]]] = None,
        require_reasoning: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        return_full: bool = False,
        dataset_name: Optional[str] = None,
    ) -> Union[str, AnswerResult]:
        image = self._load_image(image_path)
        context_text = self._prepare_context(context)
        messages, images_payload = self._prepare_answer_chat(
            question=question,
            context_text=context_text,
            require_reasoning=require_reasoning,
            dataset_name=dataset_name,
            image=image,
        )
        raw = self._generate(
            messages,
            images=images_payload,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer = self._extract_answer(raw)
        result = AnswerResult(raw_response=raw, answer=answer)
        return result if return_full else result.answer

    def answer_with_entry(
        self,
        image_path: str,
        question: str,
        entry: Optional[WikipediaKnowledgeBaseEntry] = None,
        entry_dict: Optional[Dict[str, Any]] = None,
        entry_section: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, AnswerResult]:
        context: Optional[str] = None
        if entry is not None:
            context = reconstruct_wiki_article(entry)
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
        elif entry_section is not None:
            context = entry_section
        return self.answer_question(
            image_path=image_path,
            question=question,
            context=context,
            **kwargs,
        )
