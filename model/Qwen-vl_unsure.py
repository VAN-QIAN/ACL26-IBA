"""Qwen-2.5-VL integration for EchoSight.

This module wraps the Hugging Face implementation of Qwen/Qwen2.5-VL-7B-Instruct
so that the model can be used for both entity identification (image disambiguation)
and answer generation inside the EchoSight workflow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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

LOGGER = logging.getLogger(__name__)

__all__ = ["QwenVLModel", "IdentificationResult", "AnswerResult"]


@dataclass
class IdentificationResult:
    """Structured output for identification requests."""

    raw_response: str
    selected_option: Optional[str]
    selected_index: int
    matched_by: Optional[str] = None

    @property
    def is_confident(self) -> bool:
        return self.selected_option is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "selected_option": self.selected_option,
            "selected_index": self.selected_index,
            "matched_by": self.matched_by,
        }


@dataclass
class AnswerResult:
    """Structured output for answer generation requests."""

    raw_response: str
    answer: str

    def as_dict(self) -> Dict[str, str]:
        return {"raw_response": self.raw_response, "answer": self.answer}


class QwenVLModel:
    """Thin wrapper around Qwen2.5-VL tailored for EchoSight."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        max_context_tokens: int = 1024, # 4096
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable. Falling back to CPU for Qwen-VL.")
            device = "cpu"
        self.model_name = model_name
        self.device = device
        self.max_context_tokens = max_context_tokens
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

    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
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
    ) -> Tuple[List[str], str]:
        labels = [chr(ord("A") + idx) for idx in range(len(candidate_titles))]
        option_lines = [f"{label}. {title}" for label, title in zip(labels, candidate_titles)]
        base_instruction = (
            "You are an expert visual entity recognizer. Look at the image and here are some potentially relevant options.Please also describe how you get the results step by step."
        )
        if instruction:
            base_instruction += f"\n{instruction.strip()}"
        guidance = (
            "Reply with 'Answer: <label>' where <label> is one of the option letters. "
            # "If none of the options fit, reply 'Answer: Not sure'. "
        )
        question_text = question or "Which option matches the subject shown best in the image?"
        options_block = "\n".join(option_lines)
        prompt = (
            f"{base_instruction}\n"
            f"Options:\n{options_block}\n"
            f"Question: {question_text}\n"
            f"{guidance}"
        )
        return labels, prompt

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
    ) -> Tuple[Optional[str], int, Optional[str]]:
        normalized = response.lower()
        if "not sure" in normalized or "none of the" in normalized or "unknown" in normalized:
            return None, -1, None
        label_match = re.search(r"answer\s*[:：]\s*([A-Z])", response, re.IGNORECASE)
        if label_match:
            label = label_match.group(1).upper()
            if label in labels:
                idx = labels.index(label)
                return candidate_titles[idx], idx, "label"
        for idx, title in enumerate(candidate_titles):
            if title.lower() in normalized:
                return title, idx, "text"
        return None, -1, None

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
    ) -> IdentificationResult:
        if not candidate_titles:
            raise ValueError("candidate_titles must not be empty.")
        # print(f"image_path: {image_path}")
        image = self._load_image(image_path)
        labels, prompt = self._format_identification_prompt(candidate_titles, question, instruction)
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
        selection, idx, matched_by = self._parse_identification(raw, labels, candidate_titles)
        return IdentificationResult(
            raw_response=raw,
            selected_option=selection,
            selected_index=idx,
            matched_by=matched_by,
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
