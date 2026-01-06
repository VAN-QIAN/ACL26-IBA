"""Context utilities for selecting evidence sections."""

import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


_SECTION_TITLE_RE = re.compile(r"# Wiki Article: (?P<title>.+?)\\n")


def extract_title_from_section(section_text: str) -> Optional[str]:
    match = _SECTION_TITLE_RE.search(section_text)
    if not match:
        return None
    return match.group("title").strip()


class SectionReranker:
    def __init__(self, model_name: Optional[str], device: str) -> None:
        self.device = device
        self.model_name = model_name
        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
        else:
            self.tokenizer = None
            self.model = None

    def pick_best(self, question: str, sections: List[str]) -> Tuple[int, List[float]]:
        if not sections:
            return -1, []
        if self.model is None or self.tokenizer is None:
            return 0, []
        pairs = [[question, section] for section in sections]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1).float()
        best_idx = int(torch.argmax(scores).item())
        return best_idx, scores.cpu().tolist()
