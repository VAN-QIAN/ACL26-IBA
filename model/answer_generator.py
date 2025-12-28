""" Serve for question generation for a list of given knowledge base entries. 
"""

from typing import List, Optional

from openai import OpenAI
import torch
from .retriever import WikipediaKnowledgeBaseEntry
from .prompt_templates import build_prompt_parts
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.generation import GenerationConfig

import time


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def _build_dataset_prompt_messages(
    *,
    question: str,
    context_text: Optional[str],
    dataset_name: Optional[str],
    require_reasoning: bool = False,
) -> List[dict]:
    prompt_parts = build_prompt_parts(
        dataset_name=dataset_name,
        question=question,
        context_text=context_text,
        require_reasoning=require_reasoning,
    )
    messages: List[dict] = []
    if prompt_parts.system:
        messages.append({"role": "system", "content": prompt_parts.system})
    messages.append({"role": "user", "content": prompt_parts.user})
    return messages


def reconstruct_wiki_article(knowledge_entry: WikipediaKnowledgeBaseEntry):
    """Reconstruct the wiki article from the knowledge entry class."""
    title = knowledge_entry.title
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if (
            "external link" in section_title.lower()
            or "reference" in section_title.lower()
        ):
            continue
        article += (
            "\n## Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return article


def reconstruct_wiki_sections(knowledge_entry, section_index=-1):
    """Reconstruct the wiki sections from the knowledge entry class."""
    title = knowledge_entry.title
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if it == int(section_index):
            evidence_section = (
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry.section_texts[it]
            )
        elif (
            "external links" in section_title.lower()
            or "references" in section_title.lower()
        ):
            continue
        else:
            sections.append(
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry.section_texts[it]
            )
    if section_index != -1:
        return evidence_section, sections
    return sections


def get_all_sections(knowledge_entry):
    """Get all sections in list format."""
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        sections.append(
            "* Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return sections


pseudo_tokenizer = None


# def _adjust_prompt_length(prompt, desired_token_length):
#     """Adjust the prompt length to the desired token length."""
#     global pseudo_tokenizer

#     if pseudo_tokenizer is None:
#         pseudo_tokenizer = AutoTokenizer.from_pretrained(
#             "qwenlm/qwen-7b", use_fast=False
#         )

#     # Tokenize the prompt
#     tokens = pseudo_tokenizer.encode(prompt)

#     if len(tokens) > desired_token_length:
#         # If the prompt is too long, trim it
#         trimmed_tokens = tokens[:desired_token_length]
#         # Convert tokens back to text
#         trimmed_text = pseudo_tokenizer.decode(
#             trimmed_tokens, clean_up_tokenization_spaces=True
#         )[4:]
#         return trimmed_text
#     else:
#         # If the length is already as desired
#         return prompt

def _adjust_prompt_length(prompt, desired_token_length):
    """Adjust the prompt length to the desired token length (plain text only)."""
    global pseudo_tokenizer

    if pseudo_tokenizer is None:
        # 用 Qwen-2.5 的官方分词器；纯文本统计/截断，不用模板
        pseudo_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", use_fast=True
        )

    # 仅对“原始输入”编码统计：不添加任何特殊 token
    token_ids = pseudo_tokenizer.encode(
        prompt, add_special_tokens=False
    )

    if len(token_ids) > desired_token_length:
        # 右侧截断（保留开头）；如需保留末尾，把 [:desired_token_length] 改成 [-desired_token_length:]
        trimmed_ids = token_ids[:desired_token_length]
        # 解码回文本；既然我们没加特殊 token，就不要跳过它们，也不要做额外清理
        trimmed_text = pseudo_tokenizer.decode(
            trimmed_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return trimmed_text
    else:
        return prompt

class AnswerGenerator:
    """Question generator for EchoSight."""

    def __init__(self):
        self.model = None

    def load_model(self, model_name):
        """Load the model.

        Args:
            model_name: The model to load.
        """
        raise NotImplementedError


class MistralAnswerGenerator(AnswerGenerator):
    """Mistral Question generator for EchoSight."""

    def __init__(self, device, model_path, use_embedding_model=False):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
            use_embedding_model: Whether to use the embedding model. Deprecated in the current version.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()
        if use_embedding_model:
            self._load_embedding()
        else:
            self.emb = None

    def _load_model(self):
        """Load the model."""
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).eval()
        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        question,
        entry=None,
        entry_dict=None,
        entry_section=None,
        oracle_setting="subject",
        evidence_sec=None,
        dataset_name: Optional[str] = None,
    ):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
            oracle_setting: The setting for the oracle experiment.
            evidence_sec: The evidence section.
        """
        context_text: Optional[str] = None
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context_text = _adjust_prompt_length(context, 4096)
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            context_text = _adjust_prompt_length(context, 4096)
        elif entry_section is not None:
            context_text = entry_section
        elif evidence_sec is not None:
            context_text = evidence_sec

        messages = _build_dataset_prompt_messages(
            question=question,
            context_text=context_text,
            dataset_name=dataset_name,
            require_reasoning=False,
        )

        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", max_length=8000, truncation=True
        )
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            pad_token_id=2,
        )
        response = (self.tokenizer.decode(generated_ids[0][model_inputs.shape[1] :]))[
            :-4
        ]

        return response


class LLaMA3AnswerGenerator(AnswerGenerator):
    def __init__(self, device, model_path):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the model.

        Args:
            model_path: The model to load.
        """
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") # self.model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16
            #self.model_path, torch_dtype=torch.bfloat16
        ).eval()
        self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        question,
        entry=None,
        entry_dict=None,
        entry_section=None,
        oracle_setting="subject",
        evidence_sec=None,
        dataset_name: Optional[str] = None,
    ):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
            oracle_setting: The setting for the oracle experiment.
            evidence_sec: The evidence section.
        """
        dataset = (dataset_name or "").strip().lower()
        context_text: Optional[str] = None
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context_text = _adjust_prompt_length(context, 4096)
        elif entry_dict is not None:
            entry_obj = WikipediaKnowledgeBaseEntry(entry_dict)
            if dataset == "infoseek":
                context = reconstruct_wiki_article(entry_obj)
                context_text = _adjust_prompt_length(context, 4096)
            else:
                context_text = entry_obj.title
        elif entry_section is not None:
            context_text = entry_section
        elif evidence_sec is not None:
            context_text = evidence_sec

        messages = _build_dataset_prompt_messages(
            question=question,
            context_text=context_text,
            dataset_name=dataset_name,
            require_reasoning=False,
        )
        if dataset != "infoseek" and (not messages or messages[0].get("role") != "system"):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a helpful assistant for answering encyclopedic questions.",
                },
            )
        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        # ]
        # encodeds = self.tokenizer.apply_chat_template(
        #     messages, return_tensors="pt", max_length=8000, truncation=True
        # )
        # model_inputs = encodeds.to(self.device)
        # generated_ids = self.model.generate(
        #     model_inputs,
        #     max_new_tokens=128,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # # response = self.tokenizer.decode(generated_ids[0][model_inputs.shape[1] :])
        # response = self.tokenizer.decode(generated_ids[0][model_inputs.shape[1]:], skip_special_tokens=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,   # 关键！
            truncation=True,
            max_length=8000
        ).to(self.device)

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [self.tokenizer.eos_token_id] if eot_id is None else [self.tokenizer.eos_token_id, eot_id]

        out_ids = self.model.generate(
            enc,
            max_new_tokens=128,
            do_sample=True, temperature=0.6, top_p=0.9,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out_ids[0, enc.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"Generated response: {response}")

        return response


class GPT4AnswerGenerator(AnswerGenerator):
    """OpenAI Question generator for EchoSight."""

    def __init__(self):
        """Initialize the QuestionGenerator class."""
        super().__init__()
        self.client = OpenAI(api_key="YOUR_API_KEY")

    def get_gpt4_answer(self, input):
        """Get the answer from the GPT-4 model.

        Args:
            input: The input to the model.
        """
        MAX_RETRIES = 5
        retries = 0

        while retries < MAX_RETRIES:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input},
                    ],
                )

                assistant_reply = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                retries += 1
                time.sleep(2)
        return assistant_reply

    def llm_answering(
        self,
        question,
        entry=None,
        entry_dict=None,
        entry_section=None,
        dataset_name: Optional[str] = None,
    ):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
        """
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context = _adjust_prompt_length(context, 4096)

            prompt = (
                "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            )
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            context = _adjust_prompt_length(context, 4096)
            prompt = (
                "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            )
        elif entry_section is not None:
            prompt = (
                "Context: "
                + entry_section
                + "\nQuestion: "
                + question
                + "\nThe answer is:"
            )
        else:
            prompt = "Question: " + question + "\nThe answer is:"
        response = self.get_gpt4_answer(prompt)
        return response


class PaLMAnswerGenerator(AnswerGenerator):
    """Google PaLM Question generator for EchoSight."""

    def __init__(self):
        """Initialize the QuestionGenerator class."""
        super().__init__()
        import vertexai
        from vertexai.preview.language_models import (
            ChatModel,
            InputOutputTextPair,
            TextEmbeddingModel,
            TextGenerationModel,
        )
        import os

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_CREDENTIALS.json"
        PROJECT_ID = "YOUR_PROJECT_ID"
        REGION = "YOUR_REGION"
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = TextGenerationModel.from_pretrained("text-bison@002")

    def llm_answering(
        self,
        question,
        entry=None,
        entry_dict=None,
        entry_section=None,
        dataset_name: Optional[str] = None,
    ):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
        """
        if entry is not None:
            context = reconstruct_wiki_article(entry)
            context = _adjust_prompt_length(context, 4096)

            prompt = (
                "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            )
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
            context = _adjust_prompt_length(context, 4096)
            prompt = (
                "Context: " + context + "\nQuestion: " + question + "\nThe answer is:"
            )
        elif entry_section is not None:
            prompt = (
                "Context: "
                + entry_section
                + "\nQuestion: "
                + question
                + "\nThe answer is:"
            )
        else:
            prompt = "Question: " + question + "\nThe answer is:"

        response = self.model.predict(
            prompt,
            temperature=0.2,
            max_output_tokens=128,
            top_k=40,
            top_p=0.95,
        ).text
        return response


class BgeTextReranker:
    """Text Reranker for EchoSight.

    Deprecated in the current version.
    """

    def __init__(self, model_path, device):
        """Initialize the Text Reranker"""
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank_entry_sections(self, question, sections, top_k=3, gt_index=-1):
        if gt_index == -1:
            return -1, 0
        pairs = [[question, section] for section in sections[:top_k]]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=6000
        ).to(self.device)
        scores = (
            self.model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        scores, index = torch.sort(scores, descending=True)

        return index[0], int(index[0]) == int(gt_index)
