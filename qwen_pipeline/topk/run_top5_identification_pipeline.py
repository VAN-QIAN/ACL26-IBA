"""CLI entrypoint for running the Top-K Qwen identification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .pipeline import TopKPipelineConfig, TopKQwenPipeline


def _coalesce_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    string_value = value.strip()
    return string_value or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen entity identification with top-k support and flexible rerankers",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Perform retrieval alignment, ask Qwen for top-k entities, rerank sections, and export metadata.",
    )
    prepare.add_argument("--test_file", type=str, required=True, help="CSV with evaluation questions")
    prepare.add_argument("--retrieval_results", type=str, required=True, help="JSON retrieval blob keyed by data_id")
    prepare.add_argument("--knowledge_base", type=str, required=True, help="Path to serialized KB JSON")
    prepare.add_argument("--metadata_path", type=str, required=True, help="Destination JSONL for metadata output")
    prepare.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    prepare.add_argument("--qwen_device", type=str, default="cuda:0")
    prepare.add_argument("--identification_top_k", type=int, default=20, help="Number of retrieval candidates sent to Qwen")
    prepare.add_argument("--identification_select_top", type=int, default=5, help="Number of ranked options to request from Qwen")
    prepare.add_argument("--identification_temperature", type=float, default=0.0)
    prepare.add_argument("--identification_max_new_tokens", type=int, default=256)
    prepare.add_argument(
        "--identification_include_similarity",
        action="store_true",
        help="Include initial retrieval image similarities in the identification prompt.",
    )
    prepare.add_argument("--entity_top_k", type=int, default=5, help="How many entities to expand into sections")
    prepare.add_argument("--context_mode", choices=["article", "section"], default="section")
    prepare.add_argument(
        "--section_pool_size",
        type=int,
        default=0,
        help="Deprecated; kept for compatibility. Non-positive values use all sections.",
    )
    prepare.add_argument("--section_reranker_backend", choices=["auto", "bge", "echosight"], default="bge")
    prepare.add_argument("--section_reranker", type=str, default=None, help="HF reranker checkpoint when using 'bge'")
    prepare.add_argument("--echosight_checkpoint", type=str, default=None, help="EchoSight BLIP-2 checkpoint path")
    prepare.add_argument("--echosight_batch_size", type=int, default=500)
    prepare.add_argument(
        "--echosight_retrieval_alpha",
        type=float,
        default=0.5,
        help="Weight (0-1) for blending retrieval similarities with EchoSight scores (legacy behaviour)",
    )
    prepare.add_argument("--use_reranked_sections_first", action="store_true", help="Keep reranker ordering ahead of KB reconstruction")
    prepare.add_argument("--answer_backend", type=str, default="qwen")
    prepare.add_argument("--answer_backend_device", type=str, default=None)
    prepare.add_argument("--answer_backend_model_path", type=str, default=None)
    prepare.add_argument("--answer_temperature", type=float, default=0.0)
    prepare.add_argument("--answer_max_new_tokens", type=int, default=512)
    prepare.add_argument("--require_reasoning", action="store_true")
    prepare.add_argument("--answer_rerank_sections", action="store_true")
    prepare.add_argument("--inat_mapping_path", type=str, default="/data/qianMa/EchoSight/images/val_id2name.json")
    prepare.add_argument("--evqa_landmark_root", type=str, default="/data/qianMa/EchoSight/E-VQA/landmark")
    prepare.add_argument("--log_file", type=str, default=None)
    prepare.add_argument("--log_level", type=str, default="INFO")

    answer = subparsers.add_parser(
        "answer",
        help="Consume prepared metadata and run the answer generator over the selected sections",
    )
    answer.add_argument("--metadata_path", type=str, required=True, help="Metadata JSONL produced by 'prepare'")
    answer.add_argument("--output_path", type=str, required=True, help="Where to write answers JSONL")
    answer.add_argument("--knowledge_base", type=str, required=True, help="KB JSON used during preparation")
    answer.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    answer.add_argument("--qwen_device", type=str, default="cuda:0")
    answer.add_argument("--answer_temperature", type=float, default=0.0)
    answer.add_argument("--answer_max_new_tokens", type=int, default=512)
    answer.add_argument("--answer_backend", type=str, default="qwen")
    answer.add_argument("--answer_backend_device", type=str, default=None)
    answer.add_argument("--answer_backend_model_path", type=str, default=None)
    answer.add_argument("--require_reasoning", action="store_true")
    answer.add_argument("--use_image", action="store_true")
    answer.add_argument("--section_reranker_backend", choices=["auto", "bge", "echosight"], default="auto")
    answer.add_argument("--section_reranker", type=str, default=None)
    answer.add_argument("--echosight_checkpoint", type=str, default=None)
    answer.add_argument("--echosight_batch_size", type=int, default=500)
    answer.add_argument(
        "--echosight_retrieval_alpha",
        type=float,
        default=0.0,
        help="Weight (0-1) for blending retrieval similarities with EchoSight scores during answer rerank",
    )
    answer.add_argument("--answer_rerank_sections", action="store_true")
    answer.add_argument("--inat_mapping_path", type=str, default="/data/qianMa/EchoSight/images/val_id2name.json")
    answer.add_argument("--evqa_landmark_root", type=str, default="/data/qianMa/EchoSight/E-VQA/landmark")
    answer.add_argument("--log_file", type=str, default=None)
    answer.add_argument("--log_level", type=str, default="INFO")

    return parser


def _build_config_from_args(args: argparse.Namespace, phase: str) -> TopKPipelineConfig:
    # Shared values between prepare and answer phases.
    base_kwargs = dict(
        qwen_model_name=args.qwen_model_name,
        qwen_device=args.qwen_device,
        identification_top_k=getattr(args, "identification_top_k", 5),
        identification_select_top=getattr(args, "identification_select_top", 1),
        identification_temperature=getattr(args, "identification_temperature", 0.0),
        identification_max_new_tokens=getattr(args, "identification_max_new_tokens", 256),
        identification_include_similarity=getattr(args, "identification_include_similarity", False),
        context_mode=getattr(args, "context_mode", "section"),
        section_pool_size=getattr(args, "section_pool_size", 0),
        use_reranked_sections_first=getattr(args, "use_reranked_sections_first", False),
        answer_temperature=args.answer_temperature,
        answer_max_new_tokens=args.answer_max_new_tokens,
        require_reasoning=args.require_reasoning,
        answer_backend=args.answer_backend,
        answer_backend_device=_coalesce_optional_path(args.answer_backend_device),
        answer_backend_model_path=_coalesce_optional_path(args.answer_backend_model_path),
        inat_mapping_path=_coalesce_optional_path(args.inat_mapping_path),
        answer_rerank_sections=getattr(args, "answer_rerank_sections", False),
        evqa_landmark_root=_coalesce_optional_path(args.evqa_landmark_root),
        log_file=_coalesce_optional_path(args.log_file),
        log_level=args.log_level,
        section_reranker_backend=getattr(args, "section_reranker_backend", "auto"),
        section_reranker=_coalesce_optional_path(getattr(args, "section_reranker", None)),
        echosight_checkpoint=_coalesce_optional_path(getattr(args, "echosight_checkpoint", None)),
        echosight_batch_size=getattr(args, "echosight_batch_size", 500),
        echosight_retrieval_alpha=getattr(args, "echosight_retrieval_alpha", 0.0),
        entity_top_k=getattr(args, "entity_top_k", 3),
    )

    if phase == "prepare":
        return TopKPipelineConfig(
            test_file=args.test_file,
            retrieval_results=args.retrieval_results,
            knowledge_base=args.knowledge_base,
            **base_kwargs,
        )
    # answer phase – metadata already contains retrieval outputs.
    return TopKPipelineConfig(
        test_file="",
        retrieval_results="",
        knowledge_base=args.knowledge_base,
        **base_kwargs,
    )


def run_prepare(args: argparse.Namespace) -> None:
    config = _build_config_from_args(args, phase="prepare")
    pipeline = TopKQwenPipeline(config)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.prepare_metadata(args.metadata_path)


def run_answer(args: argparse.Namespace) -> None:
    config = _build_config_from_args(args, phase="answer")
    pipeline = TopKQwenPipeline(config)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.answer_from_metadata(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        use_image=args.use_image,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare(args)
    else:
        run_answer(args)


if __name__ == "__main__":
    main()
