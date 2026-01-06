"""Command line entrypoints for the Qwen VQA pipeline."""

from __future__ import annotations

import argparse

from .pipeline import PipelineConfig, QwenVQAPipeline
from .pipeline_echosight import EchoSightPipelineConfig, EchoSightVQAPipeline
from utils import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen-2.5-VL VQA pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Run identification and context selection")
    prepare.add_argument("--test_file", type=str, required=True)
    prepare.add_argument("--retrieval_results", type=str, required=True)
    prepare.add_argument("--knowledge_base", type=str, required=True)
    prepare.add_argument("--metadata_path", type=str, required=True)
    prepare.add_argument("--pipeline_variant", choices=["qwen", "echosight"], default="qwen")
    prepare.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    prepare.add_argument("--qwen_device", type=str, default="cuda:0")
    prepare.add_argument("--identification_top_k", type=int, default=5)
    prepare.add_argument("--identification_temperature", type=float, default=0.0)
    prepare.add_argument("--identification_max_new_tokens", type=int, default=256)
    prepare.add_argument(
        "--identification_include_similarity",
        action="store_true",
        help="Include initial retrieval image similarities in the identification prompt.",
    )
    prepare.add_argument("--context_mode", type=str, choices=["article", "section"], default="article")
    prepare.add_argument("--section_reranker", type=str, default=None)
    prepare.add_argument("--echosight_checkpoint", type=str, default=None)
    prepare.add_argument("--echosight_batch_size", type=int, default=64)
    prepare.add_argument(
        "--section_pool_size",
        type=int,
        default=0,
        help="Deprecated; kept for compatibility. Non-positive values use all sections.",
    )
    prepare.add_argument("--use_reranked_sections_first", action="store_true")
    prepare.add_argument("--answer_temperature", type=float, default=0.0)
    prepare.add_argument("--answer_max_new_tokens", type=int, default=512)
    prepare.add_argument("--require_reasoning", action="store_true")
    prepare.add_argument("--answer_backend", type=str, default="qwen")
    prepare.add_argument("--answer_backend_device", type=str, default=None)
    prepare.add_argument("--answer_backend_model_path", type=str, default=None)
    prepare.add_argument("--inat_mapping_path", type=str, default="$ROOT_PATH/images/val_id2name.json")
    prepare.add_argument("--answer_rerank_sections", action="store_true")
    prepare.add_argument("--evqa_landmark_root", type=str, default="$ROOT_PATH/E-VQA/landmark")
    prepare.add_argument("--log_file", type=str, default=None)
    prepare.add_argument("--log_level", type=str, default="INFO")

    answer = subparsers.add_parser("answer", help="Generate answers using cached metadata")
    answer.add_argument("--metadata_path", type=str, required=True)
    answer.add_argument("--output_path", type=str, required=True)
    answer.add_argument("--knowledge_base", type=str, required=True)
    answer.add_argument("--pipeline_variant", choices=["qwen", "echosight"], default="qwen")
    answer.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    answer.add_argument("--qwen_device", type=str, default="cuda:0")
    answer.add_argument("--answer_temperature", type=float, default=0.0)
    answer.add_argument("--answer_max_new_tokens", type=int, default=512)
    answer.add_argument("--answer_backend", type=str, default="qwen")
    answer.add_argument("--answer_backend_device", type=str, default=None)
    answer.add_argument("--answer_backend_model_path", type=str, default=None)
    answer.add_argument("--require_reasoning", action="store_true")
    answer.add_argument("--use_image", action="store_true")
    answer.add_argument("--inat_mapping_path", type=str, default="$ROOT_PATH/images/val_id2name.json")
    answer.add_argument("--section_reranker", type=str, default=None)
    answer.add_argument("--echosight_checkpoint", type=str, default=None)
    answer.add_argument("--echosight_batch_size", type=int, default=64)
    answer.add_argument("--answer_rerank_sections", action="store_true")
    answer.add_argument("--evqa_landmark_root", type=str, default="$ROOT_PATH/E-VQA/landmark")
    answer.add_argument("--log_file", type=str, default=None)
    answer.add_argument("--log_level", type=str, default="INFO")
    answer.add_argument("--dataset_name", type=str, default="evqa",
                    help="Override dataset identifier (e.g. 'infoseek', 'evqa').")


    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seed_everything(42)
    print("#"*50)
    print(f"Seeded everything with 42")
    print("#"*50)

    if args.command == "prepare":
        variant = getattr(args, "pipeline_variant", "qwen")
        if variant == "echosight":
            config = EchoSightPipelineConfig(
                test_file=args.test_file,
                retrieval_results=args.retrieval_results,
                knowledge_base=args.knowledge_base,
                qwen_model_name=args.qwen_model_name,
                qwen_device=args.qwen_device,
                identification_top_k=args.identification_top_k,
                identification_include_similarity=args.identification_include_similarity,
                identification_temperature=args.identification_temperature,
                identification_max_new_tokens=args.identification_max_new_tokens,
                context_mode=args.context_mode,
                section_reranker=args.section_reranker,
                section_pool_size=args.section_pool_size,
                use_reranked_sections_first=args.use_reranked_sections_first,
                answer_temperature=args.answer_temperature,
                answer_max_new_tokens=args.answer_max_new_tokens,
                require_reasoning=args.require_reasoning,
                answer_backend=args.answer_backend,
                answer_backend_device=args.answer_backend_device,
                answer_backend_model_path=args.answer_backend_model_path,
                inat_mapping_path=args.inat_mapping_path,
                answer_rerank_sections=args.answer_rerank_sections,
                evqa_landmark_root=args.evqa_landmark_root,
                log_file=args.log_file,
                log_level=args.log_level,
                echosight_checkpoint=args.echosight_checkpoint or args.section_reranker,
                echosight_batch_size=args.echosight_batch_size,
            )
            pipeline = EchoSightVQAPipeline(config)
        else:
            config = PipelineConfig(
                test_file=args.test_file,
                retrieval_results=args.retrieval_results,
                knowledge_base=args.knowledge_base,
                qwen_model_name=args.qwen_model_name,
                qwen_device=args.qwen_device,
                identification_top_k=args.identification_top_k,
                identification_include_similarity=args.identification_include_similarity,
                identification_temperature=args.identification_temperature,
                identification_max_new_tokens=args.identification_max_new_tokens,
                context_mode=args.context_mode,
                section_reranker=args.section_reranker,
                section_pool_size=args.section_pool_size,
                use_reranked_sections_first=args.use_reranked_sections_first,
                answer_temperature=args.answer_temperature,
                answer_max_new_tokens=args.answer_max_new_tokens,
                require_reasoning=args.require_reasoning,
                answer_backend=args.answer_backend,
                answer_backend_device=args.answer_backend_device,
                answer_backend_model_path=args.answer_backend_model_path,
                inat_mapping_path=args.inat_mapping_path,
                answer_rerank_sections=args.answer_rerank_sections,
                evqa_landmark_root=args.evqa_landmark_root,
                log_file=args.log_file,
                log_level=args.log_level,
            )
            pipeline = QwenVQAPipeline(config)
        pipeline.prepare_metadata(args.metadata_path)
    else:
        variant = getattr(args, "pipeline_variant", "qwen")
        if variant == "echosight":
            config = EchoSightPipelineConfig(
                test_file="",
                retrieval_results="",
                knowledge_base=args.knowledge_base,
                qwen_model_name=args.qwen_model_name,
                qwen_device=args.qwen_device,
                answer_temperature=args.answer_temperature,
                answer_max_new_tokens=args.answer_max_new_tokens,
                require_reasoning=args.require_reasoning,
                answer_backend=args.answer_backend,
                answer_backend_device=args.answer_backend_device,
                answer_backend_model_path=args.answer_backend_model_path,
                inat_mapping_path=args.inat_mapping_path,
                section_reranker=args.section_reranker,
                answer_rerank_sections=args.answer_rerank_sections,
                evqa_landmark_root=args.evqa_landmark_root,
                log_file=args.log_file,
                log_level=args.log_level,
                echosight_checkpoint=args.echosight_checkpoint or args.section_reranker,
                echosight_batch_size=args.echosight_batch_size,
            )
            pipeline = EchoSightVQAPipeline(config)
        else:
            config = PipelineConfig(
                test_file="",
                retrieval_results="",
                knowledge_base=args.knowledge_base,
                qwen_model_name=args.qwen_model_name,
                qwen_device=args.qwen_device,
                answer_temperature=args.answer_temperature,
                answer_max_new_tokens=args.answer_max_new_tokens,
                require_reasoning=args.require_reasoning,
                answer_backend=args.answer_backend,
                answer_backend_device=args.answer_backend_device,
                answer_backend_model_path=args.answer_backend_model_path,
                inat_mapping_path=args.inat_mapping_path,
                section_reranker=args.section_reranker,
                answer_rerank_sections=args.answer_rerank_sections,
                evqa_landmark_root=args.evqa_landmark_root,
                log_file=args.log_file,
                log_level=args.log_level,
            )
            pipeline = QwenVQAPipeline(config)
        pipeline.answer_from_metadata(
            metadata_path=args.metadata_path,
            output_path=args.output_path,
            use_image=args.use_image,
            dataset_name=args.dataset_name,
        )


if __name__ == "__main__":
    main()
