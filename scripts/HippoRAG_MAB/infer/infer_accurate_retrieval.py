import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.config import get_config
from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.llm_interface import OpenAIClient
from src.adaptors import SingleTurnAdaptor, IterativeAdaptor, PlanAndActAdaptor, AdaptorResult
from src.hipporag_memory import HippoRAGMemory
from src.openai_usage_patch import install_openai_usage_patch
from src.token_tracker import TokenTracker, set_global_tracker

logger = get_logger()


def build_adaptor(name: str, memory: HippoRAGMemory):
    conf = get_config()
    llm = OpenAIClient(
        api_key=conf.llm["api_key"],
        base_url=conf.llm["base_url"],
        model=conf.llm["model"],
    )

    if name == "R1":
        return SingleTurnAdaptor(llm, memory)
    if name == "R2":
        return IterativeAdaptor(llm, memory)
    if name == "R3":
        return PlanAndActAdaptor(llm, memory)
    raise ValueError(f"Unknown adaptor: {name}")


def evaluate_adaptor(
    name: str,
    questions: list,
    limit: int,
    memory: HippoRAGMemory,
    tracker: TokenTracker,
    dataset: str,
    instance_idx: int,
) -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    total = len(target_questions)

    for i, q in enumerate(target_questions):
        logger.info(f"[{name}] Running Q{i+1}/{total}: {q}")
        try:
            adaptor = build_adaptor(name, memory)
            with tracker.scope(
                dataset=dataset,
                instance_idx=instance_idx,
                question_idx=i,
                adaptor=name,
                stage="infer",
                substage="question",
            ):
                res: AdaptorResult = adaptor.run(q)

            results.append(
                {
                    "question": q,
                    "answer": res.answer,
                    "steps": res.steps_taken,
                    "tokens": res.token_consumption,
                    "replan": res.replan_count,
                    "token_breakdown": res.token_breakdown,
                    "api_call_count": res.api_call_count,
                }
            )
        except Exception as e:
            logger.error(f"[{name}] Failed on Q{i+1}: {e}")
            results.append({"question": q, "error": str(e)})
    return results


def evaluate_one_instance(
    instance_idx: int,
    adaptors_to_run: List[str],
    limit: int,
    index_root: str,
    output_suffix: str = "",
    openie_mode: str = "online",
    run_id: str = "",
):
    logger.info(f"=== Evaluating Instance {instance_idx} (HippoRAG) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    questions = list(data["questions"])

    # Load HippoRAG index
    index_dir = Path(index_root) / f"hipporag_acc_ret_{instance_idx}"
    if not index_dir.exists():
        logger.error(f"HippoRAG index not found: {index_dir} (run ingest first)")
        return

    tracker = TokenTracker(run_id=run_id or None)
    set_global_tracker(tracker)
    install_openai_usage_patch()

    logger.info(f"Using HippoRAG index: {index_dir}")
    logger.info(f"Token trace file: {tracker.path}")
    memory = HippoRAGMemory(index_dir=str(index_dir), openie_mode=openie_mode, force_rebuild=False)

    results = {}

    if "all" in adaptors_to_run or "R1" in adaptors_to_run:
        results["R1"] = evaluate_adaptor("R1", questions, limit, memory, tracker, "Accurate_Retrieval", instance_idx)
    if "all" in adaptors_to_run or "R2" in adaptors_to_run:
        results["R2"] = evaluate_adaptor("R2", questions, limit, memory, tracker, "Accurate_Retrieval", instance_idx)
    if "all" in adaptors_to_run or "R3" in adaptors_to_run:
        results["R3"] = evaluate_adaptor("R3", questions, limit, memory, tracker, "Accurate_Retrieval", instance_idx)

    final_report = {
        "dataset": "Accurate_Retrieval",
        "instance_idx": instance_idx,
        "run_id": tracker.run_id,
        "token_trace": str(tracker.path),
        "results": results,
    }

    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    filename = f"acc_ret_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    output_file = output_dir / filename

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Instance {instance_idx} Finished. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptors on MemoryAgentBench (HippoRAG)")
    parser.add_argument(
        "--adaptor",
        nargs="+",
        default=["all"],
        choices=["R1", "R2", "R3", "all"],
        help="Adaptors to run (e.g., R1 R2)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to run (-1 for all)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0-5', '1,3')")
    parser.add_argument("--index_root", type=str, default="out/hipporag_indices", help="Directory containing HippoRAG indices")
    parser.add_argument("--output_suffix", type=str, default="hipporag", help="Suffix for output filename")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"], help="HippoRAG OpenIE mode (usually online)")
    parser.add_argument("--run_id", type=str, default="", help="Token tracking run id")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Target adaptors: {args.adaptor}")

    for idx in indices:
        evaluate_one_instance(idx, args.adaptor, args.limit, args.index_root, args.output_suffix, args.openie_mode, args.run_id)


if __name__ == "__main__":
    main()
