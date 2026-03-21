import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.config import get_config
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
    questions: List[str],
    answers: List[str],
    memory: HippoRAGMemory,
    tracker: TokenTracker,
    dataset: str,
    instance_idx: int,
) -> list:
    adaptor = build_adaptor(name, memory)
    results = []
    total = len(questions)

    for i, (q, a) in enumerate(zip(questions, answers)):
        logger.info(f"[{dataset}][{name}] Q{i+1}/{total}: {q}")
        try:
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
                    "ground_truth": a,
                    "steps": res.steps_taken,
                    "tokens": res.token_consumption,
                    "replan": res.replan_count,
                    "token_breakdown": res.token_breakdown,
                    "api_call_count": res.api_call_count,
                }
            )
            logger.info(
                f"[{dataset}][{name}] Q{i+1} done | tokens={res.token_consumption} | "
                f"calls={res.api_call_count} | steps={res.steps_taken} | replan={res.replan_count}"
            )
        except Exception as exc:
            logger.error(f"[{dataset}][{name}] Failed on Q{i+1}: {exc}")
            results.append({"question": q, "ground_truth": a, "error": str(exc)})
    return results


def evaluate_instance(
    instance_idx: int,
    adaptors: list,
    limit: int = -1,
    output_suffix: str = "",
    index_root: str = "out/hipporag_indices",
    openie_mode: str = "online",
    run_id: str = "",
):
    dataset = "Conflict_Resolution"
    logger.info(f"=== Evaluating {dataset} Instance {instance_idx} (HippoRAG) ===")

    data_path = f"MemoryAgentBench/preview_samples/{dataset}/instance_{instance_idx}.json"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index_dir = Path(index_root) / f"hipporag_conflict_{instance_idx}"
    if not index_dir.exists():
        logger.error(f"HippoRAG index not found: {index_dir} (run ingest first)")
        return

    tracker = TokenTracker(run_id=run_id or None)
    set_global_tracker(tracker)
    install_openai_usage_patch()
    logger.info(f"[TokenTracker] run_id={tracker.run_id} trace={tracker.path}")

    logger.info(f"Using HippoRAG index: {index_dir}")
    memory = HippoRAGMemory(index_dir=str(index_dir), openie_mode=openie_mode, force_rebuild=False)

    questions = data["questions"]
    answers = data["answers"]
    if limit > 0:
        questions = questions[:limit]
        answers = answers[:limit]

    results = {
        "dataset": dataset,
        "instance_idx": instance_idx,
        "run_id": tracker.run_id,
        "token_trace": str(tracker.path),
        "results": {},
    }

    for adaptor_name in adaptors:
        logger.info(f"[{dataset}] Running adaptor={adaptor_name} with run_id={tracker.run_id}")
        try:
            results["results"][adaptor_name] = evaluate_adaptor(
                name=adaptor_name,
                questions=questions,
                answers=answers,
                memory=memory,
                tracker=tracker,
                dataset=dataset,
                instance_idx=instance_idx,
            )
        except ValueError as exc:
            logger.warning(str(exc))

    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"conflict_res_results_{instance_idx}"
    if output_suffix:
        filename += f"_{output_suffix}"
    filename += ".json"
    out_file = output_dir / filename

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"[{dataset}] Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conflict_Resolution (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="e.g., '0-7'")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"], help="R1, R2, R3")
    parser.add_argument("--limit", type=int, default=-1, help="Limit questions per instance")
    parser.add_argument("--index_root", type=str, default="out/hipporag_indices", help="Directory containing HippoRAG indices")
    parser.add_argument("--output_suffix", type=str, default="hipporag", help="Suffix for output filename")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"], help="HippoRAG OpenIE mode (should match ingest)")
    parser.add_argument("--run_id", type=str, default="", help="Token tracking run id")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices} | adaptors={args.adaptor} | run_id={args.run_id or '<auto>'}")
    for idx in indices:
        evaluate_instance(idx, args.adaptor, args.limit, args.output_suffix, args.index_root, args.openie_mode, args.run_id)


if __name__ == "__main__":
    main()
