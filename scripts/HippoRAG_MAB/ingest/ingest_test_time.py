import argparse
import re
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.hipporag_memory import HippoRAGMemory
from src.openai_usage_patch import install_openai_usage_patch
from src.token_tracker import TokenTracker, set_global_tracker

logger = get_logger()


def chunk_dialogues(context: str) -> List[str]:
    parts = re.split(r"\n(Dialogue \d+:)", "\n" + context)
    chunks = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        full_text = f"{header}\n{body}"
        if len(full_text) > 10:
            chunks.append(full_text)
    return chunks


def chunk_accumulation(context: str, min_chars: int = 800) -> List[str]:
    lines = [line.strip() for line in context.split("\n") if line.strip()]
    chunks, current_chunk_lines, current_length = [], [], 0
    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)
        if current_length > min_chars:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines, current_length = [], 0
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
    return chunks


def ingest_one_instance(instance_idx: int, save_root: str, openie_mode: str, force_rebuild: bool, max_chunks: int, run_id: str):
    logger.info(f"=== Processing TTL Instance {instance_idx} (HippoRAG) ===")
    data_path = f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return
    try:
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    tracker = TokenTracker(run_id=run_id or None)
    set_global_tracker(tracker)
    install_openai_usage_patch()
    logger.info(f"[TokenTracker] run_id={tracker.run_id} trace={tracker.path}")

    context = data["context"]
    with tracker.scope(dataset="Test_Time_Learning", instance_idx=instance_idx, stage="ingest", substage="chunk"):
        if "Dialogue 1:" in context[:500]:
            logger.info("Strategy: Regex Split (Dialogue mode)")
            chunks = chunk_dialogues(context)
            model_name = "dialogue_regex"
        else:
            logger.info("Strategy: Accumulation > 800 chars (ShortText mode)")
            chunks = chunk_accumulation(context, min_chars=800)
            model_name = "line_accumulation"
        tracker.record(provider="local", api_kind="chunker", model=model_name, prompt_chars=len(context), output_chars=sum(len(c) for c in chunks), extra={"chunk_count": len(chunks)})

    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        logger.warning(f"[HippoRAG] max_chunks={max_chunks}, only indexing first {len(chunks)} chunks.")

    index_dir = Path(save_root) / f"hipporag_ttl_{instance_idx}"
    memory = HippoRAGMemory(index_dir=str(index_dir), openie_mode=openie_mode, force_rebuild=force_rebuild)

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 50 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    with tracker.scope(dataset="Test_Time_Learning", instance_idx=instance_idx, stage="ingest", substage="build_index"):
        memory.build_index()

    print(f"\nInstance {instance_idx} complete. HippoRAG index saved -> {index_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Test_Time_Learning data (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-5")
    parser.add_argument("--save_root", type=str, default="out/hipporag_indices")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"])
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--max_chunks", type=int, default=-1)
    parser.add_argument("--run_id", type=str, default="", help="Token tracking run id")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    for idx in indices:
        ingest_one_instance(idx, args.save_root, args.openie_mode, args.force_rebuild, args.max_chunks, args.run_id)


if __name__ == "__main__":
    main()
