import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices, chunk_context
from src.hipporag_memory import HippoRAGMemory
from src.openai_usage_patch import install_openai_usage_patch
from src.token_tracker import TokenTracker, set_global_tracker

logger = get_logger()


def ingest_one_instance(instance_idx: int, chunk_size: int, overlap: int, save_root: str, openie_mode: str, force_rebuild: bool, max_chunks: int, run_id: str):
    logger.info(f"=== Processing Long_Range_Understanding Instance {instance_idx} (HippoRAG) ===")
    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"
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

    with tracker.scope(dataset="Long_Range_Understanding", instance_idx=instance_idx, stage="ingest", substage="chunk"):
        chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)
        tracker.record(provider="local", api_kind="chunker", model="sliding_window", prompt_chars=len(data["context"]), output_chars=sum(len(c) for c in chunks), extra={"chunk_count": len(chunks), "chunk_size": chunk_size, "overlap": overlap})

    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        logger.warning(f"[HippoRAG] max_chunks={max_chunks}, only indexing first {len(chunks)} chunks.")

    index_dir = Path(save_root) / f"hipporag_long_range_{instance_idx}"
    memory = HippoRAGMemory(index_dir=str(index_dir), openie_mode=openie_mode, force_rebuild=force_rebuild)

    print(f"Starting ingestion for Instance {instance_idx} ({len(chunks)} chunks)...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"chunk_id": i, "instance_idx": instance_idx})
        if i % 100 == 0:
            print(f"  Queued {i}/{len(chunks)}...", end="\r", flush=True)

    with tracker.scope(dataset="Long_Range_Understanding", instance_idx=instance_idx, stage="ingest", substage="build_index"):
        memory.build_index()

    print(f"\nInstance {instance_idx} complete. HippoRAG index saved -> {index_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-39")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--save_root", type=str, default="out/hipporag_indices")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"])
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--max_chunks", type=int, default=-1)
    parser.add_argument("--run_id", type=str, default="", help="Token tracking run id")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")
    logger.info(f"Config: Chunk Size={args.chunk_size}, Overlap={args.overlap}")
    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.overlap, args.save_root, args.openie_mode, args.force_rebuild, args.max_chunks, args.run_id)


if __name__ == "__main__":
    main()
