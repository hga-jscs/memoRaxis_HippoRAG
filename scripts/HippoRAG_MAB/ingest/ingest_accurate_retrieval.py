import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:\memoRaxis
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.hipporag_memory import HippoRAGMemory
from src.openai_usage_patch import install_openai_usage_patch
from src.token_tracker import TokenTracker, set_global_tracker

logger = get_logger()


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    save_root: str,
    openie_mode: str,
    force_rebuild: bool,
    max_chunks: int,
    run_id: str,
):
    logger.info(f"=== Processing Instance {instance_idx} (HippoRAG) ===")
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error(f"Error loading instance {instance_idx}: {e}")
        return

    tracker = TokenTracker(run_id=run_id or None)
    set_global_tracker(tracker)
    install_openai_usage_patch()
    logger.info(f"[TokenTracker] run_id={tracker.run_id} trace={tracker.path}")

    with tracker.scope(dataset="Accurate_Retrieval", instance_idx=instance_idx, stage="ingest", substage="chunk"):
        chunks = chunk_context(data["context"], chunk_size=chunk_size)
        tracker.record(
            provider="local",
            api_kind="chunker",
            model="regex_or_sliding_window",
            prompt_chars=len(data["context"]),
            output_chars=sum(len(c) for c in chunks),
            extra={"chunk_count": len(chunks), "chunk_size": chunk_size},
        )

    if max_chunks > 0:
        chunks = chunks[:max_chunks]
        logger.warning(f"[HippoRAG] max_chunks={max_chunks}, only indexing first {len(chunks)} chunks.")

    index_dir = Path(save_root) / f"hipporag_acc_ret_{instance_idx}"
    memory = HippoRAGMemory(index_dir=str(index_dir), openie_mode=openie_mode, force_rebuild=force_rebuild)

    print(f"Starting ingestion of {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        memory.add_memory(chunk, metadata={"doc_id": i, "instance_idx": instance_idx})
        if i % 10 == 0:
            print(f"Queued {i}/{len(chunks)} chunks...", end="\r", flush=True)

    with tracker.scope(dataset="Accurate_Retrieval", instance_idx=instance_idx, stage="ingest", substage="build_index"):
        memory.build_index()

    print(f"\nIngestion complete. HippoRAG index saved to {index_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data (HippoRAG)")
    parser.add_argument("--instance_idx", type=str, default="0", help="Index range (e.g., '0', '0-5', '1,3')")
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
    parser.add_argument("--save_root", type=str, default="out/hipporag_indices", help="Where to save HippoRAG indices")
    parser.add_argument("--openie_mode", type=str, default="online", choices=["online", "offline"], help="HippoRAG OpenIE mode")
    parser.add_argument("--force_rebuild", action="store_true", help="Delete existing index_dir and rebuild")
    parser.add_argument("--max_chunks", type=int, default=-1, help="Debug: only index first N chunks (-1 = all)")
    parser.add_argument("--run_id", type=str, default="", help="Token tracking run id")
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info(f"Target instances: {indices}")

    for idx in indices:
        ingest_one_instance(idx, args.chunk_size, args.save_root, args.openie_mode, args.force_rebuild, args.max_chunks, args.run_id)


if __name__ == "__main__":
    main()
