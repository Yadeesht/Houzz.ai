"""
Script to build and persist the FAISS index and metadata for GWR customer support retrieval.

Usage:
    python scripts/build_index.py
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import (
    DATA_RETRIEVAL_PATH,
    DEFAULT_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    GOLDEN_REVIEW_PATH,
    METADATA_PATH,
)
from rag.embeddings import EmbeddingModel
from rag.index import build_faiss_index, prepare_retrieval_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("build_index")


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS Index for GWR Support Retrieval")
    parser.add_argument(
        "--retrieval-file",
        type=Path,
        default=DATA_RETRIEVAL_PATH,
        help="Path to retrieval CSV"
    )
    parser.add_argument(
        "--golden-file",
        type=Path,
        default=GOLDEN_REVIEW_PATH,
        help="Path to golden review CSV for strict quarantine"
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=FAISS_INDEX_PATH,
        help="Path to save FAISS index"
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=METADATA_PATH,
        help="Path to save metadata JSON"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("GWR SUPPORT PIPELINE: FAISS INDEX BUILDER")
    print("=" * 70)
    print(f"Retrieval file:     {args.retrieval_file}")
    print(f"Golden review file: {args.golden_file}")
    print(f"Embedding model:    {args.model_name}")
    print(f"Output FAISS index: {args.output_index}")
    print(f"Output metadata:    {args.output_metadata}")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Prepare chunks with golden set quarantine
    logger.info("Step 1: Reading data and preparing chunks...")
    chunks, stats = prepare_retrieval_chunks(
        retrieval_file=args.retrieval_file,
        golden_file=args.golden_file
    )

    print("\nCorpus Statistics:")
    print(f"  - Total input rows:               {stats['total_input_threads']:,}")
    print(f"  - Golden threads quarantined:     {stats['golden_violations_prevented']:,}")
    print(f"  - Usable threads indexed:         {stats['indexed_threads']:,}")
    print(f"  - Short threads (whole document): {stats['short_threads']:,}")
    print(f"  - Long threads (turn-chunked):    {stats['chunked_threads']:,}")
    print(f"  - Total vector chunks created:    {stats['total_chunks']:,}\n")

    # Step 2: Initialize embedding model
    logger.info("Step 2: Loading embedding model '%s'...", args.model_name)
    embedding_model = EmbeddingModel(model_name=args.model_name)

    # Step 3: Embed and build FAISS index
    logger.info("Step 3: Embedding chunks and creating FAISS index...")
    index, index_path, meta_path = build_faiss_index(
        chunks=chunks,
        embedding_model=embedding_model,
        output_index_path=args.output_index,
        output_metadata_path=args.output_metadata,
        batch_size=args.batch_size
    )

    elapsed = time.time() - start_time

    print("=" * 70)
    print("INDEX BUILD SUCCESSFUL")
    print("=" * 70)
    print(f"Vectors indexed:  {index.ntotal:,}")
    print(f"Index file:       {index_path} ({index_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Metadata file:    {meta_path} ({meta_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Total time taken: {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
