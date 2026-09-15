"""
Index builder and loader for GWR customer support RAG pipeline.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

from rag.config import (
    CHUNK_STEP_TURNS,
    CHUNK_WINDOW_TURNS,
    DATA_RETRIEVAL_PATH,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    GOLDEN_REVIEW_PATH,
    INDEX_DIR,
    MAX_CHUNK_TURNS,
    MAX_CHUNK_WORDS,
    METADATA_PATH,
)
from rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


def load_golden_thread_ids(golden_file: Path = GOLDEN_REVIEW_PATH) -> Set[str]:
    """
    Load golden evaluation thread IDs to ensure they are strictly quarantined.
    """
    if not golden_file.exists():
        logger.warning("Golden review file %s not found. Proceeding with empty exclusion set.", golden_file)
        return set()

    try:
        df_golden = pd.read_csv(golden_file, usecols=["thread_id"])
        ids = set(df_golden["thread_id"].dropna().astype(str).str.strip())
        logger.info("Loaded %d golden thread IDs to quarantine.", len(ids))
        return ids
    except Exception as e:
        logger.error("Error reading golden review file %s: %s", golden_file, e)
        return set()


def split_thread_into_turns(thread_text: str) -> List[str]:
    """
    Split chronological thread text into individual speaker turns.
    Preserves multi-line turns correctly under speaker prefixes.
    """
    lines = str(thread_text).splitlines()
    turns = []
    current_turn = []

    turn_header_re = re.compile(r"^(Customer|GWRHelp|[A-Za-z0-9_]+):", re.IGNORECASE)

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        if turn_header_re.match(line_clean):
            if current_turn:
                turns.append("\n".join(current_turn))
                current_turn = []
            current_turn.append(line_clean)
        else:
            if current_turn:
                current_turn.append(line_clean)
            else:
                current_turn.append(line_clean)

    if current_turn:
        turns.append("\n".join(current_turn))

    return turns


def chunk_thread(
    thread_id: str,
    root_tweet_id: str,
    thread_text: str,
    max_words: int = MAX_CHUNK_WORDS,
    max_turns: int = MAX_CHUNK_TURNS,
    window_turns: int = CHUNK_WINDOW_TURNS,
    step_turns: int = CHUNK_STEP_TURNS
) -> List[Dict]:
    """
    Deterministically chunk a conversation thread.
    - If short (<= max_words and <= max_turns), returns 1 chunk containing whole thread.
    - If long, splits into sliding dialogue turn windows while preserving source metadata.
    """
    thread_text = str(thread_text).strip()
    if not thread_text:
        return []

    turns = split_thread_into_turns(thread_text)
    word_count = len(thread_text.split())

    # Short / normal thread: keep whole thread intact
    if word_count <= max_words and len(turns) <= max_turns:
        return [{
            "chunk_id": f"{thread_id}#0",
            "thread_id": thread_id,
            "root_tweet_id": str(root_tweet_id),
            "text": thread_text,
            "full_thread": thread_text,
            "turn_start": 0,
            "turn_end": len(turns) - 1 if turns else 0,
            "is_chunked": False,
            "word_count": word_count
        }]

    # Long thread: split into sliding turn windows
    chunks = []
    num_turns = len(turns)
    chunk_index = 0

    for start_idx in range(0, num_turns, step_turns):
        end_idx = min(start_idx + window_turns, num_turns)
        turn_slice = turns[start_idx:end_idx]
        chunk_text = "\n".join(turn_slice).strip()

        if chunk_text:
            chunks.append({
                "chunk_id": f"{thread_id}#{chunk_index}",
                "thread_id": thread_id,
                "root_tweet_id": str(root_tweet_id),
                "text": chunk_text,
                "full_thread": thread_text,
                "turn_start": start_idx,
                "turn_end": end_idx - 1,
                "is_chunked": True,
                "word_count": len(chunk_text.split())
            })
            chunk_index += 1

        if end_idx >= num_turns:
            break

    return chunks


def prepare_retrieval_chunks(
    retrieval_file: Path = DATA_RETRIEVAL_PATH,
    golden_file: Path = GOLDEN_REVIEW_PATH
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Load data_retrieval.csv, enforce golden set quarantine, chunk long threads,
    and return list of metadata dicts ready for embedding.
    """
    if not retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval file not found at: {retrieval_file}")

    golden_ids = load_golden_thread_ids(golden_file)
    logger.info("Reading retrieval corpus from %s...", retrieval_file)

    try:
        df = pd.read_csv(retrieval_file)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV at {retrieval_file}: {e}")

    for col in ["thread_id", "root_tweet_id", "data"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {retrieval_file}")

    total_input_threads = len(df)
    golden_violations = 0
    all_chunks = []
    short_threads_count = 0
    chunked_threads_count = 0

    for row in df.itertuples(index=False):
        t_id = str(row.thread_id).strip()
        r_id = str(row.root_tweet_id).strip()
        data_text = str(row.data) if pd.notna(row.data) else ""

        # Strictly quarantine any golden thread
        if t_id in golden_ids:
            golden_violations += 1
            continue

        if not data_text.strip():
            continue

        thread_chunks = chunk_thread(t_id, r_id, data_text)
        if not thread_chunks:
            continue

        if len(thread_chunks) == 1 and not thread_chunks[0]["is_chunked"]:
            short_threads_count += 1
        else:
            chunked_threads_count += 1

        all_chunks.extend(thread_chunks)

    if golden_violations > 0:
        logger.warning(
            "QUARANTINED %d threads that matched golden review IDs!", golden_violations
        )

    stats = {
        "total_input_threads": total_input_threads,
        "golden_violations_prevented": golden_violations,
        "indexed_threads": short_threads_count + chunked_threads_count,
        "short_threads": short_threads_count,
        "chunked_threads": chunked_threads_count,
        "total_chunks": len(all_chunks)
    }

    logger.info(
        "Prepared %d chunks from %d threads (short: %d, chunked: %d)",
        stats["total_chunks"],
        stats["indexed_threads"],
        stats["short_threads"],
        stats["chunked_threads"]
    )

    return all_chunks, stats


def build_faiss_index(
    chunks: List[Dict],
    embedding_model: Optional[EmbeddingModel] = None,
    output_index_path: Path = FAISS_INDEX_PATH,
    output_metadata_path: Path = METADATA_PATH,
    batch_size: int = 128
) -> Tuple[object, Path, Path]:
    """
    Compute embeddings for all chunks, create a FAISS IndexFlatIP, and save artifacts.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is not installed. Please install faiss-cpu.")

    if not chunks:
        raise ValueError("No chunks provided to build index.")

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    logger.info("Extracting chunk texts for embedding (%d chunks)...", len(chunks))
    texts = [c["text"] for c in chunks]

    logger.info("Encoding embeddings using model '%s'...", embedding_model.model_name)
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize=True
    )

    dim = embeddings.shape[1]
    logger.info("Creating FAISS IndexFlatIP with dimension %d...", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Added %d vectors to FAISS index.", index.ntotal)

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing FAISS index to %s...", output_index_path)
    faiss.write_index(index, str(output_index_path))

    logger.info("Writing metadata to %s...", output_metadata_path)
    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return index, output_index_path, output_metadata_path


def load_index(
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH
) -> Tuple[object, List[Dict]]:
    """
    Load persisted FAISS index and metadata.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is not installed. Please install faiss-cpu.")

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found at: {index_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index dimension mismatch: FAISS index has {index.ntotal} vectors, "
            f"but metadata has {len(metadata)} entries."
        )

    return index, metadata
