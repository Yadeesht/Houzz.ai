"""
Retriever component for semantic search over indexed support conversations.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from rag.config import (
    DEFAULT_TOP_K,
    FAISS_INDEX_PATH,
    GOLDEN_REVIEW_PATH,
    METADATA_PATH,
)
from rag.embeddings import EmbeddingModel
from rag.index import load_golden_thread_ids, load_index

logger = logging.getLogger(__name__)


def build_retrieval_query(conversation_context: str, customer_message: str) -> str:
    """
    Construct a dense retrieval query combining recent conversation context
    and the latest customer message.
    """
    ctx = (conversation_context or "").strip()
    msg = (customer_message or "").strip()

    if not ctx and not msg:
        return ""

    if not ctx:
        return msg

    if not msg:
        return ctx

    # Extract the most recent lines of context (up to 3 turns) if context is long
    ctx_lines = [line.strip() for line in ctx.splitlines() if line.strip()]
    recent_ctx = " ".join(ctx_lines[-4:]) if len(ctx_lines) > 4 else " ".join(ctx_lines)

    return f"Context: {recent_ctx} | Latest problem: {msg}".strip()


class Retriever:
    """
    Semantic retriever querying the FAISS index with L2-normalized embeddings.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        golden_file: Path = GOLDEN_REVIEW_PATH,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.golden_file = Path(golden_file)
        self.embedding_model = embedding_model or EmbeddingModel()

        self._index = None
        self._metadata = None
        self._golden_ids = None

    def _ensure_loaded(self):
        if self._index is None or self._metadata is None:
            self._index, self._metadata = load_index(self.index_path, self.metadata_path)
            self._golden_ids = load_golden_thread_ids(self.golden_file)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        deduplicate_threads: bool = True
    ) -> List[Dict]:
        """
        Retrieve top_k historical conversation chunks semantically closest to the query.

        Args:
            query: Customer inquiry / conversation context string.
            top_k: Number of retrieved results to return.
            deduplicate_threads: If True, returns at most one chunk per unique thread_id.

        Returns:
            List of dicts: [
                {
                    "thread_id": "...",
                    "root_tweet_id": "...",
                    "text": "...",
                    "score": float,
                    "chunk_id": "...",
                    "turn_start": int,
                    "turn_end": int
                },
                ...
            ]
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        self._ensure_loaded()

        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return []

        # Encode query
        query_vec = self.embedding_model.encode_query(query)

        # Retrieve extra candidates if deduplicating threads
        search_k = min(top_k * 4 if deduplicate_threads else top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, search_k)

        results = []
        seen_threads = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]
            thread_id = meta.get("thread_id")

            # Absolute guarantee: skip golden evaluation set items
            if self._golden_ids and thread_id in self._golden_ids:
                logger.error("Golden thread %s leaked into index! Quarantined from results.", thread_id)
                continue

            if deduplicate_threads:
                if thread_id in seen_threads:
                    continue
                seen_threads.add(thread_id)

            results.append({
                "thread_id": thread_id,
                "root_tweet_id": meta.get("root_tweet_id", ""),
                "text": meta.get("text", ""),
                "score": float(score),
                "chunk_id": meta.get("chunk_id", ""),
                "turn_start": meta.get("turn_start", 0),
                "turn_end": meta.get("turn_end", 0),
            })

            if len(results) >= top_k:
                break

        return results
