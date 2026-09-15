"""
Embedding utilities using sentence-transformers.
"""
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from rag.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer with L2 normalization for cosine similarity.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it via: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text or list of texts to numpy float32 embeddings.
        Applies L2 normalization by default so inner product equals cosine similarity.
        """
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else list(texts)

        if not input_texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            input_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        ).astype(np.float32)

        return embeddings[0] if is_single else embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string for FAISS retrieval.
        Returns shape (1, dimension).
        """
        if not query or not query.strip():
            # Return zero vector if query is empty
            vec = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
            return vec

        embedding = self.encode(query.strip(), normalize=True)
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        return embedding
