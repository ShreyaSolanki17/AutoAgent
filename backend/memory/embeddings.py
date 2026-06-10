"""Embedding service — converts text to vectors using sentence-transformers.
"""

import os

from sentence_transformers import SentenceTransformer

# Model: all-MiniLM-L6-v2
# - 384-dimensional vectors (small, fast)
# - Good quality for semantic similarity
# - Downloads ~80MB on first run, then cached locally
# Override with EMBEDDING_MODEL env var for testing.
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton).

    Set ``EMBEDDING_MODEL`` env var to override the model name
    (useful in tests to point at a tiny fixture model).
    """
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def embed_text(text: str) -> list[float]:
    """Convert a text string to a 384-dim embedding vector.

    Args:
        text: Input text to embed.

    Returns:
        List of 384 floats representing the text.
    """
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convert multiple texts to embedding vectors.

    Args:
        texts: List of input strings.

    Returns:
        List of embedding vectors.
    """
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
