"""Embedding service — converts text to vectors using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer

# Model: all-MiniLM-L6-v2
# - 384-dimensional vectors (small, fast)
# - Good quality for semantic similarity
# - Downloads ~80MB on first run, then cached locally
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
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
