"""Embed text into 384-dim unit vectors with all-MiniLM-L6-v2, on CPU.

Ported from triage-lab's `s2_embed.py`, keeping the model singleton (with its
lock against the FastAPI threadpool hazard), `embed_texts`, and `cosine`. There is
no chunk budget or max-sequence-length trade to document here -- embed-lab only
ever embeds single words and short phrases, well inside any model's window.
"""

import threading
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.errors import StageError

EMBED_DIM = 384

# Serialises the cold construction below. See get_model().
_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(
            settings.embed_model,
            revision=settings.embed_model_revision,
            cache_folder=str(settings.model_cache_dir),
            device="cpu",
        )
    except OSError as err:
        raise StageError(
            detail=f"Could not load the embedding model {settings.embed_model!r}.",
            hint="Run it once with a network connection so the weights cache, "
                 "then retry.",
        ) from err


def get_model() -> SentenceTransformer:
    """The embedding model, loaded once per process (about 3-6 s cold).

    `lru_cache` alone is not enough here. It does not hold a lock across the
    wrapped call, and FastAPI dispatches sync `def` endpoints to a worker
    threadpool -- so two first-hits on /api/embed both miss the cache and both
    construct a SentenceTransformer (~0.5 GB, seconds each), and one of them is
    then thrown away. The answer is one instance behind a lock rather than one
    per thread.

    The lock is taken only while the cache is cold: once it is warm every call
    is a plain lru_cache hit with no synchronisation. The second waiter
    re-enters `_load_model()` inside the lock, which is now a cache hit -- the
    double-check -- so it returns the first thread's model instead of building
    another.
    """
    if _load_model.cache_info().currsize:
        return _load_model()
    with _model_lock:
        return _load_model()


# Tests clear the cache to force a reload (e.g. to make a monkeypatched
# SentenceTransformer take effect). Keep that working through the wrapper.
get_model.cache_clear = _load_model.cache_clear
get_model.cache_info = _load_model.cache_info


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts into (n, 384) float32 unit vectors.

    normalize_embeddings is NOT passed -- the model's Normalize module already
    returns unit vectors, so asking again would be a redundant second pass.
    """
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    return get_model().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Equals the dot product here because vectors are unit-length,
    but the division is kept so this stays correct if a caller passes raw vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
