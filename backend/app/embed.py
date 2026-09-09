"""Embed text into 384-dim unit vectors with all-MiniLM-L6-v2, on CPU, via ONNX Runtime.

Runtime path: onnxruntime + tokenizers + numpy only -- no torch, no
sentence-transformers, no transformers. That is the whole point of this
module: a resident torch install pushed the process to ~500 MB working set,
leaving no headroom on a 512 MB free host. onnxruntime running the same
weights does not need torch at all.

The ONNX file at `settings.onnx_model_path` is produced OFFLINE by
`scripts/export_onnx.py` (dev machine, torch allowed) from the pinned
revision `settings.embed_model_revision`. It is the bare transformer
(`BertModel`, outputting `last_hidden_state`) -- NOT the full
SentenceTransformer pipeline. The two steps SentenceTransformer would run
after that -- mean pooling and L2 normalisation -- are reimplemented here in
numpy, because getting either subtly wrong (e.g. averaging over padding
positions, or normalising before pooling) produces embeddings that look
completely plausible and are quietly wrong: every downstream number
(macro-F1, ARI, every cosine similarity) would shift with no error raised.
See `tests/test_onnx_equivalence.py`, which is the gate for this file: it
embeds 30+ words with both the retired torch path and this one and asserts
they agree to within float32 tolerance.

Mean pooling, attention-mask weighted, EXACTLY as sentence-transformers'
`Pooling` module does it:
    sum_embeddings = sum(token_embeddings * mask, axis=seq)
    sum_mask       = clamp(sum(mask, axis=seq), min=1e-9)
    mean_pooled    = sum_embeddings / sum_mask
then L2-normalise mean_pooled (never the other order -- normalising before
pooling is one of the silent-breakage modes this module exists to avoid).
"""

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

from app.config import settings
from app.errors import StageError

EMBED_DIM = 384

# Serialises the cold construction below. See get_model().
_model_lock = threading.Lock()


def _load_embedding_tokenizer() -> Tokenizer:
    """A tokenizer.json-backed Tokenizer configured for BATCH embedding.

    Deliberately separate from app.tokenize.get_tokenizer(), which must stay
    padding/truncation-off (it exists to report true wordpiece counts). This
    one needs padding on so a batch of texts with different lengths can share
    one ONNX Run() call, and its pad settings must match how the exported
    model was traced: pad with id 0 ("[PAD]" in this vocab), no truncation
    (embed-lab only ever embeds single words and short phrases -- well inside
    any sequence-length limit worth enforcing).
    """
    cache_dir = settings.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=settings.embed_model,
            filename="tokenizer.json",
            revision=settings.embed_model_revision,
            cache_dir=str(cache_dir),
        )
    except LocalEntryNotFoundError as err:
        raise StageError(
            detail="The tokenizer is not cached locally and there is no network "
                   "connection to download it.",
            hint="Run once with a network connection so the tokenizer caches, "
                 "then retry.",
        ) from err
    tokenizer = Tokenizer.from_file(path)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.no_truncation()
    return tokenizer


def _load_session(onnx_path: Path) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise StageError(
            detail=f"The ONNX model is not present at {onnx_path}.",
            hint="Run `uv run python scripts/export_onnx.py` (on a machine with "
                 "torch installed) to produce it, then retry.",
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.onnx_threads
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


@lru_cache(maxsize=1)
def _load_model() -> tuple[ort.InferenceSession, Tokenizer]:
    session = _load_session(settings.onnx_model_path)
    tokenizer = _load_embedding_tokenizer()
    return session, tokenizer


def get_model() -> tuple[ort.InferenceSession, Tokenizer]:
    """The ONNX session + its tokenizer, built once per process.

    Kept as the same double-checked-lock shape as the retired torch version:
    `lru_cache` alone does not hold a lock across the wrapped call, and
    FastAPI dispatches sync `def` endpoints to a worker threadpool, so two
    first-hits could both miss the cache and both construct a session. The
    ONNX session is far lighter than a torch model was, but there is still no
    reason to build it twice.
    """
    if _load_model.cache_info().currsize:
        return _load_model()
    with _model_lock:
        return _load_model()


# Tests clear the cache to force a reload (e.g. to point at a fixture ONNX file).
get_model.cache_clear = _load_model.cache_clear
get_model.cache_info = _load_model.cache_info


def _mean_pool_and_normalize(
    token_embeddings: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """(batch, seq, 384) token embeddings + (batch, seq) mask -> (batch, 384)
    unit vectors. Attention-mask-weighted mean, THEN L2-normalise -- see the
    module docstring for why the order and the padding-exclusion both matter.
    """
    mask = attention_mask.astype(np.float32)[:, :, None]          # (batch, seq, 1)
    summed = (token_embeddings * mask).sum(axis=1)                # (batch, 384)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)    # (batch, 1)
    mean_pooled = summed / counts
    norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (mean_pooled / norms).astype(np.float32)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts into (n, 384) float32 unit vectors."""
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    session, tokenizer = get_model()
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encodings = tokenizer.encode_batch(batch)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array(
            [e.type_ids for e in encodings], dtype=np.int64
        )
        (token_embeddings,) = session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        chunks.append(_mean_pool_and_normalize(token_embeddings, attention_mask))
    return np.concatenate(chunks, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Equals the dot product here because vectors are unit-length,
    but the division is kept so this stays correct if a caller passes raw vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
