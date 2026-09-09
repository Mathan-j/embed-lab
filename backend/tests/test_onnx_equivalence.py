"""THE GATE for the torch -> ONNX Runtime migration.

Mean pooling is attention-mask weighted: sum token embeddings where mask == 1,
divide by the mask sum, THEN L2-normalise. Getting any part of that subtly
wrong -- averaging over padding, or normalising before pooling -- produces
embeddings that look completely plausible (still unit-length, still roughly
in the right neighbourhood) and are quietly wrong. Every downstream number
(macro-F1, ARI, every cosine similarity in the README) would shift and
nothing would raise an error.

So this is not a smoke test that ONNX "returns something" -- it is a
numeric equivalence proof against the retired torch/sentence-transformers
path, over a vocabulary broad enough that a padding or ordering bug would
have somewhere to hide: single-wordpiece words, a multi-wordpiece word, the
ambiguous list, and both subword-leakage compounds (which only exist to
create padding of varying length against their plain-word neighbours in the
same batch).

Marked `dev`: it needs torch + sentence-transformers, which are dev-only
dependencies (see pyproject.toml). It does not run against the production
(torch-free) environment, and it is not part of the 30-test baseline suite
that must pass everywhere -- it is the one-time (well, every model change)
proof that the numpy pooling reimplementation in app.embed matches torch.
"""

import numpy as np
import pytest

from app.config import settings
from data.vocabulary import AMBIGUOUS, SUBWORD_LEAKAGE, VOCABULARY

pytestmark = pytest.mark.dev

# >= 30 words spanning: single-wordpiece, multi-wordpiece, the ambiguous list,
# and both subword-leakage compounds plus their plain counterparts and an
# unrelated comparison word for each (so a batch has real length variation).
WORDS = sorted(
    {
        "cat", "dog", "car", "truck",              # single-wordpiece, common
        "vermilion",                                 # multi-wordpiece (['ve', '##rmi', '##lion'])
        "catboat", "boat",                          # subword-leakage pair + its true meaning
        "ramrod", "rod", "goat", "cheese",          # subword-leakage pair + comparison words
        *AMBIGUOUS,                                  # orange, mint, turkey, jaguar, plum,
                                                       # salmon, ginger, date, olive
        *[ws[i] for ws in VOCABULARY.values() for i in (0, 1, 2)],  # 3 per category, 6 cats
    }
)


def _torch_embed(texts: list[str]) -> np.ndarray:
    """The retired path: SentenceTransformer at the exact pinned revision."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        settings.embed_model,
        revision=settings.embed_model_revision,
        cache_folder=str(settings.model_cache_dir),
        device="cpu",
    )
    return model.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)


@pytest.fixture(scope="module")
def torch_vectors() -> np.ndarray:
    return _torch_embed(WORDS)


@pytest.fixture(scope="module")
def onnx_vectors() -> np.ndarray:
    from app.embed import embed_texts

    return embed_texts(WORDS)


def test_at_least_30_words_span_the_required_categories():
    assert len(WORDS) >= 30
    assert "vermilion" in WORDS
    assert {"cat", "catboat", "ramrod"} <= set(WORDS)
    assert set(AMBIGUOUS) <= set(WORDS)


def test_onnx_matches_torch_within_float_tolerance(torch_vectors, onnx_vectors):
    assert onnx_vectors.shape == torch_vectors.shape

    max_abs_diff = float(np.max(np.abs(onnx_vectors - torch_vectors)))
    assert max_abs_diff < 1e-5, (
        f"max abs diff {max_abs_diff} >= 1e-5 -- the ONNX pooling/normalisation "
        "reimplementation has diverged from torch. Do not adjust this "
        "tolerance to make the test pass; find and fix the divergence."
    )

    cos_per_word = np.sum(onnx_vectors * torch_vectors, axis=1) / (
        np.linalg.norm(onnx_vectors, axis=1) * np.linalg.norm(torch_vectors, axis=1)
    )
    min_cos = float(np.min(cos_per_word))
    assert min_cos > 0.9999, (
        f"min cosine {min_cos} <= 0.9999 for at least one word in {WORDS}"
    )


def test_onnx_vectors_are_still_unit_normalised(onnx_vectors):
    """A sanity check independent of the torch comparison: if pooling ran
    before normalising (correct) rather than after (also produces unit
    vectors) this still passes -- the real order check is the equivalence
    test above. This just guards against a completely un-normalised output."""
    norms = np.linalg.norm(onnx_vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
