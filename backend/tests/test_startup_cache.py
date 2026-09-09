"""THE GATE for the startup-cache change (fit-at-build, load-at-boot).

`app.train.build_models` now LOADS a classifier + supervised/unsupervised
figures that were fit once, at container build time, by
`scripts/fit_models.py` calling `fit_models_from_scratch` -- instead of
re-fitting LogisticRegression, KMeans and PCA (on top of re-embedding all 600
words) on every process start. This is the proof that the substitution
changes nothing a caller can observe: a classifier loaded via `load_models`
must produce `predict_proba` output IDENTICAL (to within 1e-9) to the
freshly-fitted classifier it was saved from, on every one of the 600
vocabulary words plus the ambiguous and subword-leakage lists. Argmax alone
is not enough -- a predicted label can match while the underlying
probability distribution has moved, and every downstream number (the UI's
confidence percentage, the map) reads that distribution.

Plays the same role for this change that `tests/test_onnx_equivalence.py`
plays for the ONNX migration: not a smoke test that "loading returns
something", but a numeric equivalence proof with teeth.

The mutation test at the bottom proves those teeth are real. A test that
only compares loaded-to-loaded (both derived from the same save/load
round-trip, which is exactly what the equivalence test above does) could in
principle pass no matter what -- if `predict_proba` ignored the model
entirely and returned a constant, loaded-vs-fresh would still match itself.
So the mutation test perturbs ONE coefficient on the loaded copy and reruns
the identical comparison: the diff must jump from ~0 to something the 1e-9
gate would fail on. If it doesn't, the comparison above is not sensing
anything and this whole file is theater.
"""

import copy

import numpy as np
import pytest

from app.config import settings
from app.embed import embed_texts
from app.train import _vocab_embeddings, fit_models_from_scratch, load_models, save_models
from data.vocabulary import AMBIGUOUS, SUBWORD_LEAKAGE

# All 600 vocabulary words, plus the ambiguous list (excluded from training,
# so worth checking the loaded model still handles them the same way) and
# both subword-leakage pairs (plain word + compound).
_VOCAB_WORDS, _, _ = _vocab_embeddings()
WORDS = sorted(
    set(_VOCAB_WORDS)
    | set(AMBIGUOUS)
    | {plain for plain, _ in SUBWORD_LEAKAGE}
    | {compound for _, compound in SUBWORD_LEAKAGE}
)


def _max_predict_proba_diff(clf_a, clf_b, texts: list[str]) -> float:
    """Max absolute per-class probability difference between two fitted
    classifiers over the same texts -- the metric the acceptance test is
    built on, not just whether the two argmax predictions agree."""
    X = embed_texts(texts)
    pa = clf_a.predict_proba(X)
    pb = clf_b.predict_proba(X)
    return float(np.max(np.abs(pa - pb)))


@pytest.fixture(scope="module")
def fresh_models():
    """The ground truth: a real, from-scratch fit -- see
    `fit_models_from_scratch`'s docstring, which names this test module as
    its gate."""
    return fit_models_from_scratch(seed=settings.seed)


@pytest.fixture(scope="module")
def loaded_models(fresh_models, tmp_path_factory):
    """`fresh_models`, round-tripped through the exact save/load pair the
    real build (`scripts/fit_models.py`) and boot (`app.train.build_models`)
    use -- a scratch path, not `settings.trained_models_path`, so this test
    never depends on (or clobbers) a locally-built artifact."""
    path = tmp_path_factory.mktemp("startup-cache") / "trained_models.joblib"
    save_models(fresh_models, path)
    return load_models(path)


def test_the_word_list_covers_all_600_plus_ambiguous_and_subword_leakage():
    assert len(WORDS) >= 600
    assert set(_VOCAB_WORDS) <= set(WORDS)
    assert set(AMBIGUOUS) <= set(WORDS)
    assert {"cat", "catboat", "ram", "ramrod"} <= set(WORDS)


def test_loaded_classifier_matches_a_fresh_fit_to_1e9(fresh_models, loaded_models):
    diff = _max_predict_proba_diff(fresh_models.classifier, loaded_models.classifier, WORDS)
    assert diff < 1e-9, (
        f"max predict_proba diff {diff} across {len(WORDS)} words -- the loaded "
        "artifact has diverged from a fresh fit; do not relax this tolerance, "
        "find and fix the divergence"
    )


def test_loaded_supervised_and_unsupervised_figures_match_exactly(fresh_models, loaded_models):
    """The classifier is not the only thing loaded -- the scored macro-F1/ARI
    dicts travel in the same artifact and must round-trip byte-for-byte,
    since they are what the README's published numbers and the UI's
    'macro_f1 next to its baseline' come from."""
    assert loaded_models.seed == fresh_models.seed
    assert loaded_models.supervised == fresh_models.supervised
    assert loaded_models.unsupervised == fresh_models.unsupervised


def test_mutation_a_perturbed_loaded_coefficient_is_caught(fresh_models, loaded_models):
    """Standing lesson: would this test still pass if the loader did nothing
    useful? Prove it can't by mutating the loaded classifier and rerunning
    the exact same comparison the equivalence test above uses."""
    mutated = copy.deepcopy(loaded_models.classifier)
    mutated.coef_[0, 0] += 0.01  # a tiny perturbation, not a gross corruption

    diff = _max_predict_proba_diff(fresh_models.classifier, mutated, WORDS)
    assert diff > 1e-6, (
        f"perturbing one loaded coefficient by 0.01 only moved max predict_proba "
        f"diff to {diff} -- the equivalence comparison above is not sensitive "
        "enough to catch a real divergence, which means it proves nothing"
    )
