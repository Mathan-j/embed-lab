import pytest

from data.vocabulary import VOCABULARY

# The first word appended when VOCABULARY is flattened category-by-category --
# see `app.vocab.VocabIndex.__init__` and `app.train._vocab_embeddings`, which
# both iterate the dict in the same order. Read from the source of truth
# rather than hardcoded, so this stays right if the vocabulary is reordered.
VOCAB_FIRST_WORD = next(iter(VOCABULARY.values()))[0]


def test_the_split_is_stratified_so_every_category_appears_in_both_halves():
    """With 6 categories and a random split, a small category can vanish from the test
    set entirely and macro-F1 then averages over 5 classes while claiming 6."""
    from app.train import make_split
    tr, te = make_split(seed=0)
    assert len(set(tr.y)) == 6 and len(set(te.y)) == 6
    assert len(tr.y) == 420 and len(te.y) == 180        # 70/30 of 600


def test_macro_f1_beats_the_chance_baseline_by_a_wide_margin():
    """Six balanced categories -> chance is 1/6. A macro-F1 that does not clear it by
    a lot means the embeddings are not carrying category information at all."""
    from app.train import train_supervised
    r = train_supervised(seed=0)
    assert r["baseline_macro_f1"] == pytest.approx(1 / 6, abs=0.02)
    assert r["macro_f1"] > 0.7, f"got {r['macro_f1']}"


def test_training_is_deterministic_under_a_seed():
    """Two runs at one seed must give identical macro-F1, or the number moves between
    restarts and nobody can tell a regression from reshuffling."""
    from app.train import train_supervised
    assert train_supervised(seed=0)["macro_f1"] == train_supervised(seed=0)["macro_f1"]


def test_shuffling_labels_collapses_macro_f1_to_the_chance_baseline():
    """Mutation proof for the threshold test above. A `macro_f1 > 0.7` assertion
    passes against a wide range of implementations, including broken ones that
    leak the answer some other way. Shuffling the labels before training breaks
    the true X<->y relationship on purpose; if macro-F1 stayed high anyway, that
    would mean something upstream is leaking the label into training rather than
    the embeddings genuinely carrying category information."""
    from app.train import train_supervised
    real = train_supervised(seed=0)
    shuffled = train_supervised(seed=0, shuffle_labels=True)
    assert shuffled["macro_f1"] < real["macro_f1"] - 0.3, (
        f"shuffled-label macro_f1={shuffled['macro_f1']} did not collapse "
        f"relative to real macro_f1={real['macro_f1']} -- something is leaking"
    )
    assert shuffled["macro_f1"] == pytest.approx(shuffled["baseline_macro_f1"], abs=0.12)


def test_clustering_ari_beats_random_assignment():
    """ARI is already chance-corrected: 0.0 IS random. So the assertion is simply that
    it is meaningfully above 0 -- but state that in the docstring, because a reader who
    does not know ARI will otherwise wonder where the baseline went."""
    from app.train import train_unsupervised
    r = train_unsupervised(seed=0)
    assert r["n_clusters"] == 6
    assert r["ari"] > 0.3, f"got {r['ari']}"


def test_the_2d_map_has_one_point_per_word_and_preserves_order():
    """Row i of coords must be word i. A shuffle here draws a plausible-looking map of
    the wrong data, and nothing else in the system would notice."""
    from app.train import train_unsupervised
    r = train_unsupervised(seed=0)
    assert len(r["coords"]) == 600 and len(r["coords"][0]) == 2
    assert r["words"][0] == VOCAB_FIRST_WORD


def test_coords_rows_actually_correspond_to_their_words_not_just_row_zero():
    """The plan's ordering test above only pins `words[0]`, which would still
    pass even if every `coords` row past index 0 were shuffled relative to
    `words` -- confirmed by deliberately shuffling a copy of `coords` and
    re-running that test's exact assertions against it: they still pass. So
    this test independently recomputes PCA on the same embeddings and seed
    and checks several interior rows (not just row 0) line up, which a
    words<->coords misalignment would break."""
    from sklearn.decomposition import PCA
    from app.train import train_unsupervised, _vocab_embeddings
    words, categories, X = _vocab_embeddings()
    r = train_unsupervised(seed=0)
    assert r["words"] == words

    independent_coords = PCA(n_components=2, random_state=0).fit_transform(X)
    for i in (0, 1, 150, 300, 450, 599):
        assert r["coords"][i] == pytest.approx(independent_coords[i].tolist(), abs=1e-6)


def test_classify_returns_a_distribution_that_sums_to_one():
    from app.train import classify
    out = classify("cat")
    assert abs(sum(p["probability"] for p in out["all"]) - 1.0) < 1e-6
    assert out["predicted"] == "animal"


def test_an_ambiguous_word_is_classified_confidently_and_that_is_expected():
    """'orange' is a colour and a food. The model must pick one; the point is that the
    LABEL is underdetermined, not that the model is broken. Pinned so the UI's
    'try these' row keeps working."""
    from app.train import classify
    from data.vocabulary import AMBIGUOUS
    out = classify("orange")
    assert out["predicted"] in ("colour", "food")
    assert "orange" in AMBIGUOUS
