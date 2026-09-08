def test_vocabulary_is_600_words_across_6_categories_with_no_duplicates():
    from data.vocabulary import VOCABULARY
    assert len(VOCABULARY) == 6
    words = [w for ws in VOCABULARY.values() for w in ws]
    assert len(words) == 600
    assert len(set(words)) == 600, "duplicate word across categories"
    assert all(len(ws) == 100 for ws in VOCABULARY.values())


def test_ambiguous_words_are_not_in_the_training_vocabulary():
    """They exist to be misclassified in the UI. If one leaks into training it stops
    demonstrating anything."""
    from data.vocabulary import VOCABULARY, AMBIGUOUS
    words = {w for ws in VOCABULARY.values() for w in ws}
    assert not (words & set(AMBIGUOUS)), f"leaked: {words & set(AMBIGUOUS)}"


def test_neighbours_of_cat_are_mostly_animals(vocab):
    """Not 'returns 5 results' -- that passes against a function returning the first 5
    words alphabetically. Assert the SEMANTICS."""
    hits = vocab.neighbours("cat", k=5)
    cats = [h for h in hits if h["category"] == "animal"]
    assert len(cats) >= 3, f"expected mostly animals, got {hits}"


def test_neighbours_excludes_the_query_word_itself(vocab):
    """'cat' is its own nearest neighbour at cosine 1.0. Returning it wastes a slot
    and looks like a bug in the UI."""
    hits = vocab.neighbours("cat", k=5)
    assert all(h["word"] != "cat" for h in hits)
