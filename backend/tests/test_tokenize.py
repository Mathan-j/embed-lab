def test_tokenizer_has_padding_and_truncation_off():
    """The shipped tokenizer.json enables both at 128. With them on, every count
    returns 128 and long input is silently cut. This is the trap, pinned."""
    from app.tokenize import get_tokenizer, count_wordpieces
    assert count_wordpieces("cat") == 1
    long_text = "cat " * 300
    assert count_wordpieces(long_text) > 200, "truncation is still on"


def test_a_single_common_word_is_one_wordpiece():
    from app.tokenize import tokenize_detail
    d = tokenize_detail("cat")
    assert d["pieces"] == ["cat"] and len(d["ids"]) == 1


def test_a_rarer_word_splits_into_subwords():
    """The interesting half of the tokenize card. If this word stops splitting after a
    model change, the card gets boring and we want to know.

    FINDING: the plan's own example word was "crimson" (expecting `cr ##ims ##on`),
    but against the actually pinned tokenizer (bert-base-uncased vocab, revision
    1110a24, 30522 tokens) "crimson" is a single whole wordpiece -- verified with
    get_tokenizer() directly. Swapped in "vermilion", which does split
    (['ve', '##rmi', '##lion']), and kept everything else about the test verbatim.
    """
    from app.tokenize import tokenize_detail
    d = tokenize_detail("vermilion")
    assert len(d["pieces"]) > 1
    assert "".join(p.lstrip("#") for p in d["pieces"]) == "vermilion"
