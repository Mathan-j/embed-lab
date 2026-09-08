def test_embedding_is_384_dims_and_unit_length():
    import numpy as np
    from app.embed import embed_texts
    v = embed_texts(["cat"])[0]
    assert v.shape == (384,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_cat_is_closer_to_dog_than_to_car():
    """The whole point of the project in one assertion: spelling is not meaning.
    cat/car differ by one letter; cat/dog share none."""
    from app.embed import embed_texts, cosine
    cat, dog, car = embed_texts(["cat", "dog", "car"])
    assert cosine(cat, dog) > cosine(cat, car)
