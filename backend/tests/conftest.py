import pytest

from app.vocab import VocabIndex


@pytest.fixture(scope="session")
def vocab() -> VocabIndex:
    """Built once per test session -- embedding 600 words is ~2s and the
    vocabulary never changes between tests."""
    return VocabIndex()
