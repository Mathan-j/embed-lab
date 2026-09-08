"""Tokenize with the exact WordPiece tokenizer MiniLM uses, at one pinned revision.

Ported from triage-lab's `s1_tokenize.py`, keeping the tokenizer singleton and its
padding/truncation guard -- the only two things this project needs from that stage.
There is no chunking here: embed-lab embeds single words and short phrases, not
tickets split across a wordpiece budget.
"""

from functools import lru_cache

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Encoding, Tokenizer

from app.config import settings
from app.errors import StageError


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """The WordPiece tokenizer MiniLM actually uses, at one pinned revision.

    Pinned because the counts are load-bearing: an unpinned
    `Tokenizer.from_pretrained(model)` resolves to whatever the Hub serves that
    day, and a different tokenizer.json moves every wordpiece count and every
    tokenize-card result shown on screen.

    Downloaded through hf_hub_download rather than Tokenizer.from_pretrained
    because the latter takes no cache directory, and settings.model_cache_dir is
    documented as where model files live. Fetching the file explicitly honours
    both the revision and the cache location, and is what from_pretrained does
    internally anyway.

    tokenizer.json ships with padding=128 and truncation=128 enabled. Both MUST be
    turned off: with them on, every count returns 128 and text past 128 pieces is
    cut with no error, which would silently and invisibly cap every result.
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
        # Cold cache, no network: hf_hub_download cannot fall back to a remote
        # fetch nor find a local copy. This becomes a 409 with a hint instead of
        # an unhandled 500 with a stack trace.
        raise StageError(
            detail="The tokenizer is not cached locally and there is no network "
                   "connection to download it.",
            hint="Run once with a network connection so the tokenizer caches, "
                 "then retry.",
        ) from err
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def _encode(text: str) -> Encoding:
    """Tokenize once, refusing to run against a re-armed tokenizer.

    Everything that counts or inspects wordpieces goes through here, so the
    padding/truncation guard cannot be bypassed by a second call site.
    """
    tokenizer = get_tokenizer()
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise RuntimeError(
            "The shared tokenizer has had padding or truncation re-enabled. "
            "Counts would be capped at that length and silently wrong. Do not "
            "call enable_padding()/enable_truncation() on the object returned by "
            "get_tokenizer()."
        )
    return tokenizer.encode(text, add_special_tokens=False)


def wordpieces(text: str) -> list[str]:
    if not text:
        return []
    return _encode(text).tokens


def count_wordpieces(text: str) -> int:
    return len(wordpieces(text))


def tokenize_detail(text: str) -> dict:
    """The full tokenize-card payload: the pieces, their ids, and the count."""
    if not text:
        return {"pieces": [], "ids": [], "count": 0}
    encoding = _encode(text)
    return {
        "pieces": encoding.tokens,
        "ids": list(encoding.ids),
        "count": len(encoding.tokens),
    }
