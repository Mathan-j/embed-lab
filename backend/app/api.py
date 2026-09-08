"""The router: /api/tokenize, /api/embed, /api/neighbours, /api/classify, /api/map.

User-fixable errors (empty text, k out of range) raise StageError, which
app.main's exception handler turns into a 409 with a runnable hint -- never a
stack trace.
"""

import numpy as np
from fastapi import APIRouter, Request

from app.embed import embed_texts
from app.errors import StageError
from app.tokenize import tokenize_detail
from app.train import TrainedModels, classify as classify_text
from data.vocabulary import AMBIGUOUS, SUBWORD_LEAKAGE

router = APIRouter(prefix="/api")

MAX_K = 100


def _require_text(text: str) -> str:
    if not text or not text.strip():
        raise StageError(
            detail="text must not be empty.",
            hint="Pass a word or short phrase, e.g. /api/tokenize?text=cat",
        )
    return text


@router.get("/tokenize")
def tokenize(text: str = "") -> dict:
    text = _require_text(text)
    return tokenize_detail(text)


@router.get("/embed")
def embed(text: str = "") -> dict:
    text = _require_text(text)
    vector = embed_texts([text])[0]
    return {
        "dims": int(vector.shape[0]),
        "norm": float(np.linalg.norm(vector)),
        "vector": vector.tolist(),
    }


@router.get("/neighbours")
def neighbours(request: Request, text: str = "", k: int = 5) -> dict:
    text = _require_text(text)
    if not (1 <= k <= MAX_K):
        raise StageError(
            detail=f"k must be between 1 and {MAX_K}, got {k}.",
            hint="Pass a k in range, e.g. /api/neighbours?text=cat&k=5",
        )
    hits = request.app.state.vocab_index.neighbours(text, k=k)
    return {"hits": hits}


@router.get("/classify")
def classify(request: Request, text: str = "") -> dict:
    text = _require_text(text)
    models: TrainedModels = request.app.state.trained_models
    result = classify_text(text, models=models)
    return {
        **result,
        # Every figure travels with its baseline: the classifier's held-out
        # macro-F1 next to the chance rate it has to beat, so the UI can
        # never show one without the other.
        "macro_f1": models.supervised["macro_f1"],
        "baseline_macro_f1": models.supervised["baseline_macro_f1"],
    }


@router.get("/demos")
def demos() -> dict:
    """The two 'try these' rows the UI needs, served from the same list the
    classifier and training were built to exclude -- never hardcoded in the
    client, so the demo stays honest if the vocabulary changes.

    `ambiguous`: words that genuinely belong to two categories (e.g. `orange`
    is colour/food). The classifier picks one confidently; that is expected,
    the label is underdetermined, not the model.

    `subword_leakage`: (plain, compound) pairs where the compound tokenizes
    with the plain word as a wordpiece (e.g. `catboat` -> `cat` + `##boat`),
    pulling its embedding toward the plain word's even though the meanings
    are unrelated -- a measurable subword-driven wobble, not a meaning mixup.
    """
    return {
        "ambiguous": AMBIGUOUS,
        "subword_leakage": [
            {"plain": plain, "compound": compound}
            for plain, compound in SUBWORD_LEAKAGE
        ],
    }


@router.get("/map")
def map_(request: Request) -> dict:
    models: TrainedModels = request.app.state.trained_models
    u = models.unsupervised
    return {
        "words": u["words"],
        "categories": u["categories"],
        "clusters": u["clusters"],
        "coords": u["coords"],
        "n_clusters": u["n_clusters"],
        "ari": u["ari"],
        "seed": models.seed,
    }
