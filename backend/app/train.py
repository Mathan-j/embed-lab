"""Supervised classifier and unsupervised clustering over the 600-word vocabulary,
both scored against a chance baseline.

Every number this module produces is meant to travel with the configuration it
was measured under: a seed, and (for the classifier) the baseline it beat. A
macro-F1 or an ARI reported alone is not evidence of anything -- see the
docstrings on `train_supervised` and `train_unsupervised` for what each
baseline means and why.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.config import settings
from app.embed import embed_texts
from data.vocabulary import VOCABULARY


@lru_cache(maxsize=1)
def _vocab_embeddings() -> tuple[list[str], list[str], np.ndarray]:
    """The 600 vocabulary words, their categories, and their embeddings, in one
    fixed order (category, then word, matching `app.vocab.VocabIndex`).

    Cached because every function below needs this same (600, 384) matrix, and
    re-embedding 600 fixed words on every call would be pure waste -- the
    vocabulary does not change at runtime.
    """
    words: list[str] = []
    categories: list[str] = []
    for category, ws in VOCABULARY.items():
        for w in ws:
            words.append(w)
            categories.append(category)
    X = embed_texts(words)
    return words, categories, X


@dataclass(frozen=True)
class Split:
    X: np.ndarray
    y: list[str]
    words: list[str]


def make_split(seed: int, shuffle_labels: bool = False) -> tuple[Split, Split]:
    """A 70/30 stratified train/test split of the 600-word vocabulary.

    `stratify=y` matters here specifically because there are 6 categories: a
    plain random split can starve a small category out of the test set
    entirely, and macro-F1 would then silently average over 5 classes while
    claiming 6. With 100 words per category this is unlikely but not
    impossible, so it is enforced rather than hoped for.

    `shuffle_labels` is not used by the live app. It exists so a caller (see
    `train_supervised`) can prove the macro-F1 > baseline threshold test is
    not vacuous: fit the exact same pipeline on labels that have been randomly
    permuted, breaking the true X<->y relationship, and confirm the score
    collapses to the chance baseline. If it did not collapse, something
    upstream of the split (or the metric) would be leaking information.
    """
    words, categories, X = _vocab_embeddings()
    y = list(categories)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y = list(rng.permutation(y))

    X_train, X_test, y_train, y_test, words_train, words_test = train_test_split(
        X, y, words, test_size=0.3, stratify=y, random_state=seed
    )
    return (
        Split(X=X_train, y=list(y_train), words=list(words_train)),
        Split(X=X_test, y=list(y_test), words=list(words_test)),
    )


@lru_cache(maxsize=None)
def train_supervised(seed: int, shuffle_labels: bool = False) -> dict:
    """Fit LogisticRegression on a held-out 70/30 stratified split and score it.

    `baseline_macro_f1` is 1 / (number of categories), the expected macro-F1 of
    guessing uniformly at random over balanced classes. It is computed from
    `len(labels)` rather than hardcoded as 0.167 so it stays correct if the
    vocabulary ever stops being 6 categories -- a macro-F1 quoted without this
    number means nothing on its own.

    Cached per (seed, shuffle_labels) since training is deterministic given
    those inputs and repeated callers (e.g. every `/api/classify` request)
    should not retrain from scratch.
    """
    tr, te = make_split(seed, shuffle_labels=shuffle_labels)
    labels = sorted(set(tr.y) | set(te.y))

    clf = LogisticRegression(max_iter=1000)
    clf.fit(tr.X, tr.y)
    y_pred = clf.predict(te.X)

    report = classification_report(
        te.y, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]
    baseline_macro_f1 = 1.0 / len(labels)
    cm = confusion_matrix(te.y, y_pred, labels=labels).tolist()
    per_class = {
        label: {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        }
        for label in labels
    }

    return {
        "seed": seed,
        "shuffle_labels": shuffle_labels,
        "labels": labels,
        "macro_f1": macro_f1,
        "baseline_macro_f1": baseline_macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_train": len(tr.y),
        "n_test": len(te.y),
    }


@lru_cache(maxsize=None)
def train_unsupervised(seed: int) -> dict:
    """KMeans over the full 384-dim embeddings, scored against the true
    categories with `adjusted_rand_score`, plus a 2-D PCA projection for
    drawing.

    ARI is already chance-corrected: 0.0 IS what random cluster assignment
    scores, and 1.0 is a perfect match to the true categories. There is no
    separate baseline to compute or report here, unlike macro-F1 -- that is
    a deliberate difference between the two metrics, not an omission.

    PCA, not UMAP: PCA is a linear, deterministic projection whose two axes
    are directions of maximum variance and mean something reproducible; UMAP
    would likely look tidier but is nonlinear, sensitive to its own
    hyperparameters, and would add a dependency to justify for one scatter
    plot. That tradeoff is deliberate -- do not swap it in without knowing
    what is being traded away.

    KMeans clusters `X`, the full 384-dim vectors -- never `coords`, the 2-D
    projection. The projection is for the picture only; clustering it would
    be clustering an artifact of the drawing rather than the actual
    embedding geometry.
    """
    words, categories, X = _vocab_embeddings()

    kmeans = KMeans(n_clusters=6, n_init=10, random_state=seed)
    clusters = kmeans.fit_predict(X)
    ari = adjusted_rand_score(categories, clusters)

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)

    return {
        "seed": seed,
        "n_clusters": 6,
        "ari": float(ari),
        "words": words,
        "categories": categories,
        "clusters": clusters.tolist(),
        "coords": coords.tolist(),
    }


@dataclass(frozen=True)
class TrainedModels:
    """Everything built once at startup: the live classifier plus the two
    scored figures (`supervised`, `unsupervised`) that justify trusting it.
    """

    seed: int
    classifier: LogisticRegression
    supervised: dict
    unsupervised: dict


def build_models(seed: int = settings.seed) -> TrainedModels:
    """Builds the live classifier and both scored figures at one seed.

    The classifier used by `/api/classify` is fit on all 600 words, not just
    the 70% training split -- `train_supervised` exists to produce an honest
    held-out score, but the deployed model should see every word it can. The
    two are trained independently (a fresh `LogisticRegression` here) so the
    reported macro-F1 always reflects a model that never saw its own test
    set, even though the live classifier is a different, more-informed fit.
    """
    words, categories, X = _vocab_embeddings()
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X, categories)

    return TrainedModels(
        seed=seed,
        classifier=classifier,
        supervised=train_supervised(seed),
        unsupervised=train_unsupervised(seed),
    )


@lru_cache(maxsize=None)
def _default_models() -> TrainedModels:
    """Lazily-built models at the default seed, for callers (tests, and
    `classify()` when no explicit models are supplied) that do not go through
    the app lifespan. The live app instead builds one `TrainedModels` in
    `app.main`'s lifespan and reuses it for every request.
    """
    return build_models(settings.seed)


def classify(text: str, models: TrainedModels | None = None) -> dict:
    """Classify one piece of text against the live classifier.

    Returns the predicted category and the full probability distribution over
    all six, which always sums to 1.0 since `predict_proba` on a fitted
    multinomial LogisticRegression is a normalized distribution by
    construction.
    """
    if models is None:
        models = _default_models()

    vector = embed_texts([text])[0].reshape(1, -1)
    probs = models.classifier.predict_proba(vector)[0]
    classes = models.classifier.classes_

    all_probs = sorted(
        (
            {"category": str(category), "probability": float(p)}
            for category, p in zip(classes, probs)
        ),
        key=lambda item: -item["probability"],
    )
    predicted = str(classes[int(np.argmax(probs))])

    return {"predicted": predicted, "all": all_probs, "seed": models.seed}
