# Embed Lab — Phase 2: images and cross-modal search

> **For agentic workers:** execute with superpowers:subagent-driven-development, one task
> at a time. Phase 1 (`docs/plan.md`, Tasks 1–3) must be complete first.

**Goal:** Type a word and get back photographs. Pick a photograph and get back the words
nearest it. Both in one shared embedding space, with a classifier and a clustering over
the images — every number reported against its chance baseline, as in Phase 1.

**Tech added:** `open_clip` via sentence-transformers (`clip-ViT-B-32`), `torchvision`,
`Pillow`, Imagenette (10 real-photo classes, 342 MB, 320px).

---

## The decision that shapes everything: two spaces, never mixed

MiniLM emits **384** dimensions. CLIP emits **512**. They are *different vector spaces
trained by different objectives*, and a cosine between one model's vector and another's
is meaningless — not "less accurate", meaningless. Numerically it will happily produce a
number, which is exactly why this needs guarding rather than documenting.

So the app carries **two** spaces, and the UI must always say which one it is showing:

| Tab | Space | Text encoder | Why |
|---|---|---|---|
| **Words** (Phase 1) | MiniLM, 384-d | MiniLM | best pure-text similarity |
| **Images** | CLIP, 512-d | — | image-to-image search |
| **Cross-modal** | CLIP, 512-d | **CLIP's** text encoder | text and images comparable |

**The same word therefore has two different vectors** depending on the tab. That is not a
bug and it is the single most valuable thing this phase teaches: *an embedding is only
meaningful relative to the model that produced it.* Label it on screen; do not hide it.

**Constraint, enforced in code:** every vector carries its space, and `cosine()` refuses
to compare vectors of different dimension or different provenance. A `StageError`, not a
silent number.

**The honest cost:** CLIP's text encoder is worse at pure text similarity than MiniLM —
it is trained to match captions to pictures, caps at 77 tokens, and its word-only
neighbours are noticeably weaker. Phase 1's Words tab keeps MiniLM for that reason. Do
not "simplify" by moving everything to CLIP.

## The corpus, and its honest mismatch with Phase 1

Imagenette's 10 classes: `tench` (a fish), `English springer` (a dog), `cassette player`,
`chain saw`, `church`, `French horn`, `garbage truck`, `gas pump`, `golf ball`,
`parachute`.

Phase 1's 600 words are 6 categories: animal, food, vehicle, emotion, colour, job.

**These label sets do not correspond, and that is kept rather than fixed.** Only a few
words have any image counterpart (`dog`, `truck`, `fish`); most (`sadness`, `plumber`,
`magenta`) have none, and most image classes (`parachute`, `gas pump`) are in no word
category. A cross-modal search for `sadness` will return whatever is least-wrong, and it
will look arbitrary — because it is. **Show that honestly**: it is the difference between
a shared space and a shared *vocabulary*, and smoothing it over would teach that
cross-modal retrieval is more magic than it is.

Provide a `CROSS_MODAL_PROBES` list of ~8 words that DO have counterparts (`dog`, `fish`,
`church`, `truck`, `ball`, `horn`, `saw`, `parachute`) so the demo has reliable examples,
alongside free text entry that is allowed to disappoint.

---

## Task 4: CLIP, the image corpus, cached embeddings, image serving

**Files:** create `backend/app/clip_embed.py`, `backend/app/images.py`,
`backend/scripts/build_image_index.py`, `backend/tests/test_clip_embed.py`,
`backend/tests/test_images.py`; modify `app/api.py`, `app/main.py`, `pyproject.toml`.

- [ ] **Step 1: dependencies**

```bash
cd D:/Projects/embed-lab/backend
uv add torchvision pillow
```

Check whether `torch` resolves to a CUDA build. The sibling project found `torch` pulls
**19 CUDA/triton packages** gated `sys_platform == 'linux'` — invisible on Windows,
multi-gigabyte on Linux or CI. If they appear, pin the CPU index as `triage-lab` does.
Report what you find either way.

- [ ] **Step 2: extract the corpus, and sample it deliberately**

`data/images/imagenette2-320.tgz` is already downloaded (342 MB). Extract it. Use the
**train** split only; ignore `val` (there is no held-out need beyond our own split).

Sample **100 images per class = 1000 images**, chosen by sorted filename with a fixed
seed so the set is reproducible. Record the count per class — Imagenette is roughly but
not exactly balanced, and the chance baseline depends on the real distribution, not on
an assumed 1/10.

- [ ] **Step 3: the failing tests**

```python
def test_clip_embeds_an_image_to_512_unit_dims():
    import numpy as np
    from app.clip_embed import embed_images
    v = embed_images([SOME_JPEG_PATH])[0]
    assert v.shape == (512,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_clip_text_and_image_land_in_the_same_space():
    """The whole premise. A photo of a dog must sit closer to the WORD 'dog' than to
    the word 'church' -- with no labels involved anywhere in the computation."""
    from app.clip_embed import embed_images, embed_texts_clip, cosine_clip
    img = embed_images([A_SPRINGER_SPANIEL_JPEG])[0]
    dog, church = embed_texts_clip(["a photo of a dog", "a photo of a church"])
    assert cosine_clip(img, dog) > cosine_clip(img, church)


def test_comparing_a_minilm_vector_to_a_clip_vector_is_refused():
    """384 vs 512 would raise on shape anyway; the point is that a same-dimension
    mismatch must ALSO be refused, so the guard is about provenance not luck."""
    from app.embed import embed_texts
    from app.clip_embed import cosine_clip
    from app.errors import SpaceMismatch
    with pytest.raises(SpaceMismatch):
        cosine_clip(embed_texts(["cat"])[0], embed_texts(["dog"])[0])


def test_image_embeddings_are_cached_not_recomputed(tmp_path):
    """1000 images take ~40s on CPU. Recomputing at every startup would make the app
    feel broken. Assert the second load reads the .npy and does not touch the model."""


def test_the_sampled_corpus_is_100_per_class_and_reproducible():
    from app.images import load_index
    idx = load_index()
    assert len(idx.paths) == 1000
    counts = Counter(idx.labels)
    assert set(counts) == set(IMAGENETTE_CLASSES) and all(c == 100 for c in counts.values())
    assert load_index().paths == idx.paths          # same order every time


def test_nearest_images_to_a_dog_photo_are_mostly_dogs(image_index):
    """Semantics, not 'returns 5 rows'. A 'returns 5' assertion passes against a
    function returning the first five files on disk."""
    hits = image_index.neighbours_of_image(A_SPRINGER_SPANIEL_JPEG, k=5)
    assert sum(1 for h in hits if h["label"] == "English springer") >= 3
```

- [ ] **Step 4: run, watch fail. Step 5: implement.**

`scripts/build_image_index.py` embeds the 1000 images once and writes
`data/images/index.npz` (paths, labels, `(1000, 512)` float32 matrix) plus 160px
thumbnails to `data/images/thumbs/`. **Report the real embedding rate with its
configuration** — e.g. "1000 images in 38s (26 img/sec, CPU, 4 torch threads)". A bare
rate is not acceptable; that rule cost the sibling project four review rounds.

`app/errors.py` gains `SpaceMismatch(StageError)`. Vectors are wrapped or tagged so
provenance is checkable — a plain `np.ndarray` cannot defend itself, so decide how
(a small dataclass, or a dimension-plus-registry check) and say why in the docstring.

- [ ] **Step 6: serving images**

`GET /api/image/{id}` returns the JPEG; `GET /api/thumb/{id}` returns the 160px version.
The grid uses thumbnails — 1000 full-size JPEGs will make the UI crawl. `id` is an index
into the corpus, validated; an out-of-range id is a 409 with a hint, not a 500, and
**never** a path that can escape `data/images/` (join and then verify the resolved path
is inside the corpus directory — a `../` in an id must not read arbitrary files).

- [ ] **Step 7: real check.** Start the backend on **8100**, curl
`/api/images/neighbours?id=<a dog>&k=5`, paste the output, and say whether the returned
images actually look like the query. Stop the server by the PID you started.

- [ ] **Step 8: commit.**

---

## Task 5: Cross-modal search, plus supervised and unsupervised over images

**Files:** create `backend/app/cross_modal.py`, `backend/app/train_images.py`, and their
tests; modify `app/api.py`.

- [ ] **Step 1: the failing tests**

```python
def test_text_to_image_search_finds_the_right_class_with_no_labels():
    """Zero-shot: 'a photo of a church' must return churches, and nothing in the
    computation touches a label. This is the headline claim of the whole phase."""
    hits = cross_modal.text_to_images("a photo of a church", k=5)
    assert sum(1 for h in hits if h["label"] == "church") >= 3


def test_image_to_text_returns_plausible_words():
    """The reverse direction, over Phase 1's 600 words embedded with CLIP's text
    encoder. A springer spaniel photo should surface 'dog' or another animal well
    above 'magenta'."""


def test_a_word_with_no_visual_counterpart_is_reported_as_weak_not_wrong():
    """'sadness' has no Imagenette class. The endpoint must still answer, but the top
    score should be markedly lower than for 'dog' -- and the response must carry that
    score so the UI can say 'nothing here matches well' instead of showing a confident
    arbitrary result."""
    strong = cross_modal.text_to_images("a photo of a dog", k=5)
    weak = cross_modal.text_to_images("sadness", k=5)
    assert weak[0]["score"] < strong[0]["score"] - 0.05


def test_image_classifier_beats_chance_by_a_wide_margin():
    """10 classes, ~100 each -> chance is ~1/10. COMPUTE the baseline from the real
    class counts; Imagenette is only roughly balanced and assuming 0.10 would be
    quoting a figure we did not measure."""
    r = train_images_supervised(seed=0)
    assert r["baseline_macro_f1"] == pytest.approx(0.10, abs=0.02)
    assert r["macro_f1"] > 0.85


def test_shuffled_labels_collapse_image_accuracy_to_chance():
    """The control that proves the threshold above is not vacuous. Phase 1's word
    classifier collapsed from 0.9226 to 0.1675 under this; the image one must too."""
    assert train_images_supervised(seed=0, shuffle_labels=True)["macro_f1"] < 0.2


def test_image_clustering_ari_beats_random():
    """ARI is already chance-corrected: 0.0 IS random, so there is no separate
    baseline to compute. State that, or a reader wonders where it went."""
    assert train_images_unsupervised(seed=0)["ari"] > 0.4
```

- [ ] **Step 2: run, watch fail. Step 3: implement. Step 4: pass.**

CLIP text embedding convention: prompt as `"a photo of a {word}"`, not the bare word —
that is what CLIP was trained on and it measurably improves retrieval. **Measure both
and report the difference** rather than taking my word for it; if the bare word wins on
this corpus, use it and say so.

`train_images_supervised` mirrors Phase 1: `LogisticRegression` on the frozen 512-d
embeddings, stratified 70/30, macro-F1 + per-class + confusion matrix +
`baseline_macro_f1` computed from real class counts. `shuffle_labels=True` is a
first-class parameter, not test-only scaffolding.

- [ ] **Step 5: report the real numbers**

macro-F1 with its baseline, the per-class table, the confusion matrix, ARI, and the
shuffled-label control. **Do not tune to improve them.** Imagenette classes are
deliberately easy to separate, so expect high accuracy — say so plainly rather than
presenting it as a triumph. If `cassette player` and `gas pump` confuse each other, name
it.

Also report, for the cross-modal endpoint: the top score for a well-matched probe
(`dog`) against a poorly-matched one (`sadness`). That gap is what justifies showing a
"nothing matches well" state in the UI.

- [ ] **Step 6: commit.**

---

## Task 6: Flutter — Images and Cross-modal tabs

**Blocked on Phase 1 Task 3.** Toolchain is ready: Flutter 3.47.2, Android SDK 36,
Visual Studio Build Tools 2022, Chrome — all three targets green.

- [ ] **Step 1:** Convert the single screen to three tabs: **Words** (Phase 1, unchanged),
  **Images**, **Cross-modal**. Each tab shows a small badge naming its space — `MiniLM ·
  384-d` or `CLIP · 512-d`. That badge is not decoration: it is what stops a reader
  assuming the numbers are comparable.
- [ ] **Step 2 — Images tab:** a grid of thumbnails; tap one to see its 5 nearest images
  with scores, its predicted class with the probability bar, and `macro_f1` beside
  `baseline_macro_f1`.
- [ ] **Step 3 — Cross-modal tab:** a text field; typing returns a row of matching
  photographs with scores. A "try these" row from `CROSS_MODAL_PROBES`. **When the top
  score is below the weak threshold, say so on screen** — "nothing here matches well" —
  rather than presenting an arbitrary result confidently. That state is the honest half
  of the demo.
- [ ] **Step 4:** the shared 2-D map: images and words projected together from the CLIP
  space, words as labels and images as thumbnails, so `dog` visibly sits among the
  spaniels. This is the picture that explains the whole phase — worth the effort.
- [ ] **Step 5:** build all three targets; **run** at least Windows desktop and Chrome.
  Report which you compiled versus actually drove.
- [ ] **Step 6:** commit.

---

## Self-Review

**Coverage:** CLIP image embedding → Task 4. Image-to-image search → Task 4. Text-to-image
and image-to-text → Task 5. Supervised on images → Task 5. Unsupervised on images →
Task 5. Three platforms → Task 6.

**The risk worth naming:** the two-space distinction is the thing most likely to be
quietly broken, because mixing them produces a plausible number rather than an error.
`test_comparing_a_minilm_vector_to_a_clip_vector_is_refused` is the guard, and it must
test a *same-dimension* mismatch too — a check that only catches 384-vs-512 is relying on
luck, and would pass a future model that happens to also emit 512.

**Second risk:** every accuracy threshold here (`> 0.85`, `> 0.4`) passes against a range
of implementations including wrong ones. The shuffled-label control is what makes them
mean something. If it does not collapse to chance, stop and report — something is
leaking, and that is more important than any other finding in this phase.
