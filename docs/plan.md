# Embed Lab — MVP Implementation Plan

> **For agentic workers:** use superpowers:subagent-driven-development to execute this
> task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** One input box. You type a word or sentence; four cards show its tokens, its
384-dim embedding, its nearest neighbours, and what a trained classifier thinks it is —
every number real, computed live, on the thing you just typed.

**Architecture:** FastAPI backend holding 600 pre-embedded words in a numpy array (no
vector DB — brute-force cosine over 600×384 is instant). Two models trained at startup
from those embeddings: a supervised LogisticRegression and an unsupervised KMeans. One
Flutter screen consuming four endpoints, built for Android, Web and Windows.

**Tech Stack:** Python 3.13 / uv, FastAPI, sentence-transformers (`all-MiniLM-L6-v2`),
HuggingFace `tokenizers`, scikit-learn, numpy, pytest. Flutter for the three clients.

**Spec:** none — this plan *is* the spec. That is deliberate: this is an MVP, and the
process weight should match the scope. Where a decision needs a reason it is stated
inline.

**Deliberately NOT here:** no Qdrant, no Docker, no SQLite, no chunking, no LLM. Those
belong to `triage-lab`, the production-shaped sibling project. This one is small on
purpose.

---

## Global Constraints

- **`all-MiniLM-L6-v2` pinned at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.**
  An unpinned model resolves to whatever the Hub serves that day and moves every number
  on screen.
- **The tokenizer ships with padding AND truncation enabled at 128.** Both MUST be
  turned off (`no_padding()`, `no_truncation()`) or every wordpiece count returns 128
  and long input is silently cut. This is a real trap that cost the sibling project a
  review round.
- **Never report an accuracy or F1 without its baseline.** Six balanced categories means
  chance is 1/6 ≈ 0.167. A macro-F1 of 0.9 means something; a macro-F1 quoted alone
  does not. Same rule for every figure: quote it with the configuration it was measured
  on.
- **`torch.set_num_threads()` at most once, in the app lifespan.** No module-level call.
- **Errors the user can fix return 409 with a runnable hint**, never a stack trace.
- **A bare `pytest` must pass with no network.** The model is cached locally.
- **Before trusting a green test, ask "would this still pass if the implementation did
  nothing?"** If a test passes first try when you expected red, investigate and report
  it — that is a finding, not good news.

## Model weights — copy, do not download

The model is already cached in the sibling project:
`D:\Projects\triage-lab\backend\data\models` (19 files, 92 MB).

```bash
mkdir -p D:/Projects/embed-lab/backend/data/models
cp -r D:/Projects/triage-lab/backend/data/models/* D:/Projects/embed-lab/backend/data/models/
```

`embed-lab` stays self-contained (it is a separate repo) and needs no network.

## The vocabulary — 600 words, 6 categories

`backend/data/vocabulary.py` holds a single dict: category → list of 100 words.

| Category | Examples |
|---|---|
| `animal` | cat, dog, kitten, horse, eagle, salmon, beetle … |
| `food` | bread, cheese, mango, rice, soup, pepper, noodle … |
| `vehicle` | car, truck, bicycle, tram, ferry, glider, scooter … |
| `emotion` | joy, dread, envy, relief, grief, delight, unease … |
| `colour` | crimson, teal, ochre, indigo, beige, magenta … |
| `job` | plumber, surgeon, teacher, welder, chemist, baker … |

Rules for choosing words:
- **Single words**, lowercase, no spaces or hyphens.
- **Include a mix of tokenization shapes on purpose**: some are one wordpiece (`cat`),
  some split (`crimson` → `cr ##ims ##on`). The tokenize card is more interesting when
  it sometimes has something to show.
- **No duplicates across categories.**

### The ambiguous set — a feature, not a bug

`backend/data/vocabulary.py` also holds `AMBIGUOUS`, about 8 words that genuinely belong
to two categories: `orange` (colour/food), `mint` (food/colour), `date` (food/other),
`turkey` (animal/food), `jaguar` (animal/vehicle), `plum` (food/colour), `salmon`
(animal/food/colour), `ginger` (food/colour).

**These are excluded from training and from the 600.** They exist so the UI can show
them as a "try these" row: the classifier will confidently pick one category, and it is
not wrong — the *label* is underdetermined. That is a real lesson about supervised
learning and it costs one list to teach.

---

## Task 1: Backend, vocabulary, tokenize + embed + neighbours

**Files:**
- Create: `backend/pyproject.toml`, `backend/app/config.py`, `backend/app/main.py`
- Create: `backend/app/tokenize.py`, `backend/app/embed.py` (adapted from `triage-lab`)
- Create: `backend/data/vocabulary.py`
- Create: `backend/app/vocab.py` — loads the vocabulary, embeds it once, holds the matrix
- Create: `backend/app/api.py` — the router
- Create: `backend/tests/test_tokenize.py`, `test_embed.py`, `test_vocab.py`, `test_api.py`

**Produces:** `GET /api/tokenize?text=`, `GET /api/embed?text=`, `GET /api/neighbours?text=&k=`

- [ ] **Step 1: Scaffold**

```bash
cd D:/Projects/embed-lab/backend
uv init --no-workspace
uv add fastapi "uvicorn[standard]" sentence-transformers tokenizers scikit-learn numpy
uv add --dev pytest
```

Copy the model cache as described above. Add `data/models/` to `.gitignore` (92 MB of
weights do not belong in git); commit `.gitignore` first.

- [ ] **Step 2: Port `tokenize.py` and `embed.py` from the sibling project**

Read `D:\Projects\triage-lab\backend\app\stages\s1_tokenize.py` and `s2_embed.py`.

Take **only** what this project needs — the tokenizer singleton with its
`no_padding()`/`no_truncation()` calls and its explanation, `count_wordpieces`, and
`embed_texts` with its threading lock. **Leave behind** all the chunking machinery
(`_hard_slice`, sentence segmentation, budgets) — there is no chunking here.

Keep the comments that explain *why*. They were expensive to write and they are still
true. Drop any comment that refers to chunks, tickets, or milestones.

- [ ] **Step 3: Write the failing tests**

```python
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
    model change, the card gets boring and we want to know."""
    from app.tokenize import tokenize_detail
    d = tokenize_detail("crimson")
    assert len(d["pieces"]) > 1
    assert "".join(p.lstrip("#") for p in d["pieces"]) == "crimson"


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
    """Not 'returns 5 results' — that passes against a function returning the first 5
    words alphabetically. Assert the SEMANTICS."""
    hits = vocab.neighbours("cat", k=5)
    cats = [h for h in hits if h["category"] == "animal"]
    assert len(cats) >= 3, f"expected mostly animals, got {hits}"


def test_neighbours_excludes_the_query_word_itself(vocab):
    """'cat' is its own nearest neighbour at cosine 1.0. Returning it wastes a slot
    and looks like a bug in the UI."""
    hits = vocab.neighbours("cat", k=5)
    assert all(h["word"] != "cat" for h in hits)
```

- [ ] **Step 4: Run them, watch them fail.** Expected: `ModuleNotFoundError`.

- [ ] **Step 5: Implement**

`app/vocab.py` holds a `VocabIndex`: embeds all 600 words once at startup (~2s), keeps
an `(600, 384)` float32 matrix, the word list, and the category list.

```python
def neighbours(self, text: str, k: int = 5) -> list[dict]:
    """Brute-force cosine over 600 rows. No index, no approximation.

    600 x 384 float32 is 920 KB and one matmul is microseconds, so an ANN index here
    would add a dependency, an approximation, and a recall number to explain, in
    exchange for nothing measurable. The sibling project uses Qdrant because it has
    2,014 vectors and needs payload filtering; this has neither.
    """
    q = embed_texts([text])[0]
    scores = self.matrix @ q                      # rows are unit, q is unit -> cosine
    order = np.argsort(-scores)
    out = []
    for i in order:
        if self.words[i].lower() == text.strip().lower():
            continue                              # a word is its own best neighbour
        out.append({"word": self.words[i], "category": self.categories[i],
                    "score": round(float(scores[i]), 4)})
        if len(out) == k:
            break
    return out
```

`app/main.py` builds the index in the lifespan, calls `torch.set_num_threads` once,
and mounts the router. Endpoints return 409 with a runnable hint on empty input.

- [ ] **Step 6: Run the tests, verify they pass.**

- [ ] **Step 7: Real check, and report the numbers**

```bash
uv run uvicorn app.main:app --port 8100   # NOTE: 8100, not 8000 -- 8000 is in use
```

In another shell: `curl "http://localhost:8100/api/neighbours?text=cat&k=5"`.

Paste the real output into the report, and **say whether the neighbours look right**. If
`cat` returns vehicles, something is wrong and I want to hear it rather than see a green
suite.

Then stop the server by **the PID you started** — never `taskkill /F /IM`, which killed
an unrelated service on this machine during the sibling project.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: tokenize, embed, and 600-word neighbour search"
```

---

## Task 2: Supervised classifier and unsupervised clustering, both measured

**Files:**
- Create: `backend/app/train.py`, `backend/tests/test_train.py`
- Modify: `backend/app/api.py`, `backend/app/main.py`

**Produces:** `GET /api/classify?text=`, `GET /api/map`, and a `TrainedModels` object built
at startup.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run and watch fail. Step 3: Implement. Step 4: Run and pass.**

`train_supervised` uses `LogisticRegression(max_iter=1000)` on the 384-dim embeddings
with `train_test_split(..., stratify=y, random_state=seed)`. Report `macro_f1`,
`per_class` precision/recall/F1, the confusion matrix, **and `baseline_macro_f1`**
(1/6 for six balanced classes — compute it, do not hardcode 0.167, so it stays right
if the vocabulary changes).

`train_unsupervised` uses `KMeans(n_clusters=6, n_init=10, random_state=seed)` plus
`PCA(n_components=2)` for the map, and scores clusters against the true categories with
`adjusted_rand_score`.

**Note on PCA vs UMAP:** PCA is a linear projection and will look less tidy than UMAP.
It is chosen anyway — it is deterministic, needs no extra dependency, and its axes mean
something. Say so in the docstring so nobody "upgrades" it without knowing the trade.

**Cluster the full 384-dim vectors, never the 2-D projection.** The 2-D coordinates are
for drawing only; clustering them would be clustering an artifact of the drawing.

- [ ] **Step 5: Report the real numbers**

Run it and put in the report: `macro_f1` with its `baseline_macro_f1`, the per-class
table, the confusion matrix, and `ari`. **Do not tune to make them look better.** If a
category scores badly, name it — that is the finding. Every figure travels with its seed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: supervised classifier and kmeans clustering, both scored"
```

---

## Task 3: The Flutter app — one screen, three platforms

**BLOCKED until Flutter is installed** (~4 GB with the Android SDK). Tasks 1–2 do not
depend on it. Confirm with `flutter doctor` before starting.

**Files:** `frontend/lib/main.dart`, `frontend/lib/api.dart`, `frontend/lib/cards/*.dart`

- [ ] **Step 1:** `flutter create frontend --platforms=android,web,windows`
- [ ] **Step 2:** `api.dart` — a thin client over the four endpoints. Base URL from
  `--dart-define=API_BASE`, defaulting to `http://localhost:8100`. **Note the Android
  emulator cannot reach `localhost`** — it needs `10.0.2.2`. Handle that explicitly
  rather than leaving someone to discover it.
- [ ] **Step 3:** One screen: a text field at the top, four cards below.
  1. **Tokens** — the wordpieces as chips, with their ids and the count.
  2. **Embedding** — norm, dimension, the first 12 values, and a sparkline of all 384.
  3. **Neighbours** — the top 5 with scores and category colours, plus one deliberate
     contrast row (`cat` vs `car`) so the lesson is on screen, not just in the data.
  4. **Prediction** — the predicted category, the probability bar for all six, and the
     macro-F1 **with its baseline** underneath. A "try these" row of the ambiguous words.
- [ ] **Step 4:** The cluster map as a scatter plot, 600 points coloured by cluster, with
  the typed word plotted on it. `CustomPainter` is enough — do not add a charting
  dependency for one scatter plot.
- [ ] **Step 5:** Build all three and confirm each runs:

```bash
flutter build web
flutter build windows
flutter build apk --debug
```

Report which platforms you actually launched versus only compiled. "It builds" and
"it runs" are different claims.

- [ ] **Step 6:** Commit.

---

## Self-Review

**Coverage.** Tokenization → Task 1 cards 1. Embeddings → Task 1 cards 2–3. Supervised →
Task 2 (LogReg, stratified split, macro-F1 vs baseline, confusion matrix). Unsupervised →
Task 2 (KMeans + ARI) and Task 1 (nearest neighbours). Three platforms → Task 3.

**Placeholders.** None: every test above has its assertion, and the two elided values
(`VOCAB_FIRST_WORD`) are resolved by reading `vocabulary.py`.

**The risk worth naming.** The tests most likely to be weak are the semantic ones —
`test_neighbours_of_cat_are_mostly_animals` and the ARI/F1 thresholds. A threshold test
passes against a range of implementations including some wrong ones. Mitigation: the
`cat`-closer-to-`dog`-than-`car` test is exact and directional, and the ordering test on
the 2-D map is exact. Prove the threshold tests by mutation — shuffle the labels before
training and confirm macro-F1 collapses to the baseline.
