# Embed Lab

Type a word. See what a language model actually does with it — the tokens, the 384
numbers, the words it thinks are nearby, what a classifier trained on those numbers
predicts, and where it sits on a map of 600 words.

One FastAPI backend, one Flutter screen, three platforms: **web, Android, Windows**.

Every number on screen carries what it needs to be read correctly. That is the point of
the project as much as the code is.

---

## What it shows, for the input `cat`

| Card | Output |
|---|---|
| **Tokens** | `cat` → id `4937`, 1 wordpiece. Type `vermilion` and watch it split into 3. |
| **Embedding** | 384 dimensions, L2 norm `1.0000`, plus a sparkline of all 384 values |
| **Neighbours** | dog `0.6606` · tiger `0.5453` · rabbit `0.5426` · lion `0.5257` · ferret `0.5074` |
| **Prediction** | `animal` 66.1% — with **macro-F1 0.9119 shown beside chance 0.1667** |
| **Cluster map** | 600 words in 2-D, 6 clusters, **ARI 0.7667**, with `cat` plotted on it |

The neighbours card also prints the line the whole app exists for:

```
Spelling is not meaning -- one letter apart is not one meaning apart:
cos(cat, dog) = 0.6606     cos(cat, car) = 0.4633
```

`cat` and `car` differ by one letter and share almost no meaning. `cat` and `dog` share
no letters and a great deal of meaning. The embedding knows the difference; string
matching never could.

## Results, with the configuration they were measured on

Supervised — `LogisticRegression` over frozen 384-d embeddings, stratified 70/30 split:

| | value |
|---|---|
| macro-F1 | **0.9226** (seed 0) · 0.9119 (seed 42) |
| chance baseline | **0.1667** — computed as 1/6, not hardcoded |
| shuffled-label control | **0.1675** — collapses to chance, as it must |
| weakest classes | `animal` and `colour`, F1 0.871 each, confusing *each other* |

Unsupervised — `KMeans(k=6)` on the full 384-d vectors, scored against the true
categories:

| | value |
|---|---|
| Adjusted Rand Index | **0.7607** (seed 0) · 0.7667 (seed 42) |
| baseline | none needed — **ARI is already chance-corrected: 0.0 *is* random** |

### Why the baseline is printed next to every figure

A macro-F1 of 0.92 sounds excellent. On six balanced categories, chance is 0.167 — so
0.92 is real. But that comparison is the only thing that makes the number meaningful,
and a figure published without it invites the reader to over-read it.

The **shuffled-label control** is the other half: train on randomly scrambled labels and
macro-F1 must collapse to chance. It does (0.1675). That proves the 0.92 is learning
rather than leakage, and it proves the `> 0.7` threshold test in the suite can actually
fail. A test suite where every assertion passes no matter what the code does is not a
test suite.

## Two things the app teaches by not hiding them

**Ambiguous words.** `orange`, `mint`, `olive`, `turkey`, `jaguar`, `plum`, `salmon`,
`ginger`, `date` genuinely belong to two categories each. They are excluded from
training and shown as a "try these" row. The classifier picks one confidently — and it
is not wrong. The *label* is underdetermined. That is a real property of supervised
learning, not a defect.

**Subword leakage.** `catboat` tokenizes as `cat` + `##boat`, so its embedding is pulled
toward `cat` by spelling rather than meaning:

```
cos(cat, catboat) = 0.5941    above tiger at 0.5453
cos(catboat, boat) = 0.7455   the model does know it is a boat
cos(catboat, dog)  = 0.3205   and not an animal
```

Embeddings are *mostly* semantic, not purely. Pretending otherwise would make the app
teach something false, so the pairs `cat`/`catboat` and `ram`/`ramrod` are shown with an
honest caption. Of ~15 candidate pairs tried, only these two survived verification —
`car`/`carpet` shares no subword at all, and `bee`/`beetroot` leaks *more* than the true
meaning, which teaches the opposite lesson.

---

## Running it

Requires Python 3.13 + [uv](https://docs.astral.sh/uv/), and Flutter 3.47+ for the app.

One-time setup on a fresh clone — produces the ONNX weights and the fitted classifier,
both gitignored under `backend/data/models/`:

```bash
cd backend
uv sync
uv run python scripts/export_onnx.py    # needs network once, downloads MiniLM (~92 MB)
uv run python scripts/fit_models.py     # fits LogisticRegression + KMeans + PCA once
```

Then run it:

```bash
uv run uvicorn app.main:app --port 8100
```

After the one-time setup above, everything runs fully offline —
`HF_HUB_OFFLINE=1 uv run pytest -q` passes with no network. The Docker build (see
`deploy/README.md`) runs both of these same scripts automatically, so this manual step is
only needed for local development.

```bash
cd frontend
flutter run -d chrome                      # web
flutter run -d windows                     # Windows desktop
flutter run -d <device>                    # Android
```

On the **Android emulator** the host is `10.0.2.2`, not `localhost` — `localhost` inside
an emulator means the emulator itself. The client handles this automatically; override
with `--dart-define=API_BASE=http://host:8100` for a real device.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/tokenize?text=` | `{pieces, ids, count}` |
| `GET /api/embed?text=` | `{vector[384], dims, norm}` |
| `GET /api/neighbours?text=&k=` | `{hits: [{word, category, score}]}` |
| `GET /api/classify?text=` | `{predicted, all[], macro_f1, baseline_macro_f1, seed}` |
| `GET /api/map` | `{words, categories, clusters, coords, n_clusters, ari, seed}` |
| `GET /api/demos` | the ambiguous words and subword-leakage pairs |

Errors a user can fix — empty text, `k` out of range — return **409 with a `hint`**
naming a runnable command, never a stack trace.

## Tests

```bash
cd backend && uv run pytest -q          # 30 tests
```

Written to be able to fail. Several were proven by mutation — deliberately breaking the
implementation and confirming the test goes red:

- removing `no_padding()`/`no_truncation()` from the tokenizer breaks 3 tests
  (the shipped `tokenizer.json` enables both at 128, so every wordpiece count would
  otherwise silently return 128)
- replacing nearest-neighbour search with alphabetical order breaks the semantic test
- shuffling the 2-D map's rows against its word list breaks the ordering test
- shuffling the training labels collapses macro-F1 to chance

## Stack, and what is deliberately absent

`sentence-transformers` (`all-MiniLM-L6-v2`, revision pinned), HuggingFace `tokenizers`,
scikit-learn, numpy, FastAPI, Flutter.

**No vector database.** 600 vectors is 920 KB and one matmul over them is microseconds —
an ANN index would add a dependency, an approximation, and a recall number to explain,
in exchange for nothing measurable. (The sibling project uses Qdrant because it has
2,014 vectors *and* needs payload filtering. This has neither.)

No Docker, no SQLite, no chunking, no LLM.

## Roadmap

Phase 2 (planned, `docs/plan-images.md`): CLIP for **cross-modal** search — type a word,
get photographs back; pick a photograph, get words back; both in one shared 512-d space,
with a classifier and clustering over 1,000 Imagenette images.

The design decision that phase turns on: MiniLM is 384-d and CLIP is 512-d, and they are
*different spaces*. A cosine between them is not less accurate, it is meaningless — and
it will happily return a plausible number, so the code refuses the comparison rather
than documenting it.
