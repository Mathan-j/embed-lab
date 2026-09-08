---
title: Embed Lab API
emoji: 🔤
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Tokens, embeddings, neighbours and a trained classifier over 600 words
---

# Embed Lab — API

Backend for [Embed Lab](https://github.com/Mathan-j/embed-lab): type a word, get its
WordPiece tokens, its 384-dimensional embedding, its nearest neighbours among 600 known
words, a trained classifier's prediction, and a 2-D cluster map.

Model: `sentence-transformers/all-MiniLM-L6-v2`, revision pinned to
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` so the numbers cannot move under nobody's
control.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/tokenize?text=` | `{pieces, ids, count}` |
| `GET /api/embed?text=` | `{vector[384], dims, norm}` |
| `GET /api/neighbours?text=&k=` | `{hits: [{word, category, score}]}` |
| `GET /api/classify?text=` | `{predicted, all[], macro_f1, baseline_macro_f1, seed}` |
| `GET /api/map` | `{words, categories, clusters, coords, n_clusters, ari, seed}` |
| `GET /api/demos` | the ambiguous words and subword-leakage pairs |

Try it: [`/docs`](./docs) for the interactive OpenAPI page.

## The numbers, and why the baseline travels with them

| | value |
|---|---|
| macro-F1 (supervised, held-out 30%) | **0.9119** (seed 42) |
| chance baseline | **0.1667** — six balanced categories, computed as 1/6 |
| shuffled-label control | **0.1675** — collapses to chance, as it must |
| Adjusted Rand Index (KMeans, k=6) | **0.7667** |

`macro_f1` is never returned without `baseline_macro_f1`. On six balanced classes chance
is 0.167, so 0.91 is a real result — but that comparison is the only thing that makes it
meaningful, and a figure published alone invites the reader to over-read it.

ARI needs no baseline: it is already chance-corrected, so **0.0 *is* random**.

## Cold start

This is a free CPU Space, so it sleeps after inactivity. The first request after a sleep
takes a few seconds while the model loads into memory. The weights are baked into the
image, so it is a load, not a download.

## Notes

Runs on CPU only. `torch` is installed from PyTorch's CPU index rather than PyPI — the
default install pulls roughly 2 GB of CUDA wheels this Space can never use.

Source, tests, and the Flutter client (web / Android / Windows):
**https://github.com/Mathan-j/embed-lab**
