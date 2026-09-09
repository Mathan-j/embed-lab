# Deploying Embed Lab's backend

**Target: Render's free tier.** Hugging Face Spaces would need PRO for a Docker Space,
so `deploy/hf-space/` (and its `deploy/build-space.sh` helper) have been **deleted** --
they are not a fallback, they are a trap for the next reader who tries to deploy there
and hits a paywall. Render is the only supported target now.

## Why this needed a runtime change, not just a smaller instance

The original backend loaded `torch` + `sentence-transformers` at startup and measured
**~500 MB working set** with the model loaded and all endpoints exercised. Render's free
instances cap at 512 MB. That is 12 MB of headroom for a language runtime, a web
framework, scikit-learn, and every request that comes in — not viable.

The fix was **onnxruntime instead of torch**, not a smaller torch. See
`.superpowers/sdd/embed-lab-mvp/onnx-migration-report.md` for the full before/after,
the numeric-equivalence proof against torch, and the mutation test that proves the proof
can fail. In short: after the swap, the same measurement method reads **~279 MB at
startup, ~286 MB after hitting `/api/classify`, `/api/map`, `/api/neighbours`** — comfortably
under 512 MB, with room to spare.

## What's here

```
deploy/render/Dockerfile   -- two-stage build (see below)
deploy/render/render.yaml  -- Render Blueprint (optional; manual setup works too)
```

### The Dockerfile is two stages on purpose

1. **`export`** -- installs torch + sentence-transformers, runs
   `backend/scripts/export_onnx.py` to produce a single fp32 ONNX file from the pinned
   revision (`sentence-transformers/all-MiniLM-L6-v2` @
   `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), plus warms the tokenizer's HF cache.
   This stage's layers, and everything installed in them, are **discarded** — they never
   reach the image that actually runs.
2. **`runtime`** -- installs only the `uv.lock` non-dev dependency group (fastapi,
   onnxruntime, tokenizers, numpy, scikit-learn, huggingface-hub, uvicorn), copies in the
   app code and the ONNX file + tokenizer cache built in stage 1, and runs as a
   non-root user. No torch anywhere in this image.

**fp32, not int8-quantised.** Quantising would shrink the ~86 MB ONNX file further, but
it moves the outputs — and the whole point of this migration is that no published number
may move. See the migration report for what fp32 costs: about 86 MB of resident weights
versus roughly half that for int8, which was judged not worth trading the numbers for.

## Deploy steps (manual Web Service — no render.yaml needed)

1. Push this repo (with `deploy/render/Dockerfile`) to GitHub.
2. Render dashboard → **New +** → **Web Service** → connect the repo.
3. Runtime: **Docker**. Set:
   - **Dockerfile Path**: `deploy/render/Dockerfile`
   - **Docker Build Context Directory**: `.` (the repo root — the Dockerfile's `COPY`
     lines reach into `backend/`, so the context must be the root, not `deploy/render/`)
4. **Instance Type**: Free.
5. **Health Check Path**: `/api/demos` (cheap, no model inference, still proves the app
   booted and the router is wired).
6. Deploy. The first build runs the `export` stage (needs network access to fetch the
   pinned weights from the Hub) — expect several minutes for that build, one-time per
   deploy that doesn't hit Render's build cache.

## Deploy steps (Blueprint / render.yaml)

Render only auto-discovers `render.yaml` at the repo root. Since this one lives at
`deploy/render/render.yaml` instead (co-located with the Dockerfile it describes, matching
this project's `deploy/<target>/` layout), point the Blueprint at it explicitly:
**New +** → **Blueprint** → pick the repo → set the Blueprint file path to
`deploy/render/render.yaml`. Everything else follows from that file.

## Free tier: idle spin-down and the cold-start cost

Render's free instances **spin down after 15 minutes of no inbound traffic** and spin
back up on the next request. Two costs land on that first request after a sleep:

- **Container cold boot** — Render provisions and starts the container (several seconds,
  outside the app's control).
- **Model load** — `app.main`'s lifespan builds the `VocabIndex` (embeds 600 words, ~1-2s
  with ONNX) and trains the live classifier + both scored figures once. The ONNX weights
  are baked into the image (no download at boot), so this is a load, not a fetch — the
  same trade the retired HF Space Dockerfile made for the same reason.

Every visitor after a 15-minute-idle gap pays this once; subsequent requests in the same
warm window do not.

## Verifying a deploy locally before pushing

```bash
cd /path/to/embed-lab           # repo root, NOT backend/ -- see build context above
docker build -f deploy/render/Dockerfile -t embed-lab-api .
docker run --rm -p 8100:10000 -e PORT=10000 embed-lab-api
curl http://localhost:8100/api/classify?text=cat
```

Then confirm no torch reached the image:

```bash
docker run --rm embed-lab-api .venv/bin/python -c "import torch" 2>&1
# must fail with ModuleNotFoundError
```
