# Embed Lab backend, for Render's free Docker web service.
#
# Two-stage build. This is the whole point of the ONNX migration, made
# concrete in the image itself:
#
#   Stage "export"  -- torch + sentence-transformers ARE installed here, to
#                       run scripts/export_onnx.py once and produce a single
#                       fp32 ONNX file (plus the tokenizer.json HF cache).
#                       This stage's layers are discarded; none of its
#                       packages reach the final image.
#
#   Stage "runtime" -- fastapi + onnxruntime + tokenizers + numpy +
#                       scikit-learn only, installed from uv.lock's non-dev
#                       group. No torch, no sentence-transformers, no
#                       transformers. This is what actually runs, and what
#                       actually has to fit a 512 MB free instance.
#
# Build context MUST be the repository root (the folder containing both
# `backend/` and `deploy/`), not `deploy/render/` -- see deploy/README.md.
#
# Measured: ~279 MB resident at startup, ~286 MB after exercising
# /api/classify, /api/map and /api/neighbours (fp32 ONNX weights, all six
# endpoints warm). Down from ~500 MB with torch loaded. See
# .superpowers/sdd/embed-lab-mvp/onnx-migration-report.md for the full
# before/after and the equivalence proof that no published number moved.

# ---------------------------------------------------------------------------
# Stage 1: export -- torch allowed, discarded after this stage
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS export

WORKDIR /build

# torch from the CPU index, not PyPI -- the default `pip install torch` on
# Linux pulls ~2 GB of nvidia-*/triton wheels a CPU-only export step never
# uses. Irrelevant to the final image's size (this stage is discarded), but
# it is also irrelevant to *this* build's time and bandwidth, so avoid it here.
RUN pip install --no-cache-dir torch==2.14.0 \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
      sentence-transformers==6.0.1 \
      onnx==1.22.0 \
      huggingface-hub

COPY backend/app ./app
COPY backend/data/__init__.py backend/data/vocabulary.py ./data/
COPY backend/scripts ./scripts

ENV PYTHONPATH=/build \
    PYTHONUNBUFFERED=1

# Produces /build/data/models/all-MiniLM-L6-v2.onnx (fp32, ~86 MB) AND
# populates the huggingface_hub cache under /build/data/models/ with
# tokenizer.json at the pinned revision -- app.tokenize and app.embed both
# resolve that same cache layout offline at runtime.
RUN python scripts/export_onnx.py \
    && python -c "from app.tokenize import get_tokenizer; get_tokenizer()"

# ---------------------------------------------------------------------------
# Stage 2: runtime -- no torch, ever
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN useradd -m -u 1000 appuser
RUN pip install --no-cache-dir uv

# Everything below runs AS appuser and every COPY sets its owner directly.
# The earlier version ended with `RUN chown -R appuser:appuser /home/appuser`,
# which rewrites every file it touches into a fresh layer -- a 526 MB
# duplicate of the venv, the model and the app in a 1.87 GB image. Docker
# layers are copy-on-write, so a recursive chown after the fact never edits
# in place; it copies. `--chown` on each COPY and a `USER` switch before the
# install get the same ownership for nothing.
USER appuser
WORKDIR /home/appuser/app

# Resolve and materialise the RUNTIME dependency group only. `--no-dev`
# excludes torch/sentence-transformers/onnx (dev-only in pyproject.toml);
# `--frozen` refuses to silently re-resolve against a stale lockfile.
COPY --chown=appuser:appuser backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser backend/app ./app
COPY --chown=appuser:appuser backend/data/__init__.py backend/data/vocabulary.py ./data/
# The exported ONNX file + the tokenizer's HF cache, baked in at build time --
# a cold start is then a model LOAD, not a download, and the container never
# needs network access at runtime (HF_HUB_OFFLINE=1 below enforces that).
COPY --from=export --chown=appuser:appuser /build/data/models ./data/models

ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1

# Render sets $PORT at runtime and expects the process to bind it; there is
# no fixed port to EXPOSE reliably, so the CMD below falls back to 10000 for
# any environment that does not set it (e.g. a plain `docker run`).
EXPOSE 10000

# Single worker on purpose: the ONNX session is one shared, already-loaded
# object per process; more workers would be more copies of it for no
# throughput gain on a free instance's fractional CPU.
CMD ["sh", "-c", ".venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
