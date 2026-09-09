# Embed Lab backend, for a Docker container on Oracle Cloud's Always Free
# Ampere A1 (ARM64) VM. See deploy/README.md for provisioning the instance,
# opening the port (OCI security list AND the instance's own iptables), and
# building/running this image.
#
# No architecture-specific lines in this file: `python:3.13-slim` and every
# pinned wheel below (torch, onnxruntime, scikit-learn) publish linux/arm64
# builds for cp313, so building this Dockerfile ON an ARM64 host (the
# documented, simplest path -- see deploy/README.md) just naturally produces
# an arm64 image. No buildx, no --platform flag, no emulation.
#
# Three-stage build:
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
#                       actually has to fit a 512 MB budget (Always Free ARM
#                       has far more than that available, but the budget this
#                       image was built to is still 512 MB -- see README.md).
#                       This stage ALSO fits LogisticRegression + KMeans + PCA
#                       once (`scripts/fit_models.py`) and bakes the result in
#                       -- see the RUN step below for why it happens HERE and
#                       not in the export stage.
#
# Build context MUST be the repository root (the folder containing both
# `backend/` and `deploy/`), not `deploy/oracle/` -- see deploy/README.md.
#
# Building this image requires OUTBOUND network access (the export stage
# downloads torch (~200 MB) and the MiniLM weights (~90 MB) from the Hub to
# produce the ONNX file). The RUNTIME CONTAINER needs none: `HF_HUB_OFFLINE=1`
# below enforces that everything it needs was baked in at build time.
#
# Measured (amd64, this Dockerfile, --memory=512m, cold start to first
# successful /api/classify): see
# .superpowers/sdd/embed-lab-mvp/startup-cache-report.md for the full
# before/after. ARM64 has not been measured -- there is no ARM hardware
# available where this was built; the guide in README.md is written so the
# first real run happens on the target VM itself.

# ---------------------------------------------------------------------------
# Stage 1: export -- torch allowed, discarded after this stage
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS export

WORKDIR /build

# torch from the CPU index, not PyPI -- the default `pip install torch` pulls
# ~2 GB of nvidia-*/triton wheels a CPU-only export step never uses. The CPU
# index carries linux/aarch64 wheels too, not just x86_64 -- this line needs
# no change to build on the ARM VM.
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
# An earlier version ended with `RUN chown -R appuser:appuser /home/appuser`,
# which rewrites every file it touches into a fresh layer -- a 526 MB
# duplicate of the venv, the model and the app in a 1.87 GB image. Docker
# layers are copy-on-write, so a recursive chown after the fact never edits
# in place; it copies. `--chown` on each COPY and a `USER` switch before the
# install get the same ownership for nothing.
USER appuser
WORKDIR /home/appuser/app

# Resolve and materialise the RUNTIME dependency group only. `--no-dev`
# excludes torch/sentence-transformers/onnx (dev-only in pyproject.toml);
# `--frozen` refuses to silently re-resolve against a stale lockfile. This is
# also, deliberately, the environment `scripts/fit_models.py` fits in below --
# see that RUN step.
COPY --chown=appuser:appuser backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser backend/app ./app
COPY --chown=appuser:appuser backend/data/__init__.py backend/data/vocabulary.py ./data/
# The exported ONNX file + the tokenizer's HF cache, baked in at build time --
# a cold start is then a model LOAD, not a download, and the container never
# needs network access at runtime (HF_HUB_OFFLINE=1 below enforces that).
COPY --from=export --chown=appuser:appuser /build/data/models ./data/models
COPY --chown=appuser:appuser backend/scripts/fit_models.py ./scripts/fit_models.py

ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1

# Fits LogisticRegression + KMeans + PCA over the 600-word vocabulary ONCE,
# right here in the runtime stage's own scikit-learn -- deliberately NOT in
# the export stage above, which has torch and a different resolved
# scikit-learn. joblib/pickle artifacts are version-coupled: fitting with one
# scikit-learn build and loading with another (app.main's lifespan, at boot)
# is a silent-corruption or hard-failure risk. Fitting and loading in the
# same environment removes the question. Writes
# ./data/models/trained_models.joblib, which app.train.build_models loads at
# boot instead of re-fitting -- see app.train's module docstring and
# tests/test_startup_cache.py (the equivalence gate for this change). No
# network needed: HF_HUB_OFFLINE=1 is already set, and the ONNX weights +
# tokenizer cache were baked in by the COPY above.
RUN .venv/bin/python scripts/fit_models.py

# The container binds $PORT, defaulting to 10000 if unset -- see
# deploy/README.md for which port to open in the OCI security list and the
# instance's iptables.
EXPOSE 10000

# Single worker on purpose: the ONNX session is one shared, already-loaded
# object per process; more workers would be more copies of it for no
# throughput gain relative to what this app needs.
CMD ["sh", "-c", ".venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
