# Embed Lab backend, for a Hugging Face Space (Docker SDK, free CPU tier).
#
# Two decisions worth knowing before editing this file:
#
# 1. torch is installed from PyTorch's CPU index, NOT from PyPI. The default
#    `pip install torch` on Linux pulls ~2 GB of nvidia-* and triton wheels that a
#    CPU Space can never use -- they are invisible on Windows (the deps are gated
#    `sys_platform == 'linux'`), which is exactly how they slip in. The sibling
#    project shipped 19 such packages before noticing.
#
# 2. The model is baked in at build time, not fetched on first request. A cold
#    start then costs a model LOAD (~6 s) instead of a 92 MB DOWNLOAD, which
#    matters because a free Space sleeps on inactivity and every visitor after
#    that pays the cold start.

FROM python:3.13-slim

# Spaces run the container as uid 1000. Creating that user here means the baked
# model cache and the app directory are owned by whoever actually runs them --
# otherwise the build writes as root and the runtime cannot read its own cache.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /home/user/app

# torch first, from the CPU index, so nothing later can resolve a CUDA build.
RUN pip install --no-cache-dir --user torch==2.14.0 \
      --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user app ./app
COPY --chown=user:user data ./data

# Download the pinned weights and tokenizer into the image. `get_model()` and
# `get_tokenizer()` resolve the same revision the app pins at runtime, so this
# cannot drift from what the app expects.
RUN python -c "from app.embed import get_model; from app.tokenize import get_tokenizer; get_model(); get_tokenizer()"

# Spaces route external traffic to 7860. Single worker on purpose: the model is
# ~90 MB resident and four workers would be four copies for no throughput gain
# on 2 vCPUs.
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
