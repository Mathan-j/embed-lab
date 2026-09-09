# Deploying Embed Lab's backend

**Target: Oracle Cloud's Always Free tier, Ampere A1 (ARM64) shape.** Render's free Docker
service was the original target; it has been dropped in favour of Oracle's Always Free
compute, which is not time-limited (Render's free instances also spin down after 15
minutes idle -- Oracle's does not). `deploy/render/` (and Hugging Face Spaces before it,
`deploy/hf-space/`) have been **deleted**, not left behind -- a stale deploy dir aimed at
a host that's been rejected is a trap for the next reader.

## Which Always Free shape, and why it matters

Oracle's Always Free tier offers **two** compute shapes. Only one of them works here:

| Shape | Architecture | Always Free allowance | Use it? |
|---|---|---|---|
| VM.Standard.A1.Flex ("Ampere A1") | **ARM64 (aarch64)** | up to 4 OCPUs / 24 GB RAM, total across all A1 instances | **Yes -- this is the target** |
| VM.Standard.E2.1.Micro (AMD) | amd64 | 1 GB RAM, 1/8 OCPU | No -- 1 GB is too tight, and it's the wrong point of this guide |

**The image this project builds is amd64 by default (the export/runtime stages both use
`python:3.13-slim`, which is multi-arch, but whatever host builds it decides the arch of
what comes out).** An amd64 image will not run on the ARM64 Ampere A1 instance. The fix is
not to cross-compile -- it's to build ON the ARM VM, which is the simplest correct answer
and is what this guide does. Ampere A1 gives you up to 4 OCPUs and 24 GB of RAM, far more
than this build needs, so there is no reason to fight `buildx` and QEMU emulation for it.

If you really want to build cross-platform from an amd64 machine instead: `docker buildx
build --platform linux/arm64 ...` works, but emulated (QEMU) ARM builds are **slow** --
plan for the export stage's torch install and the transformer export to take several times
longer than a native build. This guide does not use that path.

## What's here

```
deploy/oracle/Dockerfile   -- three-stage build (see the file's own comments)
deploy/README.md           -- this guide
```

No `deploy/oracle/*.yaml` or platform-specific config -- there is no Oracle equivalent of
Render's Blueprint needed here; you build and run the container directly on the VM.

### The Dockerfile's three stages

1. **`export`** -- installs torch + sentence-transformers, runs
   `backend/scripts/export_onnx.py` to produce a single fp32 ONNX file from the pinned
   revision (`sentence-transformers/all-MiniLM-L6-v2` @
   `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), plus warms the tokenizer's HF cache. This
   stage's layers, and everything installed in them, are **discarded** -- they never reach
   the image that actually runs.
2. **`runtime`** -- installs only the `uv.lock` non-dev dependency group (fastapi,
   onnxruntime, tokenizers, numpy, scikit-learn, huggingface-hub, uvicorn), copies in the
   app code and the ONNX file + tokenizer cache built in stage 1, then **fits
   LogisticRegression + KMeans + PCA once** (`scripts/fit_models.py`) and bakes the result
   in as `data/models/trained_models.joblib`. Runs as a non-root user. No torch anywhere in
   this image.

The fit happens in the `runtime` stage deliberately, not the `export` stage: `export` has a
different resolved scikit-learn (installed alongside torch/onnx), and joblib/pickle
artifacts are version-coupled to the scikit-learn that wrote them. Fitting and loading in
the same environment removes that risk entirely. See
`.superpowers/sdd/embed-lab-mvp/startup-cache-report.md` for the measured before/after and
the equivalence proof that no published number moved.

**fp32, not int8-quantised.** Quantising would shrink the ~86 MB ONNX file further, but it
moves the outputs -- and the whole point of the ONNX migration was that no published
number may move.

**Building this image needs outbound network access; running the container does not.**
The `export` stage downloads torch (~200 MB) and the MiniLM weights (~90 MB) from the Hub
to produce the ONNX file -- expect this to take several minutes even on a 4-core ARM
instance, mostly download time, not CPU. Once built, the runtime container needs no
network at all: `HF_HUB_OFFLINE=1` is set, and every weight, tokenizer file, and fitted
model it needs was baked in at build time.

## Deploy steps

### 1. Create the Always Free ARM instance

In the OCI Console: **Compute -> Instances -> Create Instance**.

- **Image**: Canonical Ubuntu (22.04 or later) -- the "Always Free-eligible" images are
  marked as such in the picker.
- **Shape**: click "Change shape" -> **Ampere** -> **VM.Standard.A1.Flex**. Set OCPUs and
  memory within the Always Free allowance (e.g. 2 OCPU / 12 GB, or up to 4 OCPU / 24 GB --
  whatever this tenancy has left of the always-free A1 pool).
- **Networking**: use an existing VCN/subnet or let OCI create one, and make sure "Assign a
  public IPv4 address" is checked.
- **SSH keys**: add your public key (or have OCI generate a pair and download the private
  key) -- you'll need SSH access to run the steps below.

Create the instance and wait for it to reach the Running state. Note its **public IP** from
the instance details page -- you'll use it locally to SSH in and to curl the API; it does
not belong in this repo (see the note at the bottom).

### 2. Open the port -- in BOTH places

This is the step that catches almost everyone. Oracle's Ubuntu images ship with **local
iptables rules that drop inbound traffic by default**, on top of the **OCI security list**
that gates the subnet. Opening only one of the two still leaves the port unreachable, with
no error on either side -- the connection just hangs or resets. Both are required.

**a. OCI security list / NSG** (controls traffic at the network level):

Instance details -> the subnet link -> **Security Lists** -> the list attached to that
subnet -> **Add Ingress Rules**:
- Source CIDR: `0.0.0.0/0` (or narrower, if you want to restrict who can reach it)
- IP Protocol: TCP
- Destination Port Range: `10000` (the port the container binds -- see below)

**b. The instance's own iptables** (controls traffic inside the VM):

SSH into the instance, then:

```bash
sudo iptables -I INPUT -p tcp --dport 10000 -j ACCEPT
sudo netfilter-persistent save   # or: sudo iptables-save > /etc/iptables/rules.v4
```

Skip the second command and the rule vanishes on reboot -- persist it.

### 3. Install Docker on the instance

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

sudo usermod -aG docker "$USER"   # log out and back in for this to take effect
```

Confirm: `docker run --rm hello-world` and `uname -m` should print `aarch64`.

### 4. Clone and build

```bash
git clone <this-repo-url>
cd embed-lab
docker build -f deploy/oracle/Dockerfile -t embed-lab-api .
```

This runs natively on the instance's own ARM64 CPUs -- no `--platform` flag, no buildx, no
emulation. Expect several minutes for the `export` stage (network-bound: downloading torch
and the MiniLM weights) the first time; subsequent builds reuse Docker's layer cache unless
`backend/app`, `backend/pyproject.toml`, or `backend/uv.lock` changed.

### 5. Run it, with a restart policy

```bash
docker run -d \
  --name embed-lab-api \
  --restart unless-stopped \
  -p 10000:10000 \
  -e PORT=10000 \
  embed-lab-api
```

`--restart unless-stopped` brings the container back after a reboot or a crash without
manual intervention -- the only sane default for something meant to stay up unattended on
a free instance nobody is watching.

### 6. Confirm

From your own machine (not the instance):

```bash
curl "http://<instance-public-ip>:10000/api/classify?text=cat"
curl "http://<instance-public-ip>:10000/api/demos"
```

A hang or a connection reset here almost always means step 2 was only done in one place --
recheck both the OCI security list and the instance's iptables.

## Cold start

Cold start is now: FastAPI lifespan builds the `VocabIndex` (embeds 600 words with ONNX,
~1.9 s) then **loads** the pre-fit classifier + scored figures from
`data/models/trained_models.joblib` (a few ms) instead of re-fitting
LogisticRegression/KMeans/PCA on every start. See
`.superpowers/sdd/embed-lab-mvp/startup-cache-report.md` for the measured before/after (amd64;
this has not been measured on the ARM instance itself -- see that report's verification
section).

Unlike Render's free tier, Oracle's Always Free instances do not spin down when idle, so
this cost is paid once at container start (or restart), not on every visitor after a gap.

## Verifying a build locally (amd64) before touching the VM

You cannot produce an ARM64 image this way, but you can confirm the Dockerfile itself is
sound -- the build steps, the fit-at-build step, the endpoints, and the torch-free
runtime -- before ever SSHing into the instance:

```bash
cd /path/to/embed-lab           # repo root, NOT backend/ or deploy/oracle/
docker build -f deploy/oracle/Dockerfile -t embed-lab-api .
docker run --rm -p 8100:10000 -e PORT=10000 embed-lab-api
curl http://localhost:8100/api/classify?text=cat
```

Then confirm no torch reached the image:

```bash
docker run --rm embed-lab-api .venv/bin/python -c "import torch" 2>&1
# must fail with ModuleNotFoundError
```

## No IPs, hostnames, or credentials belong in this repo

The steps above use `<instance-public-ip>` as a placeholder on purpose. Keep the real
public IP, any SSH key material, and the repo clone URL (if private) out of version
control -- note them somewhere private (a password manager, your own README that is not
committed) instead.
