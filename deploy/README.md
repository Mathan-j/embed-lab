# Deploying Embed Lab

Four pieces, all free:

| target | host | always on? |
|---|---|---|
| Backend API | **Render** (free Docker web service) | no — spins down after ~15 min idle |
| Web app | Cloudflare Pages / GitHub Pages | yes |
| Android `.apk` | GitHub Releases | yes |
| Windows `.zip` | GitHub Releases | yes |

The backend is the only part that *runs* anything; the rest are static files.

---

## 1. Backend → Render

`deploy/render.yaml` is a Blueprint, so Render can read the whole config:

**Render → New → Blueprint → connect `Mathan-j/embed-lab`.**

Or configure it by hand — **New → Web Service → Docker**:

| field | value |
|---|---|
| Repository | `Mathan-j/embed-lab` |
| Dockerfile path | `./deploy/docker/Dockerfile` |
| Docker context | `.` (the repo root, **not** `deploy/docker/`) |
| Plan | Free |
| Health check path | `/api/demos` |

The context matters: the Dockerfile `COPY`s from `backend/`, so building with
`deploy/docker/` as the context fails on the first COPY.

### What the build does, and why it takes a while

The ONNX weights are **not** in git (`**/data/models/` is gitignored), so the image
builds them:

1. the `export` stage installs torch from PyTorch's CPU index (~200 MB) and downloads
   MiniLM from the Hugging Face Hub (~90 MB)
2. `scripts/export_onnx.py` exports the pinned revision to one fp32 ONNX file
3. `scripts/fit_models.py` fits LogisticRegression + KMeans + PCA once
4. the `runtime` stage installs only the non-dev dependency group — no torch — and
   copies the artifacts across

So the **build needs outbound network** and several minutes. The **runtime does not**:
weights are baked in and `HF_HUB_OFFLINE=1` enforces it. If a build looks hung, it is
almost certainly step 1.

> **If the free plan's build times out**, the fix is to commit the exported ONNX file
> instead of building it, which trades ~86 MB of repo size for a build that only has to
> `pip install`. Measure before reaching for it.

### What to expect once it is live

- **~229 MiB resident**, verified under a hard `docker run --memory=512m` with every
  endpoint exercised — 45% of the free plan's 512 MB.
- **Cold start ~6 s** from container start to the first `200`, plus Render's own wake
  time after an idle spin-down.
- Confirm it:
  ```bash
  curl https://<your-service>.onrender.com/api/classify?text=cat
  ```
  Expect `macro_f1: 0.9119` beside `baseline_macro_f1: 0.1667`.

---

## 2. Web app → Cloudflare Pages or GitHub Pages

Flutter compiles the API URL in at **build** time, so the backend must exist first:

```bash
cd frontend
flutter build web \
  --dart-define=API_BASE=https://<your-service>.onrender.com \
  --base-href /embed-lab/          # GitHub Pages only; omit for Cloudflare Pages
```

Then publish `frontend/build/web/`. CORS is already configured on the backend.

If you deploy the web app **before** the backend, it still loads — it shows one amber
banner saying the API is unreachable, rather than five stack traces. That is deliberate
(see `frontend/lib/main.dart`), but a visitor cannot try anything, so it is worth doing
in order.

---

## 3. Android and Windows → GitHub Releases

```bash
cd frontend
flutter build apk --release --dart-define=API_BASE=https://<your-service>.onrender.com
flutter build windows      --dart-define=API_BASE=https://<your-service>.onrender.com

cd build/windows/x64/runner && powershell Compress-Archive Release embed-lab-windows.zip

gh release create v0.1.0 \
  ../../../app/outputs/flutter-apk/app-release.apk \
  embed-lab-windows.zip \
  --title "Embed Lab v0.1.0" --notes "..."
```

Both binaries bake in the same `API_BASE`, so they need the backend reachable too.

---

## Alternative: an always-on host

Render's free plan spins down. If an always-awake link matters more than setup time,
**Oracle Cloud Always Free** gives an Ampere A1 ARM VM (up to 4 cores / 24 GB) that
never sleeps. The same `deploy/docker/Dockerfile` builds there natively — every pinned
wheel publishes `linux/arm64` cp313 builds, so no buildx and no emulation.

Two things catch people on OCI:

- open the port in the subnet's **security list** *and* in the instance's own
  **iptables** — Oracle's Ubuntu images ship rules that silently drop inbound traffic
- build **on** the VM rather than pushing an amd64 image to it

This path has not been exercised here: the Dockerfile has only ever run on amd64. ARM
wheel availability was checked against the package indexes, not by building.
