"""Build-time fit: LogisticRegression + KMeans + PCA over the 600-word
vocabulary, fit ONCE and pickled to `settings.trained_models_path`.

Run this in the RUNTIME stage of `deploy/oracle/Dockerfile`, AFTER
`uv sync --frozen --no-dev` has installed that stage's own scikit-learn --
not in the export stage, which has torch and a different resolved
scikit-learn. joblib/pickle artifacts are version-coupled: fitting with one
scikit-learn build and loading with another is a silent-corruption or
hard-failure risk. Fitting in the same environment that will load the
artifact removes the question entirely.

`app.main`'s lifespan then calls `app.train.build_models`, which LOADS this
artifact instead of re-fitting on every process start -- see `app.train`'s
module docstring and `tests/test_startup_cache.py` (the gate proving the
substitution is invisible to callers).

Usage (from backend/):
    uv run python scripts/fit_models.py [--seed N]
"""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.train import fit_models_from_scratch, save_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=settings.seed,
        help=f"default: {settings.seed} (settings.seed)",
    )
    args = parser.parse_args()

    print(f"Fitting LogisticRegression + KMeans + PCA at seed={args.seed} ...")
    models = fit_models_from_scratch(seed=args.seed)
    save_models(models, settings.trained_models_path)

    sup, uns = models.supervised, models.unsupervised
    print(
        f"Wrote {settings.trained_models_path}\n"
        f"  macro_f1={sup['macro_f1']:.4f}  baseline={sup['baseline_macro_f1']:.4f}\n"
        f"  ari={uns['ari']:.4f}"
    )


if __name__ == "__main__":
    main()
