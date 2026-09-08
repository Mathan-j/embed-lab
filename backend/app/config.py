"""Central settings.

No .env, no environment overrides: the model path and the pinned revision are the
only configuration this project has, and unlike `triage-lab` there is no
per-environment Qdrant/SQLite target to switch between, so a plain dataclass is
enough -- pydantic-settings would be a dependency with nothing to earn it here.
"""

from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_cache_dir: Path = BACKEND_DIR / "data" / "models"

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # ONE pinned revision, for the weights AND the tokenizer.json. Unpinned,
    # wordpiece counts and every embedding depend on whatever the Hub serves that
    # day -- the numbers on screen would move under nobody's control.
    embed_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

    torch_threads: int = 4
    seed: int = 42


settings = Settings()
