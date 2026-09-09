"""Build-time export: sentence-transformers/all-MiniLM-L6-v2 (pinned revision) ->
a single fp32 ONNX file, run once on a dev machine where torch is installed.

This script is the ONLY place in the project allowed to import torch or
sentence-transformers for anything other than the equivalence test. Its output
(`data/models/all-MiniLM-L6-v2.onnx`) is what `app.embed` loads at runtime with
onnxruntime alone -- no torch, no transformers, no optimum.

What gets exported is the bare `BertModel` (the SentenceTransformer's module 0,
`Transformer`), NOT the full SentenceTransformer pipeline. Pooling (mean,
attention-mask weighted) and L2 normalisation are module 1 (`Pooling`) and
module 2 (`Normalize`) in the SentenceTransformer -- both are a handful of numpy
operations, so `app.embed` reimplements them itself rather than dragging in a
graph-export of a mean/normalize op for no benefit. See app.embed's module
docstring for exactly how, and tests/test_onnx_equivalence.py for the proof
that the reimplementation matches torch bit-for-bit within tolerance.

Usage (from backend/):
    uv run python scripts/export_onnx.py
"""

import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402

OUTPUT_PATH = settings.model_cache_dir / "all-MiniLM-L6-v2.onnx"


def main() -> None:
    print(f"Loading {settings.embed_model!r} @ {settings.embed_model_revision} ...")
    st_model = SentenceTransformer(
        settings.embed_model,
        revision=settings.embed_model_revision,
        cache_folder=str(settings.model_cache_dir),
        device="cpu",
    )
    transformer_module = st_model[0]
    auto_model = transformer_module.auto_model
    auto_model.eval()
    auto_model.config.use_cache = False  # avoids a tracer kwarg clash in forward()

    architecture = transformer_module.auto_model.config.architectures
    print(f"Underlying architecture: {architecture}")
    assert st_model[1].pooling_mode == "mean", (
        "Export assumes mean pooling -- app.embed reimplements exactly that. "
        f"Got pooling_mode={st_model[1].pooling_mode!r}."
    )

    tokenizer = transformer_module.tokenizer
    sample = tokenizer(
        ["cat", "a longer example phrase to exercise a real sequence length"],
        padding=True,
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    dummy_inputs = (
        sample["input_ids"],
        sample["attention_mask"],
        sample.get("token_type_ids", torch.zeros_like(sample["input_ids"])),
    )

    class ForwardOnly(torch.nn.Module):
        """Tracing `auto_model` directly hits a tracer/kwarg-injection clash on
        `use_cache` inside transformers' decorated forward(). Wrapping it in a
        plain module that calls it with explicit kwargs and unpacks a tuple
        (not a ModelOutput dataclass, which the tracer can't flatten) sidesteps
        both issues without changing a single number the model computes."""

        def __init__(self, base: torch.nn.Module):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )
            return out[0]  # last_hidden_state

    wrapped = ForwardOnly(auto_model)
    wrapped.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {OUTPUT_PATH} ...")
    torch.onnx.export(
        wrapped,
        dummy_inputs,
        str(OUTPUT_PATH),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript-based exporter: no onnxscript dependency
    )

    import onnx as onnx_pkg

    onnx_model = onnx_pkg.load(str(OUTPUT_PATH))
    onnx_pkg.checker.check_model(onnx_model)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUTPUT_PATH} ({size_mb:.1f} MB), opset 17, fp32, checked OK.")


if __name__ == "__main__":
    main()
