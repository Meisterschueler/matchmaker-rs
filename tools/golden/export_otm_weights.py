#!/usr/bin/env python3
"""Export the pinned Parangonar 3.4.1 CPU checkpoint and inference fixture.

Run from the repository root with tools/golden/.venv/Scripts/python.exe
tools/golden/export_otm_weights.py. No pickle is read by the Rust runtime.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import parangonar
import torch
from parangonar.match.online_matchers import tokenize
from parangonar.match.pretrained_models import AlignmentTransformer

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = "9ec8a1b8e6731c01a9bb359635dab3fa0b9e60045bbbf18f27227237b249af97"


def main():
    checkpoint = Path(parangonar.__file__).parent / "assets" / "alignment_transformer_checkpoint.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("unexpected upstream checkpoint checksum")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    model = AlignmentTransformer(91, 64, 2, 8, 6, 0.1)
    model.load_state_dict(state)
    model.eval()
    parts, tensors, offset = [], [], 0
    for name, value in state.items():
        if name == "positional_encoder.pos_encoding":
            value = value[:26]
        array = value.detach().cpu().contiguous().numpy().astype("<f4")
        raw = array.tobytes(order="C")
        tensors.append({"name": name, "shape": list(array.shape), "offset": offset})
        parts.append(raw)
        offset += len(raw)
    data = b"".join(parts)
    manifest = {
        "format": "matchmaker-otm",
        "version": 1,
        "source": "Parangonar 3.4.1 alignment_transformer_checkpoint.pt",
        "source_url": "https://pypi.org/project/parangonar/3.4.1/",
        "source_sha256": SOURCE_SHA256,
        "license": "Apache-2.0",
        "sha256": hashlib.sha256(data).hexdigest(),
        "tensors": tensors,
    }
    out = ROOT / "assets" / "otm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.bin").write_bytes(data)
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    tokens = tokenize([{60, 64, 67}, {62}, {65, 69}, set(), {72}], [60, 61, 64, 67]).astype(np.int64)
    layers = []
    hooks = [layer.register_forward_hook(lambda _m, _i, output: layers.append(output.detach().flatten().tolist()))
             for layer in model.transformerDECODER.layers]
    with torch.no_grad():
        logits = model(torch.from_numpy(tokens[None, :, :])).flatten().tolist()
    for hook in hooks:
        hook.remove()
    fixture = {"tokens": tokens.tolist(), "layers": layers, "logits": logits}
    out = ROOT / "tests" / "golden" / "otm_model_layers"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixture.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
