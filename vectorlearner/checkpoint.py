from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import ShapeVectorizer
from .model_v2 import ShapeVectorizerV2


@dataclass(frozen=True)
class LoadedVectorizer:
    model: nn.Module
    model_version: str
    trained_shapes: int
    checkpoint: dict[str, Any]


def detect_model_version(checkpoint: Mapping[str, Any]) -> str:
    """Identify explicit checkpoints and V2 checkpoints made before the marker existed."""
    declared = checkpoint.get("model_version")
    if declared is not None:
        normalized = str(declared).strip().lower()
        if normalized in {"1", "v1"}:
            return "v1"
        if normalized in {"2", "v2"}:
            return "v2"
        raise ValueError(f"Unsupported checkpoint model_version: {declared!r}")

    config = checkpoint.get("model_config")
    state = checkpoint.get("model")
    if isinstance(config, Mapping) and (
        "encoder_base_channels" in config
        or "correction_steps" in config
        or "correction_side" in config
    ):
        return "v2"
    if isinstance(state, Mapping) and any(
        str(key).startswith(("encoder.", "correction_encoder.")) or key == "anchor_tiers"
        for key in state
    ):
        return "v2"
    return "v1"


def load_vectorizer_checkpoint(path: str | Path, device: torch.device) -> LoadedVectorizer:
    checkpoint_path = Path(path)
    checkpoint_object = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint_object, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    checkpoint: dict[str, Any] = checkpoint_object
    config = checkpoint.get("model_config")
    state = checkpoint.get("model")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Checkpoint is missing model_config or model weights")

    model_version = detect_model_version(checkpoint)
    model: nn.Module
    if model_version == "v2":
        model = ShapeVectorizerV2(**config)
    else:
        model = ShapeVectorizer(**config)
    model = model.to(device)
    model.load_state_dict(state)
    model.eval()

    max_shapes = int(getattr(model, "max_shapes"))
    trained_shapes = int(checkpoint.get("active_shapes", max_shapes))
    if not 1 <= trained_shapes <= max_shapes:
        raise ValueError(
            f"Checkpoint active_shapes must be within 1..{max_shapes}, got {trained_shapes}"
        )
    return LoadedVectorizer(model, model_version, trained_shapes, checkpoint)


def newest_checkpoint(directory: str | Path, fallback: str | Path) -> Path:
    """Return the most recently written .pt checkpoint, or the supplied fallback."""
    root = Path(directory)
    candidates = (
        [path for path in root.rglob("checkpoint_*.pt") if path.is_file()]
        if root.is_dir()
        else []
    )
    if not candidates:
        return Path(fallback)

    def modified(path: Path) -> tuple[int, str]:
        try:
            timestamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            timestamp = -1
        return timestamp, str(path)

    return max(candidates, key=modified)
