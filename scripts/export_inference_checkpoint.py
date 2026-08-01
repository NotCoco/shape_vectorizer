from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from vectorlearner.checkpoint import detect_model_version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip a training checkpoint down to portable inference data"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    model = checkpoint.get("model")
    config = checkpoint.get("model_config")
    if not isinstance(model, Mapping) or not isinstance(config, Mapping):
        raise ValueError("Checkpoint is missing model weights or model_config")

    active_shapes = int(checkpoint.get("active_shapes", config.get("max_shapes", 0)))
    if active_shapes < 1:
        raise ValueError("Checkpoint has no valid active shape count")

    inference_checkpoint = {
        "model": dict(model),
        "model_config": dict(config),
        "model_version": detect_model_version(checkpoint),
        "active_shapes": active_shapes,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, args.destination)
    print(f"Saved inference checkpoint to {args.destination}")


if __name__ == "__main__":
    main()
