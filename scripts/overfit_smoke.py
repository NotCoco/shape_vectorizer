from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from vectorlearner.model import ShapeVectorizer
from vectorlearner.renderer import SoftShapeRenderer, random_raw_scenes
from vectorlearner.utils import image_metrics, save_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit two fixed scenes to verify GPU learning")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--shapes", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/overfit-smoke"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test expects CUDA")
    torch.manual_seed(123)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = SoftShapeRenderer(chunk_size=8).to(device)
    model = ShapeVectorizer(
        max_shapes=450,
        hidden_dim=64,
        decoder_layers=1,
        attention_heads=4,
        max_feature_side=12,
    ).to(device)
    target_raw, target_background = random_raw_scenes(2, 7, device)
    with torch.no_grad():
        target = renderer(
            target_raw,
            target_background,
            args.image_size,
            args.image_size,
            use_presence=False,
            softness_px=0.65,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    initial_loss = None
    prediction = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        raw_shapes, raw_background = model(target, args.shapes)
        prediction = renderer(
            raw_shapes,
            raw_background,
            args.image_size,
            args.image_size,
            use_presence=False,
        )
        loss = F.l1_loss(prediction, target) + 0.25 * F.mse_loss(prediction, target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f"step={step + 1}/{args.steps} loss={float(loss.detach()):.6f}")

    assert initial_loss is not None and prediction is not None
    final_loss = float(loss.detach())
    if not final_loss < initial_loss * 0.85:
        raise RuntimeError(
            f"Learning check failed: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )
    save_comparison(
        target[:1],
        prediction[:1],
        args.output_dir / "comparison.png",
        label="target | learned reconstruction | 3x error",
    )
    result = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement_percent": 100.0 * (initial_loss - final_loss) / initial_loss,
        "metrics": image_metrics(target, prediction),
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
