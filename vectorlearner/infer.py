from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_vectorizer_checkpoint
from .cuda_renderer import CudaSampledShapeRenderer
from .data import load_rgb, pil_to_tensor, resize_max_side
from .model import ShapeVectorizer
from .model_v2 import ShapeVectorizerV2
from .renderer import SoftShapeRenderer
from .svg import save_svg
from .utils import choose_device, image_metrics, save_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an image to SVG with a trained checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/infer"))
    parser.add_argument("--shapes", type=int)
    parser.add_argument("--encoder-max-side", type=int, default=512)
    parser.add_argument("--preview-max-side", type=int, default=1024)
    parser.add_argument(
        "--correction-steps",
        type=int,
        help="V2 learned correction passes; defaults to the number trained into the checkpoint",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    device = choose_device(args.device)
    loaded = load_vectorizer_checkpoint(args.checkpoint, device)
    if not isinstance(loaded.model, (ShapeVectorizer, ShapeVectorizerV2)):
        raise TypeError("Unsupported vectorizer model")
    model = loaded.model
    active_shapes = args.shapes or loaded.trained_shapes
    if active_shapes > model.max_shapes:
        parser.error(f"--shapes cannot exceed the checkpoint maximum of {model.max_shapes}")
    if active_shapes < 1:
        parser.error("--shapes must be positive")

    if device.type == "cuda":
        renderer: SoftShapeRenderer | CudaSampledShapeRenderer = CudaSampledShapeRenderer(
            min_size=model.min_size,
            max_size=model.max_size,
            allow_fallback=True,
        ).to(device)
    else:
        renderer = SoftShapeRenderer(
            min_size=model.min_size,
            max_size=model.max_size,
            chunk_size=16,
        ).to(device)
    if isinstance(model, ShapeVectorizerV2):
        correction_steps = (
            model.correction_steps if args.correction_steps is None else args.correction_steps
        )
        if not 0 <= correction_steps <= model.correction_steps:
            parser.error(
                f"--correction-steps must be within 0..{model.correction_steps} for this checkpoint"
            )
    else:
        if args.correction_steps not in (None, 0):
            parser.error("--correction-steps is only supported by V2 checkpoints")
        correction_steps = 0

    original = load_rgb(args.image)
    encoder_image = resize_max_side(original, args.encoder_max_side)
    encoder_tensor = pil_to_tensor(encoder_image)[None].to(device)
    with torch.inference_mode():
        if isinstance(model, ShapeVectorizerV2):
            raw_shapes, raw_background = model(
                encoder_tensor,
                active_shapes,
                renderer=renderer,
                correction_steps=correction_steps,
            )
        else:
            raw_shapes, raw_background = model(encoder_tensor, active_shapes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_presence = loaded.model_version == "v2"
    save_svg(
        args.output_dir / "result.svg",
        raw_shapes,
        raw_background,
        original.width,
        original.height,
        min_size=model.min_size,
        max_size=model.max_size,
        use_presence=use_presence,
        presence_threshold=0.25 if use_presence else 0.0,
        title=args.image.name,
    )

    preview_image = resize_max_side(original, args.preview_max_side)
    target = pil_to_tensor(preview_image)[None].to(device)
    with torch.inference_mode():
        prediction = renderer(
            raw_shapes,
            raw_background,
            preview_image.height,
            preview_image.width,
            use_presence=use_presence,
            softness_px=0.7,
        )
    save_comparison(
        target,
        prediction,
        args.output_dir / "comparison.png",
        label=f"target | reconstruction | 3x error    {active_shapes} shapes",
    )
    metrics = image_metrics(target, prediction)
    (args.output_dir / "inference.json").write_text(
        json.dumps(
            {
                "source": str(args.image.resolve()),
                "native_size": [original.width, original.height],
                "encoder_size": list(encoder_image.size),
                "shapes": active_shapes,
                "model_version": loaded.model_version,
                "correction_steps": correction_steps,
                "checkpoint": str(args.checkpoint.resolve()),
                "preview_metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {args.output_dir / 'result.svg'}")


if __name__ == "__main__":
    main()
