from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import RasterImageDataset
from .gpu_guard import GpuGuard
from .model import ShapeVectorizer
from .renderer import SoftShapeRenderer, decode_shapes, random_raw_scenes
from .utils import choose_device, image_metrics, save_comparison


def parse_schedule(value: str, maximum: int) -> list[int]:
    schedule = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not schedule or any(item < 1 or item > maximum for item in schedule):
        raise argparse.ArgumentTypeError(f"Shape schedule must contain values within 1..{maximum}")
    return schedule


def edge_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the image-to-SVG object network")
    parser.add_argument("--data-dir", type=Path, help="Folder containing natural raster images")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-shapes", type=int, default=450)
    parser.add_argument("--shape-schedule", default="16,32,64")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--no-gpu-guard",
        action="store_true",
        help="Disable read-only NVIDIA temperature and global VRAM safety checks",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    gpu_guard_enabled = device.type == "cuda" and not args.no_gpu_guard
    gpu_guard = GpuGuard(enabled=gpu_guard_enabled)
    gpu_guard.check(force=True)
    schedule = parse_schedule(args.shape_schedule, args.max_shapes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    renderer = SoftShapeRenderer(min_size=0.01, max_size=0.5, chunk_size=32).to(device)
    model = ShapeVectorizer(max_shapes=args.max_shapes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    loader: DataLoader[Tensor] | None = None
    loader_iterator = None
    if args.data_dir:
        dataset = RasterImageDataset(args.data_dir, args.image_size, max_images=args.max_images)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=len(dataset) >= args.batch_size,
        )
        loader_iterator = iter(loader)
        print(f"Loaded {len(dataset)} images from {args.data_dir}")
    else:
        print("Using unlimited on-the-fly synthetic vector scenes")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    last_target: Tensor | None = None
    last_prediction: Tensor | None = None

    for step in range(start_step, args.steps):
        gpu_guard.callback(step=step)
        stage_index = min(len(schedule) - 1, step * len(schedule) // max(1, args.steps))
        active_shapes = schedule[stage_index]

        if loader is not None:
            assert loader_iterator is not None
            try:
                target = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(loader)
                target = next(loader_iterator)
            target = target.to(device, non_blocking=True)
        else:
            target_shape_count = max(3, min(active_shapes, 4 + active_shapes // 3))
            target_raw, target_background = random_raw_scenes(
                args.batch_size,
                target_shape_count,
                device,
                min_size=renderer.min_size,
                max_size=renderer.max_size,
            )
            with torch.no_grad():
                target = renderer(
                    target_raw,
                    target_background,
                    args.image_size,
                    args.image_size,
                    use_presence=False,
                    softness_px=0.65,
                )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw_shapes, raw_background = model(target, active_shapes)
            prediction = renderer(
                raw_shapes,
                raw_background,
                args.image_size,
                args.image_size,
                use_presence=False,
            )
            reconstruction = F.l1_loss(prediction, target) + 0.25 * F.mse_loss(prediction, target)
            edges = edge_loss(prediction, target)
            decoded_shapes = decode_shapes(
                raw_shapes,
                min_size=renderer.min_size,
                max_size=renderer.max_size,
            )
            total_bbox_area = decoded_shapes.sizes.prod(dim=-1).sum(dim=-1)
            anti_collapse = ((total_bbox_area - 1.6) / 1.6).square().mean()
            loss = reconstruction + args.edge_weight * edges + 0.01 * anti_collapse

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        last_target = target
        last_prediction = prediction
        completed = step + 1
        if completed == 1 or completed % args.log_every == 0:
            elapsed = time.perf_counter() - started
            metrics = image_metrics(target, prediction)
            print(
                f"step={completed}/{args.steps} shapes={active_shapes} "
                f"loss={float(loss.detach()):.5f} psnr={metrics['psnr']:.2f} "
                f"steps_per_sec={completed / max(elapsed, 1e-6):.2f}"
            )

        if completed % args.save_every == 0 or completed == args.steps:
            checkpoint_path = args.output_dir / f"checkpoint_{completed:07d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": completed,
                    "model_config": model.config(),
                    "training": vars(args),
                    "active_shapes": active_shapes,
                },
                checkpoint_path,
            )
            save_comparison(
                target[:1],
                prediction[:1],
                args.output_dir / f"preview_{completed:07d}.png",
                label=f"target | reconstruction | 3x error    step {completed}",
            )

    if last_target is not None and last_prediction is not None:
        final_gpu_sample = gpu_guard.check(force=True)
        final_metrics = image_metrics(last_target, last_prediction)
        summary = {
            "steps": args.steps,
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
            "last_batch_metrics": final_metrics,
            "shape_schedule": schedule,
            "gpu_guard": {
                "enabled": gpu_guard_enabled,
                "last_sample": asdict(final_gpu_sample) if final_gpu_sample is not None else None,
            },
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
