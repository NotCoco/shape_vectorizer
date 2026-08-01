from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import RasterImageDataset
from .cuda_renderer import (
    CudaSampledShapeRenderer,
    cuda_renderer_available,
    cuda_renderer_unavailable_reason,
)
from .gpu_guard import GpuGuard
from .model_v2 import ShapeVectorizerV2, V2Prediction, circular_angle_loss
from .renderer import PARAM_COUNT, SoftShapeRenderer, decode_shapes, inverse_sigmoid
from .utils import choose_device, image_metrics, save_comparison


@dataclass(frozen=True)
class SyntheticSceneBatch:
    image: Tensor
    raw_shapes: Tensor
    raw_background: Tensor


def parse_schedule(value: str, maximum: int) -> list[int]:
    schedule = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not schedule or any(item < 1 or item > maximum for item in schedule):
        raise argparse.ArgumentTypeError(f"Shape schedule must contain values within 1..{maximum}")
    if schedule != sorted(set(schedule)):
        raise argparse.ArgumentTypeError("Shape schedule must be strictly increasing")
    return schedule


def resize_max_side_tensor(image: Tensor, max_side: int) -> Tensor:
    height, width = image.shape[-2:]
    if max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    size = (max(8, round(height * scale)), max(8, round(width * scale)))
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def make_synthetic_slot_batch(
    model: ShapeVectorizerV2,
    renderer: nn.Module,
    batch_size: int,
    shape_count: int,
    image_size: int,
    device: torch.device,
) -> SyntheticSceneBatch:
    """Create labelled coarse-to-detail scenes aligned with V2's stable slots.

    Alignment removes the permutation ambiguity that makes pixel-only SVG
    training slow: slot N always owns the vicinity and scale of anchor N.
    """
    if not 1 <= shape_count <= model.max_shapes:
        raise ValueError(f"shape_count must be within 1..{model.max_shapes}")
    raw = torch.zeros(batch_size, shape_count, PARAM_COUNT, device=device)
    types = torch.randint(0, 3, (batch_size, shape_count), device=device)
    raw[..., 0:3] = -6.0
    raw[..., 0:3].scatter_(-1, types[..., None], 6.0)

    anchors = model.anchor_centers[:shape_count].to(device=device)
    tiers = model.anchor_tiers[:shape_count].to(device=device)
    jitter_by_tier = raw.new_tensor((0.10, 0.052, 0.024))
    jitter = torch.randn(batch_size, shape_count, 2, device=device)
    jitter = jitter * jitter_by_tier[tiers][None, :, None]
    centers = (anchors[None] + jitter).clamp(0.015, 0.985)
    raw[..., 3:5] = inverse_sigmoid(centers).to(raw)

    priors = model.anchor_sizes[:shape_count].to(device=device)[None]
    overall_scale = torch.empty(batch_size, shape_count, 1, device=device).uniform_(0.62, 1.30)
    aspect = torch.empty(batch_size, shape_count, 1, device=device).normal_(0.0, 0.28).exp()
    sizes = torch.cat((priors[..., :1] * aspect, priors[..., 1:] / aspect), dim=-1)
    sizes = sizes * overall_scale
    sizes = sizes.clamp(model.min_size + 1e-4, model.max_size - 1e-4)
    size_unit = (sizes - model.min_size) / (model.max_size - model.min_size)
    raw[..., 5:7] = inverse_sigmoid(size_unit).to(raw)

    normalized_angles = torch.empty(batch_size, shape_count, device=device).uniform_(-0.95, 0.95)
    raw[..., 7] = torch.atanh(normalized_angles)
    colors = torch.empty(batch_size, shape_count, 3, device=device).uniform_(0.025, 0.975)
    raw[..., 8:11] = inverse_sigmoid(colors).to(raw)

    keep_probability = torch.empty(batch_size, 1, device=device).uniform_(0.72, 1.0)
    active = torch.rand(batch_size, shape_count, device=device) < keep_probability
    active[:, 0] = True
    raw[..., 11] = torch.where(active, raw.new_tensor(8.0), raw.new_tensor(-8.0))

    background_color = torch.empty(batch_size, 3, device=device).uniform_(0.025, 0.975)
    raw_background = inverse_sigmoid(background_color).to(device=device)
    with torch.no_grad():
        image = renderer(
            raw,
            raw_background,
            image_size,
            image_size,
            use_presence=True,
            softness_px=0.7,
        )
    return SyntheticSceneBatch(image, raw, raw_background)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value).to(dtype=value.dtype)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def synthetic_parameter_losses(
    prediction: V2Prediction,
    target_shapes: Tensor,
    target_background: Tensor,
    *,
    min_size: float,
    max_size: float,
) -> dict[str, Tensor]:
    """Supervise stable synthetic slots without comparing unstable raw logits."""
    predicted_shapes = decode_shapes(
        prediction.raw_shapes,
        min_size=min_size,
        max_size=max_size,
        hard_types=False,
    )
    decoded_targets = decode_shapes(
        target_shapes,
        min_size=min_size,
        max_size=max_size,
        hard_types=False,
    )
    active = target_shapes[..., 11] > 0.0
    type_targets = target_shapes[..., 0:3].argmax(dim=-1)
    type_loss = F.cross_entropy(
        prediction.raw_shapes[..., 0:3].reshape(-1, 3),
        type_targets.reshape(-1),
        reduction="none",
    ).reshape_as(type_targets)
    type_loss = _masked_mean(type_loss, active)
    center_loss = _masked_mean(
        F.smooth_l1_loss(predicted_shapes.centers, decoded_targets.centers, reduction="none"),
        active,
    )
    size_loss = _masked_mean(
        F.smooth_l1_loss(predicted_shapes.sizes, decoded_targets.sizes, reduction="none"),
        active,
    )
    angle_loss = _masked_mean(
        circular_angle_loss(predicted_shapes.angles, decoded_targets.angles),
        active,
    )
    color_loss = _masked_mean(
        F.smooth_l1_loss(predicted_shapes.colors, decoded_targets.colors, reduction="none"),
        active,
    )
    presence_loss = F.binary_cross_entropy_with_logits(
        prediction.raw_shapes[..., 11],
        active.to(dtype=prediction.raw_shapes.dtype),
    )
    background_loss = F.smooth_l1_loss(
        prediction.raw_background.sigmoid(),
        target_background.sigmoid(),
    )
    total = (
        0.4 * type_loss
        + 2.0 * center_loss
        + 2.0 * size_loss
        + 0.2 * angle_loss
        + color_loss
        + 0.25 * presence_loss
        + background_loss
    )
    return {
        "total": total,
        "type": type_loss,
        "center": center_loss,
        "size": size_loss,
        "angle": angle_loss,
        "color": color_loss,
        "presence": presence_loss,
        "background": background_loss,
    }


def raster_reconstruction_losses(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target images must have identical shapes")
    l1 = F.l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    edge = F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
    total = l1 + 0.25 * mse + 0.1 * edge
    return {"total": total, "l1": l1, "mse": mse, "edge": edge}


def _next_natural_batch(
    loader: DataLoader[Tensor],
    iterator: Iterator[Tensor],
) -> tuple[Tensor, Iterator[Tensor]]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the coarse-to-detail V2 SVG initializer")
    parser.add_argument("--data-dir", type=Path, help="Optional folder of natural raster images")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/train-v2"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--raster-side", type=int, default=128)
    parser.add_argument("--max-shapes", type=int, default=450)
    parser.add_argument("--shape-schedule", default="32,64,128,256,450")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--correction-steps", type=int, default=1)
    parser.add_argument("--correction-side", type=int, default=128)
    parser.add_argument("--synthetic-ratio", type=float, default=0.25)
    parser.add_argument("--full-image-ratio", type=float, default=0.35)
    parser.add_argument("--parameter-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--renderer-chunk-size", type=int, default=8)
    parser.add_argument("--renderer-backend", choices=("auto", "cuda", "torch"), default="auto")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-gpu-guard", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.synthetic_ratio <= 1.0:
        parser.error("--synthetic-ratio must be within 0..1")
    if not 0.0 <= args.full_image_ratio <= 1.0:
        parser.error("--full-image-ratio must be within 0..1")
    if args.data_dir is not None and args.full_image_ratio > 0.0 and args.batch_size != 1:
        parser.error("--full-image-ratio currently requires --batch-size 1")
    if args.data_dir is None:
        args.synthetic_ratio = 1.0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    gpu_guard = GpuGuard(enabled=device.type == "cuda" and not args.no_gpu_guard)
    gpu_guard.check(force=True)
    schedule = parse_schedule(args.shape_schedule, args.max_shapes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = ShapeVectorizerV2(
        max_shapes=args.max_shapes,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.decoder_layers,
        correction_steps=args.correction_steps,
        correction_side=args.correction_side,
    ).to(device)
    use_cuda_renderer = args.renderer_backend != "torch" and device.type == "cuda"
    if args.renderer_backend == "cuda" and not cuda_renderer_available(probe_kernel=True):
        raise RuntimeError(
            f"Custom CUDA renderer requested but unavailable: {cuda_renderer_unavailable_reason()}"
        )
    if args.renderer_backend == "cuda" and args.batch_size != 1:
        parser.error("--renderer-backend cuda currently requires --batch-size 1")
    if use_cuda_renderer:
        renderer: nn.Module = CudaSampledShapeRenderer(
            min_size=model.min_size,
            max_size=model.max_size,
            allow_fallback=args.renderer_backend == "auto",
        ).to(device)
    else:
        renderer = SoftShapeRenderer(
            min_size=model.min_size,
            max_size=model.max_size,
            chunk_size=args.renderer_chunk_size,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    loader: DataLoader[Tensor] | None = None
    loader_iterator: Iterator[Tensor] | None = None
    if args.data_dir is not None:
        dataset = RasterImageDataset(
            args.data_dir,
            args.image_size,
            full_image_probability=args.full_image_ratio,
            max_images=args.max_images,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=len(dataset) >= args.batch_size,
        )
        loader_iterator = iter(loader)
        print(f"Loaded {len(dataset)} natural images; synthetic ratio={args.synthetic_ratio:.2f}")
    else:
        print("Using unlimited labelled synthetic coarse-to-detail scenes")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    started = time.perf_counter()
    last_target: Tensor | None = None
    last_render: Tensor | None = None

    for step in range(start_step, args.steps):
        gpu_guard.callback(step=step)
        stage_index = min(len(schedule) - 1, step * len(schedule) // max(1, args.steps))
        active_shapes = schedule[stage_index]
        use_synthetic = loader is None or random.random() < args.synthetic_ratio
        synthetic_batch: SyntheticSceneBatch | None = None
        if use_synthetic:
            synthetic_batch = make_synthetic_slot_batch(
                model,
                renderer,
                args.batch_size,
                active_shapes,
                args.image_size,
                device,
            )
            target = synthetic_batch.image
        else:
            assert loader is not None and loader_iterator is not None
            target, loader_iterator = _next_natural_batch(loader, loader_iterator)
            target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            history = model.prediction_history(
                target,
                active_shapes,
                renderer=renderer,
                correction_steps=args.correction_steps,
            )
            final_prediction = history[-1]
            raster_target = resize_max_side_tensor(target, args.raster_side)
            rendered = renderer(
                final_prediction.raw_shapes,
                final_prediction.raw_background,
                raster_target.shape[-2],
                raster_target.shape[-1],
                use_presence=True,
            )
            raster_losses = raster_reconstruction_losses(rendered, raster_target)
            loss = raster_losses["total"]

            if synthetic_batch is not None:
                initial_parameter_losses = synthetic_parameter_losses(
                    history[0],
                    synthetic_batch.raw_shapes,
                    synthetic_batch.raw_background,
                    min_size=model.min_size,
                    max_size=model.max_size,
                )
                final_parameter_losses = synthetic_parameter_losses(
                    final_prediction,
                    synthetic_batch.raw_shapes,
                    synthetic_batch.raw_background,
                    min_size=model.min_size,
                    max_size=model.max_size,
                )
                parameter_loss = initial_parameter_losses["total"] + 0.5 * final_parameter_losses["total"]
                loss = loss + args.parameter_weight * parameter_loss
            else:
                parameter_loss = loss.new_zeros(())
                presence = final_prediction.raw_shapes[..., 11].sigmoid()
                loss = loss + 0.001 * presence.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        completed = step + 1
        last_target = raster_target.detach()
        last_render = rendered.detach()
        if completed == 1 or completed % args.log_every == 0:
            elapsed = time.perf_counter() - started
            metrics = image_metrics(raster_target, rendered)
            print(
                f"step={completed}/{args.steps} shapes={active_shapes} "
                f"source={'synthetic' if use_synthetic else 'natural'} "
                f"loss={float(loss.detach()):.5f} raster={float(raster_losses['total'].detach()):.5f} "
                f"parameters={float(parameter_loss.detach()):.5f} psnr={metrics['psnr']:.2f} "
                f"steps_per_sec={(completed - start_step) / max(elapsed, 1e-6):.2f}"
            )

        if completed % args.save_every == 0 or completed == args.steps:
            checkpoint_path = args.output_dir / f"checkpoint_{completed:07d}.pt"
            torch.save(
                {
                    "model_version": "v2",
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
                raster_target[:1],
                rendered[:1],
                args.output_dir / f"preview_{completed:07d}.png",
                label=f"target | reconstruction | 3x error    V2 step {completed}",
            )

    gpu_guard.check(force=True)
    summary = {
        "steps": args.steps,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "shape_schedule": schedule,
        "correction_steps": args.correction_steps,
        "renderer_backend": (
            renderer.last_backend if isinstance(renderer, CudaSampledShapeRenderer) else "torch"
        ),
        "gpu_safety": {
            "enabled": gpu_guard.enabled,
            "temperature_c": (
                gpu_guard.last_sample.temperature_c if gpu_guard.last_sample is not None else None
            ),
            "free_vram_mib": (
                gpu_guard.last_sample.memory_free_mib if gpu_guard.last_sample is not None else None
            ),
        },
    }
    if last_target is not None and last_render is not None:
        summary["last_batch_metrics"] = image_metrics(last_target, last_render)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
