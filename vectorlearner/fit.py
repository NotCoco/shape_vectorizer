from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cuda_renderer import (
    CudaSampledShapeRenderer,
    cuda_renderer_available,
    cuda_renderer_unavailable_reason,
)
from .checkpoint import load_vectorizer_checkpoint
from .data import load_rgb, pil_to_tensor, resize_max_side
from .gpu_guard import GpuGuard
from .model_v2 import ShapeVectorizerV2
from .renderer import PARAM_COUNT, SoftShapeRenderer, inverse_sigmoid
from .svg import save_svg
from .utils import choose_device, image_metrics, save_comparison


def integer_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers")
    return result


def resized_target(target_cpu: Tensor, max_side: int, device: torch.device) -> Tensor:
    height, width = target_cpu.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    output_size = (max(1, round(height * scale)), max(1, round(width * scale)))
    return F.interpolate(
        target_cpu[None].to(device),
        size=output_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def sampled_target(target: Tensor, points: Tensor) -> Tensor:
    height, width = target.shape[-2:]
    x_indices = (points[:, 0] * width).long().clamp(0, width - 1)
    y_indices = (points[:, 1] * height).long().clamp(0, height - 1)
    flat_indices = y_indices * width + x_indices
    return target.reshape(3, -1)[:, flat_indices][None].float()


def reconstruction_loss(prediction: Tensor, target: Tensor, include_edges: bool) -> Tensor:
    loss = F.l1_loss(prediction, target) + 0.25 * F.mse_loss(prediction, target)
    if include_edges:
        pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_x = target[..., :, 1:] - target[..., :, :-1]
        pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + 0.08 * (F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y))
    return loss


@torch.no_grad()
def preview_render(
    renderer: SoftShapeRenderer,
    sampled_renderer: CudaSampledShapeRenderer | None,
    raw_shapes: Tensor,
    raw_background: Tensor,
    target_cpu: Tensor,
    active_shapes: int,
    preview_side: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    target = resized_target(target_cpu, preview_side, device)
    height, width = target.shape[-2:]
    if active_shapes == 0:
        prediction = raw_background.sigmoid()[:, :, None, None].expand(-1, -1, height, width)
    elif sampled_renderer is not None:
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        prediction = sampled_renderer(
            raw_shapes[:, :active_shapes],
            raw_background,
            sample_points=points,
            canvas_size=(height, width),
            use_presence=False,
            softness_px=0.7,
        ).reshape(1, 3, height, width)
    else:
        prediction = renderer(
            raw_shapes[:, :active_shapes],
            raw_background,
            height,
            width,
            use_presence=False,
            softness_px=0.7,
        )
    return target, prediction


@torch.no_grad()
def initialize_new_shapes(
    renderer: SoftShapeRenderer,
    sampled_renderer: CudaSampledShapeRenderer | None,
    raw_shapes: Tensor,
    raw_background: Tensor,
    target_cpu: Tensor,
    previous_count: int,
    active_count: int,
    preview_side: int,
    device: torch.device,
) -> None:
    new_count = active_count - previous_count
    if new_count <= 0:
        return
    target, prediction = preview_render(
        renderer,
        sampled_renderer,
        raw_shapes,
        raw_background,
        target_cpu,
        previous_count,
        preview_side,
        device,
    )
    error = (target - prediction).abs().mean(dim=1)[0]
    weights = error.flatten().square() + 1e-5
    indices = torch.multinomial(weights, new_count, replacement=new_count > weights.numel())
    height, width = error.shape
    y_indices = torch.div(indices, width, rounding_mode="floor")
    x_indices = indices % width
    centers = torch.stack(
        ((x_indices + 0.5) / width, (y_indices + 0.5) / height), dim=-1
    ).clamp(0.001, 0.999)
    colors = target[0, :, y_indices, x_indices].transpose(0, 1).clamp(0.01, 0.99)

    slots = raw_shapes[0, previous_count:active_count]
    slots.zero_()
    shape_types = torch.arange(previous_count, active_count, device=device) % 3
    slots[:, 0:3] = -2.0
    slots[:, 0:3].scatter_(1, shape_types[:, None], 2.0)
    slots[:, 3:5] = inverse_sigmoid(centers).to(device)

    desired_size = max(0.025, min(0.18, 0.30 / math.sqrt(max(active_count, 1) / 8.0)))
    sizes = torch.empty(new_count, 2, device=device).uniform_(desired_size * 0.65, desired_size * 1.35)
    size_unit = (sizes - renderer.min_size) / (renderer.max_size - renderer.min_size)
    slots[:, 5:7] = inverse_sigmoid(size_unit).to(device)
    slots[:, 7] = torch.atanh(torch.empty(new_count, device=device).uniform_(-0.9, 0.9))
    slots[:, 8:11] = inverse_sigmoid(colors).to(device)
    slots[:, 11] = 8.0


@torch.no_grad()
def error_distribution(
    renderer: SoftShapeRenderer,
    sampled_renderer: CudaSampledShapeRenderer | None,
    raw_shapes: Tensor,
    raw_background: Tensor,
    target_cpu: Tensor,
    active_shapes: int,
    device: torch.device,
) -> tuple[Tensor, int, int]:
    target, prediction = preview_render(
        renderer,
        sampled_renderer,
        raw_shapes,
        raw_background,
        target_cpu,
        active_shapes,
        256,
        device,
    )
    error = (target - prediction).abs().mean(dim=1)[0]
    height, width = error.shape
    return error.flatten().square() + 1e-5, height, width


@torch.no_grad()
def sample_error_biased_points(
    weights: Tensor,
    height: int,
    width: int,
    sample_count: int,
    device: torch.device,
) -> Tensor:
    biased_count = sample_count // 2
    indices = torch.multinomial(weights, biased_count, replacement=True)
    y_indices = torch.div(indices, width, rounding_mode="floor")
    x_indices = indices % width
    jitter = torch.rand(biased_count, 2, device=device)
    biased = torch.stack(
        ((x_indices + jitter[:, 0]) / width, (y_indices + jitter[:, 1]) / height), dim=-1
    )
    uniform = torch.rand(sample_count - biased_count, 2, device=device)
    return torch.cat((biased, uniform), dim=0)


def refine_initialized(
    image_path: Path,
    output_dir: Path,
    initial_shapes: Tensor,
    initial_background: Tensor,
    *,
    steps: int,
    sample_count: int,
    error_refresh: int,
    preview_side: int = 768,
    learning_rate: float = 0.035,
    min_size: float = 0.005,
    max_size: float = 0.5,
    seed: int = 11,
) -> dict[str, object]:
    """Refine a model prediction in-process without reloading Python or the checkpoint."""
    if steps < 1 or sample_count < 1 or error_refresh < 1:
        raise ValueError("steps, sample_count, and error_refresh must be positive")
    if initial_shapes.ndim != 3 or initial_shapes.shape[0] != 1:
        raise ValueError("initial_shapes must have shape [1, shapes, parameters]")
    if initial_shapes.shape[-1] != PARAM_COUNT or initial_background.shape != (1, 3):
        raise ValueError("Invalid initializer parameter shapes")

    started = time.perf_counter()
    torch.manual_seed(seed)
    device = initial_shapes.device
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(image_path)
    target_cpu = pil_to_tensor(image)
    target_native = target_cpu.to(
        device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    original_width, original_height = image.size
    preview_renderer = SoftShapeRenderer(
        min_size=min_size,
        max_size=max_size,
        softness_px=1.0,
        chunk_size=32,
        hard_types=True,
        learn_shape_types=False,
    ).to(device)
    sampled_renderer = CudaSampledShapeRenderer(
        min_size=preview_renderer.min_size,
        max_size=preview_renderer.max_size,
        softness_px=preview_renderer.softness_px,
        allow_fallback=True,
    ).to(device)
    raw_shapes = nn.Parameter(initial_shapes.detach().float().clone())
    raw_background = nn.Parameter(initial_background.detach().float().clone())
    with torch.no_grad():
        raw_shapes[..., 11] = 8.0

    gpu_guard = GpuGuard(enabled=device.type == "cuda")
    gpu_guard.check(force=True)
    optimizer = torch.optim.Adam((raw_shapes, raw_background), lr=learning_rate)
    active_shapes = raw_shapes.shape[1]
    error_state: tuple[Tensor, int, int] | None = None
    for step in range(steps):
        gpu_guard.callback(step=step)
        progress = step / max(1, steps - 1)
        current_lr = learning_rate * (
            0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        if error_state is None or step % error_refresh == 0:
            error_state = error_distribution(
                preview_renderer,
                sampled_renderer,
                raw_shapes,
                raw_background,
                target_cpu,
                active_shapes,
                device,
            )
        points = sample_error_biased_points(*error_state, sample_count, device)
        target = sampled_target(target_native, points)
        prediction = sampled_renderer(
            raw_shapes,
            raw_background,
            sample_points=points,
            canvas_size=(original_height, original_width),
            use_presence=False,
            softness_px=max(0.65, 1.5 - progress),
        )
        loss = reconstruction_loss(prediction, target, include_edges=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_shapes, raw_background), 5.0)
        optimizer.step()

    target_preview, prediction_preview = preview_render(
        preview_renderer,
        sampled_renderer,
        raw_shapes,
        raw_background,
        target_cpu,
        active_shapes,
        preview_side,
        device,
    )
    metrics = image_metrics(target_preview, prediction_preview)
    save_comparison(
        target_preview,
        prediction_preview,
        output_dir / f"stage_{active_shapes:03d}.png",
        label=f"target | reconstruction | 3x error    {active_shapes} shapes",
    )
    save_svg(
        output_dir / "result.svg",
        raw_shapes,
        raw_background,
        original_width,
        original_height,
        min_size=preview_renderer.min_size,
        max_size=preview_renderer.max_size,
        use_presence=False,
        presence_threshold=0.0,
        title=image_path.name,
    )
    torch.save(
        {
            "raw_shapes": raw_shapes.detach().cpu(),
            "raw_background": raw_background.detach().cpu(),
            "width": original_width,
            "height": original_height,
            "min_size": preview_renderer.min_size,
            "max_size": preview_renderer.max_size,
            "source": str(image_path.resolve()),
        },
        output_dir / "fit.pt",
    )
    gpu_guard.check(force=True)
    elapsed = time.perf_counter() - started
    summary: dict[str, object] = {
        "source": str(image_path.resolve()),
        "native_size": [original_width, original_height],
        "active_shapes": active_shapes,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "sampled_backend": sampled_renderer.last_backend or "torch",
        "history": [{"stage": 1, "shapes": active_shapes, "seconds": elapsed, **metrics}],
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
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressively fit up to 450 SVG shapes to an image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fit"))
    parser.add_argument("--max-shapes", type=int, default=450)
    parser.add_argument("--stages", type=integer_list, default=integer_list("16,32,64,128,256,450"))
    parser.add_argument("--steps-per-stage", type=int, default=120)
    parser.add_argument("--stage-sides", type=integer_list, default=integer_list("160,224,320,384,512,768"))
    parser.add_argument("--sample-count", type=int, default=32_768)
    parser.add_argument("--error-refresh", type=int, default=8)
    parser.add_argument("--sampled-backend", choices=("auto", "cuda", "torch"), default="auto")
    parser.add_argument("--initializer-checkpoint", type=Path)
    parser.add_argument("--initializer-parameters", type=Path)
    parser.add_argument("--initializer-corrections", type=int, default=1)
    parser.add_argument("--preview-side", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-gpu-guard", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if args.stages[-1] > args.max_shapes:
        parser.error("No stage may exceed --max-shapes")
    if args.initializer_corrections < 0:
        parser.error("--initializer-corrections cannot be negative")
    if args.initializer_checkpoint is not None and args.initializer_parameters is not None:
        parser.error("Use only one initializer source")
    if len(args.stage_sides) < len(args.stages):
        args.stage_sides.extend([args.stage_sides[-1]] * (len(args.stages) - len(args.stage_sides)))

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    gpu_guard = GpuGuard(enabled=device.type == "cuda" and not args.no_gpu_guard)
    gpu_guard.check(force=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb(args.image)
    target_cpu = pil_to_tensor(image)
    target_native = target_cpu.to(
        device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    original_width, original_height = image.size

    renderer = SoftShapeRenderer(
        min_size=0.005,
        max_size=0.5,
        softness_px=1.0,
        chunk_size=32,
        hard_types=True,
    ).to(device)
    preview_renderer = SoftShapeRenderer(
        min_size=renderer.min_size,
        max_size=renderer.max_size,
        softness_px=renderer.softness_px,
        chunk_size=32,
        hard_types=True,
        learn_shape_types=False,
    ).to(device)
    sampled_renderer: CudaSampledShapeRenderer | None = None
    if args.sampled_backend != "torch" and device.type == "cuda":
        if args.sampled_backend == "cuda" and not cuda_renderer_available(probe_kernel=True):
            raise RuntimeError(
                f"Custom CUDA renderer requested but unavailable: {cuda_renderer_unavailable_reason()}"
            )
        sampled_renderer = CudaSampledShapeRenderer(
            min_size=renderer.min_size,
            max_size=renderer.max_size,
            softness_px=renderer.softness_px,
            allow_fallback=args.sampled_backend == "auto",
        ).to(device)
    raw_shapes = nn.Parameter(torch.zeros(1, args.max_shapes, PARAM_COUNT, device=device))
    mean_color = target_cpu.mean(dim=(1, 2)).clamp(0.01, 0.99)
    raw_background = nn.Parameter(inverse_sigmoid(mean_color).to(device)[None])

    started = time.perf_counter()
    previous_count = 0
    if args.initializer_parameters is not None:
        initializer = torch.load(args.initializer_parameters, map_location=device, weights_only=False)
        initial_shapes = initializer.get("raw_shapes")
        initial_background = initializer.get("raw_background")
        if not isinstance(initial_shapes, Tensor) or initial_shapes.shape != raw_shapes.shape:
            raise ValueError(
                f"Initializer raw_shapes must have shape {tuple(raw_shapes.shape)}"
            )
        if not isinstance(initial_background, Tensor) or initial_background.shape != raw_background.shape:
            raise ValueError(
                f"Initializer raw_background must have shape {tuple(raw_background.shape)}"
            )
        with torch.no_grad():
            raw_shapes.copy_(initial_shapes)
            raw_shapes[..., 11] = 8.0
            raw_background.copy_(initial_background)
        previous_count = args.max_shapes
    elif args.initializer_checkpoint is not None:
        loaded = load_vectorizer_checkpoint(args.initializer_checkpoint, device)
        if args.max_shapes > loaded.trained_shapes:
            raise ValueError(
                f"Initializer supports {loaded.trained_shapes} shapes, fewer than requested "
                f"{args.max_shapes}"
            )
        encoder_image = resize_max_side(image, 512)
        encoder_tensor = pil_to_tensor(encoder_image)[None].to(device)
        with torch.inference_mode():
            if isinstance(loaded.model, ShapeVectorizerV2):
                corrections = min(args.initializer_corrections, loaded.model.correction_steps)
                initial_shapes, initial_background = loaded.model(
                    encoder_tensor,
                    args.max_shapes,
                    renderer=sampled_renderer or preview_renderer,
                    correction_steps=corrections,
                )
            else:
                initial_shapes, initial_background = loaded.model(
                    encoder_tensor,
                    args.max_shapes,
                )
            raw_shapes.copy_(initial_shapes)
            raw_shapes[..., 11] = 8.0
            raw_background.copy_(initial_background)
        previous_count = args.max_shapes
        del encoder_tensor, loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history: list[dict[str, float | int | str]] = []
    for stage_index, active_shapes in enumerate(args.stages):
        initialize_new_shapes(
            preview_renderer,
            sampled_renderer,
            raw_shapes,
            raw_background,
            target_cpu,
            previous_count,
            active_shapes,
            min(args.preview_side, 384),
            device,
        )
        optimizer = torch.optim.Adam((raw_shapes, raw_background), lr=args.learning_rate)
        stage_side = args.stage_sides[stage_index]
        sampled_threshold = 64 if sampled_renderer is not None else 128
        use_sampled_loss = active_shapes >= sampled_threshold or stage_side > 384
        fixed_target = None if use_sampled_loss else resized_target(target_cpu, stage_side, device)
        error_state: tuple[Tensor, int, int] | None = None
        stage_started = time.perf_counter()

        for step in range(args.steps_per_stage):
            gpu_guard.callback(step=step)
            progress = step / max(1, args.steps_per_stage - 1)
            learning_rate = args.learning_rate * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)

            if use_sampled_loss:
                if error_state is None or step % args.error_refresh == 0:
                    error_state = error_distribution(
                        preview_renderer,
                        sampled_renderer,
                        raw_shapes,
                        raw_background,
                        target_cpu,
                        active_shapes,
                        device,
                    )
                points = sample_error_biased_points(
                    *error_state,
                    args.sample_count,
                    device,
                )
                target = sampled_target(target_native, points)
                sampled_softness = max(0.65, 1.5 - progress)
                if sampled_renderer is not None:
                    prediction = sampled_renderer(
                        raw_shapes[:, :active_shapes],
                        raw_background,
                        sample_points=points,
                        canvas_size=(original_height, original_width),
                        use_presence=False,
                        softness_px=sampled_softness,
                    )
                    backend_name = sampled_renderer.last_backend or "cuda"
                else:
                    prediction = renderer(
                        raw_shapes[:, :active_shapes],
                        raw_background,
                        sample_points=points,
                        canvas_size=(original_height, original_width),
                        use_presence=False,
                        softness_px=sampled_softness,
                    )
                    backend_name = "torch"
                loss = reconstruction_loss(prediction, target, include_edges=False)
                loss_kind = f"native sampled/{backend_name}"
            else:
                assert fixed_target is not None
                target = fixed_target
                prediction = renderer(
                    raw_shapes[:, :active_shapes],
                    raw_background,
                    target.shape[-2],
                    target.shape[-1],
                    use_presence=False,
                    softness_px=max(0.65, 1.5 - progress),
                )
                loss = reconstruction_loss(prediction, target, include_edges=True)
                loss_kind = f"{target.shape[-1]}x{target.shape[-2]} full"

            loss.backward()
            torch.nn.utils.clip_grad_norm_((raw_shapes, raw_background), 5.0)
            optimizer.step()

            completed = step + 1
            if completed == 1 or completed % args.log_every == 0 or completed == args.steps_per_stage:
                print(
                    f"stage={stage_index + 1}/{len(args.stages)} shapes={active_shapes} "
                    f"step={completed}/{args.steps_per_stage} loss={float(loss.detach()):.5f} "
                    f"mode={loss_kind}"
                )

        target_preview, prediction_preview = preview_render(
            preview_renderer,
            sampled_renderer,
            raw_shapes,
            raw_background,
            target_cpu,
            active_shapes,
            args.preview_side,
            device,
        )
        metrics = image_metrics(target_preview, prediction_preview)
        history.append(
            {
                "stage": stage_index + 1,
                "shapes": active_shapes,
                "seconds": time.perf_counter() - stage_started,
                **metrics,
            }
        )
        save_comparison(
            target_preview,
            prediction_preview,
            args.output_dir / f"stage_{active_shapes:03d}.png",
            label=f"target | reconstruction | 3x error    {active_shapes} shapes",
        )
        previous_count = active_shapes

    active_shapes = args.stages[-1]
    gpu_guard.check(force=True)
    save_svg(
        args.output_dir / "result.svg",
        raw_shapes[:, :active_shapes],
        raw_background,
        original_width,
        original_height,
        min_size=renderer.min_size,
        max_size=renderer.max_size,
        use_presence=False,
        presence_threshold=0.0,
        title=args.image.name,
    )
    torch.save(
        {
            "raw_shapes": raw_shapes[:, :active_shapes].detach().cpu(),
            "raw_background": raw_background.detach().cpu(),
            "width": original_width,
            "height": original_height,
            "min_size": renderer.min_size,
            "max_size": renderer.max_size,
            "source": str(args.image.resolve()),
        },
        args.output_dir / "fit.pt",
    )
    summary = {
        "source": str(args.image.resolve()),
        "native_size": [original_width, original_height],
        "active_shapes": active_shapes,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "sampled_backend": (
            sampled_renderer.last_backend if sampled_renderer is not None else "torch"
        ),
        "initializer_checkpoint": (
            str(args.initializer_checkpoint.resolve()) if args.initializer_checkpoint else None
        ),
        "initializer_parameters": (
            str(args.initializer_parameters.resolve()) if args.initializer_parameters else None
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
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {args.output_dir / 'result.svg'}")


if __name__ == "__main__":
    main()
