from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    tensor = tensor.detach().float().clamp(0.0, 1.0).cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def save_comparison(target: Tensor, prediction: Tensor, path: str | Path, label: str = "") -> None:
    target_image = tensor_to_pil(target)
    prediction_image = tensor_to_pil(prediction)
    difference = (target.detach().float() - prediction.detach().float()).abs()
    difference = (difference * 3.0).clamp(0.0, 1.0)
    difference_image = tensor_to_pil(difference)

    header = 34 if label else 0
    canvas = Image.new(
        "RGB",
        (target_image.width * 3, target_image.height + header),
        color=(24, 24, 24),
    )
    canvas.paste(target_image, (0, header))
    canvas.paste(prediction_image, (target_image.width, header))
    canvas.paste(difference_image, (target_image.width * 2, header))
    if label:
        ImageDraw.Draw(canvas).text((8, 9), label, fill=(240, 240, 240))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def image_metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    error = (target.detach().float() - prediction.detach().float()) ** 2
    mse = float(error.mean().cpu())
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    mae = float(error.sqrt().mean().cpu())
    return {"mse": mse, "mae": mae, "psnr": psnr}


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
    return torch.device(requested)

