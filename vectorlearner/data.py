from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image.copy()
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


class RasterImageDataset(Dataset[Tensor]):
    def __init__(
        self,
        root: str | Path,
        crop_size: int = 256,
        min_crop_scale: float = 0.35,
        full_image_probability: float = 0.0,
        max_images: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.min_crop_scale = min_crop_scale
        if not 0.0 <= full_image_probability <= 1.0:
            raise ValueError("full_image_probability must be within 0..1")
        self.full_image_probability = full_image_probability
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.paths = paths[:max_images] if max_images else paths
        if not self.paths:
            raise FileNotFoundError(f"No supported images found below {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        image = load_rgb(self.paths[index])
        if random.random() < self.full_image_probability:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            return pil_to_tensor(resize_max_side(image, self.crop_size))
        short_side = min(image.size)
        crop_side = max(8, round(short_side * random.uniform(self.min_crop_scale, 1.0)))
        left = random.randint(0, max(0, image.width - crop_side))
        top = random.randint(0, max(0, image.height - crop_side))
        image = image.crop((left, top, left + crop_side, top + crop_side))
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        image = image.resize((self.crop_size, self.crop_size), Image.Resampling.LANCZOS)
        return pil_to_tensor(image)
