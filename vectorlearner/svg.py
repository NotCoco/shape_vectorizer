from __future__ import annotations

import html
import math
from pathlib import Path

import torch
from torch import Tensor

from .renderer import SHAPE_NAMES, decode_shapes


def _color(rgb: Tensor) -> str:
    values = (rgb.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.int64).tolist()
    return "#{:02x}{:02x}{:02x}".format(*values)


def svg_string(
    raw_shapes: Tensor,
    raw_background: Tensor,
    width: int,
    height: int,
    *,
    min_size: float = 0.01,
    max_size: float = 0.5,
    presence_threshold: float = 0.25,
    use_presence: bool = True,
    title: str = "Vector Shape Learner output",
) -> str:
    if raw_shapes.ndim == 3:
        raw_shapes = raw_shapes[0]
    if raw_background.ndim == 2:
        raw_background = raw_background[0]
    decoded = decode_shapes(
        raw_shapes[None, ...],
        min_size=min_size,
        max_size=max_size,
        hard_types=True,
    )
    background = raw_background.sigmoid()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        f"  <title>{html.escape(title)}</title>",
        f'  <rect width="{width}" height="{height}" fill="{_color(background)}"/>',
    ]

    for index in range(raw_shapes.shape[0]):
        presence = float(decoded.presence[0, index].detach().cpu()) if use_presence else 1.0
        if presence < presence_threshold:
            continue
        shape_index = int(decoded.type_weights[0, index].argmax().detach().cpu())
        shape_name = SHAPE_NAMES[shape_index]
        center_x = float(decoded.centers[0, index, 0].detach().cpu()) * width
        center_y = float(decoded.centers[0, index, 1].detach().cpu()) * height
        shape_width = float(decoded.sizes[0, index, 0].detach().cpu()) * width
        shape_height = float(decoded.sizes[0, index, 1].detach().cpu()) * height
        angle = math.degrees(float(decoded.angles[0, index].detach().cpu()))
        color = _color(decoded.colors[0, index])
        opacity = min(1.0, max(0.0, presence))
        common = f'fill="{color}" fill-opacity="{opacity:.4f}"'

        if shape_name == "ellipse":
            lines.append(
                f'  <ellipse cx="{center_x:.3f}" cy="{center_y:.3f}" '
                f'rx="{shape_width / 2:.3f}" ry="{shape_height / 2:.3f}" {common} '
                f'transform="rotate({angle:.3f} {center_x:.3f} {center_y:.3f})"/>'
            )
        elif shape_name == "rectangle":
            lines.append(
                f'  <rect x="{center_x - shape_width / 2:.3f}" '
                f'y="{center_y - shape_height / 2:.3f}" width="{shape_width:.3f}" '
                f'height="{shape_height:.3f}" {common} '
                f'transform="rotate({angle:.3f} {center_x:.3f} {center_y:.3f})"/>'
            )
        else:
            local_points = ((0.0, -0.5), (-0.5, 0.5), (0.5, 0.5))
            cos_angle = math.cos(math.radians(angle))
            sin_angle = math.sin(math.radians(angle))
            points: list[str] = []
            for local_x, local_y in local_points:
                x_value = local_x * shape_width
                y_value = local_y * shape_height
                x_rotated = cos_angle * x_value - sin_angle * y_value + center_x
                y_rotated = sin_angle * x_value + cos_angle * y_value + center_y
                points.append(f"{x_rotated:.3f},{y_rotated:.3f}")
            lines.append(f'  <polygon points="{" ".join(points)}" {common}/>' )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def save_svg(path: str | Path, *args: object, **kwargs: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg_string(*args, **kwargs), encoding="utf-8")
