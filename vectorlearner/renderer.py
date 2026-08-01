from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SHAPE_NAMES = ("ellipse", "rectangle", "triangle")
PARAM_COUNT = 12


@dataclass(frozen=True)
class DecodedShapes:
    type_weights: Tensor
    centers: Tensor
    sizes: Tensor
    angles: Tensor
    colors: Tensor
    presence: Tensor


def inverse_sigmoid(value: Tensor | float, eps: float = 1e-5) -> Tensor:
    value_tensor = torch.as_tensor(value).clamp(eps, 1.0 - eps)
    return torch.logit(value_tensor)


def decode_shapes(
    raw: Tensor,
    min_size: float = 0.01,
    max_size: float = 0.5,
    type_temperature: float = 0.7,
    hard_types: bool = True,
) -> DecodedShapes:
    if raw.ndim != 3 or raw.shape[-1] != PARAM_COUNT:
        raise ValueError(f"Expected [batch, shapes, {PARAM_COUNT}], got {tuple(raw.shape)}")

    type_soft = F.softmax(raw[..., 0:3] / type_temperature, dim=-1)
    if hard_types:
        type_hard = F.one_hot(type_soft.argmax(dim=-1), len(SHAPE_NAMES)).to(type_soft.dtype)
        type_weights = type_hard + type_soft - type_soft.detach()
    else:
        type_weights = type_soft

    centers = raw[..., 3:5].sigmoid()
    sizes = min_size + (max_size - min_size) * raw[..., 5:7].sigmoid()
    angles = math.pi * raw[..., 7].tanh()
    colors = raw[..., 8:11].sigmoid()
    presence = raw[..., 11].sigmoid()
    return DecodedShapes(type_weights, centers, sizes, angles, colors, presence)


class SoftShapeRenderer(nn.Module):
    """Differentiable rasterizer for simple SVG-compatible filled shapes.

    Coordinates and sizes are normalized to the canvas. Rendering can target a
    complete raster grid or only sampled points, which keeps native-resolution
    fitting practical without allocating one full mask per shape.
    """

    def __init__(
        self,
        min_size: float = 0.01,
        max_size: float = 0.5,
        softness_px: float = 1.0,
        chunk_size: int = 16,
        type_temperature: float = 0.7,
        hard_types: bool = True,
        learn_shape_types: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < min_size < max_size <= 1.0:
            raise ValueError("Expected 0 < min_size < max_size <= 1")
        self.min_size = min_size
        self.max_size = max_size
        self.softness_px = softness_px
        self.chunk_size = chunk_size
        self.type_temperature = type_temperature
        self.hard_types = hard_types
        self.learn_shape_types = learn_shape_types
        self._grid_cache: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def grid(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (height, width, device, dtype)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        if len(self._grid_cache) >= 8:
            self._grid_cache.clear()
        self._grid_cache[key] = grid
        return grid

    def _masks(
        self,
        shapes: DecodedShapes,
        points: Tensor,
        canvas_size: tuple[int, int],
        softness_px: float,
    ) -> Tensor:
        height, width = canvas_size
        points = points[None, None, :, :]
        centers = shapes.centers[:, :, None, :]
        half_sizes = shapes.sizes[:, :, None, :] * 0.5

        delta = points - centers
        cos_angle = shapes.angles.cos()[:, :, None]
        sin_angle = shapes.angles.sin()[:, :, None]
        local_x = cos_angle * delta[..., 0] + sin_angle * delta[..., 1]
        local_y = -sin_angle * delta[..., 0] + cos_angle * delta[..., 1]
        local = torch.stack((local_x, local_y), dim=-1)

        pixel_scale = float(min(height, width))
        safe_half = half_sizes.clamp_min(1e-5)

        def ellipse_distance(local_points: Tensor, half: Tensor) -> Tensor:
            radius = torch.sqrt(((local_points / half) ** 2).sum(dim=-1) + 1e-8)
            return (1.0 - radius) * half.min(dim=-1).values * pixel_scale

        def rectangle_distance(local_points: Tensor, half: Tensor) -> Tensor:
            rectangle_q = local_points.abs() - half
            outside = torch.sqrt((F.relu(rectangle_q) ** 2).sum(dim=-1) + 1e-8)
            inside = torch.minimum(
                torch.maximum(rectangle_q[..., 0], rectangle_q[..., 1]),
                torch.zeros_like(rectangle_q[..., 0]),
            )
            return -(outside + inside) * pixel_scale

        def triangle_distance(local_points: Tensor, half: Tensor) -> Tensor:
            triangle_points = local_points / half
            vertices = local_points.new_tensor(((0.0, -1.0), (-1.0, 1.0), (1.0, 1.0)))
            edge_distances: list[Tensor] = []
            for index in range(3):
                vertex_a = vertices[index]
                vertex_b = vertices[(index + 1) % 3]
                edge = vertex_b - vertex_a
                relative = triangle_points - vertex_a
                cross = edge[0] * relative[..., 1] - edge[1] * relative[..., 0]
                edge_distances.append(-cross / edge.norm().clamp_min(1e-5))
            distance = torch.stack(edge_distances, dim=-1).min(dim=-1).values
            return distance * half.min(dim=-1).values * pixel_scale

        softness = max(softness_px, 1e-3)
        if not self.learn_shape_types:
            batch_size, shape_count, point_count = local.shape[:3]
            flat_local = local.reshape(batch_size * shape_count, point_count, 2)
            flat_half = safe_half.reshape(batch_size * shape_count, 1, 2)
            flat_types = shapes.type_weights.argmax(dim=-1).reshape(-1)
            flat_masks = local.new_zeros((batch_size * shape_count, point_count))
            distance_functions = (ellipse_distance, rectangle_distance, triangle_distance)
            for type_index, distance_function in enumerate(distance_functions):
                indices = torch.nonzero(flat_types == type_index, as_tuple=False).flatten()
                selected_distance = distance_function(
                    flat_local.index_select(0, indices),
                    flat_half.index_select(0, indices),
                )
                flat_masks = flat_masks.index_copy(
                    0, indices, torch.sigmoid(selected_distance / softness)
                )
            return flat_masks.reshape(batch_size, shape_count, point_count)

        ellipse = ellipse_distance(local, safe_half)
        rectangle = rectangle_distance(local, safe_half)
        triangle = triangle_distance(local, safe_half)
        distances = torch.stack((ellipse, rectangle, triangle), dim=-1)
        primitive_masks = torch.sigmoid(distances / softness)
        return (primitive_masks * shapes.type_weights[:, :, None, :]).sum(dim=-1)

    def forward(
        self,
        raw_shapes: Tensor,
        raw_background: Tensor,
        height: int | None = None,
        width: int | None = None,
        *,
        sample_points: Tensor | None = None,
        canvas_size: tuple[int, int] | None = None,
        use_presence: bool = True,
        softness_px: float | None = None,
    ) -> Tensor:
        if sample_points is None:
            if height is None or width is None:
                raise ValueError("height and width are required for full-grid rendering")
            canvas_size = (height, width)
            points = self.grid(height, width, raw_shapes.device, raw_shapes.dtype)
        else:
            if canvas_size is None:
                raise ValueError("canvas_size is required when rendering sampled points")
            points = sample_points.to(device=raw_shapes.device, dtype=raw_shapes.dtype)
            if points.ndim != 2 or points.shape[-1] != 2:
                raise ValueError("sample_points must have shape [point_count, 2]")

        if raw_background.ndim != 2 or raw_background.shape != (raw_shapes.shape[0], 3):
            raise ValueError("raw_background must have shape [batch, 3]")

        decoded = decode_shapes(
            raw_shapes,
            min_size=self.min_size,
            max_size=self.max_size,
            type_temperature=self.type_temperature,
            hard_types=self.hard_types,
        )
        background = raw_background.sigmoid()
        canvas = background[:, :, None].expand(-1, -1, points.shape[0])
        softness = self.softness_px if softness_px is None else softness_px

        for start in range(0, raw_shapes.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, raw_shapes.shape[1])
            chunk = DecodedShapes(
                decoded.type_weights[:, start:end],
                decoded.centers[:, start:end],
                decoded.sizes[:, start:end],
                decoded.angles[:, start:end],
                decoded.colors[:, start:end],
                decoded.presence[:, start:end],
            )
            masks = self._masks(chunk, points, canvas_size, softness)
            if use_presence:
                masks = masks * chunk.presence[:, :, None]
            alpha = masks[:, :, None, :]
            one_minus_alpha = 1.0 - alpha
            suffix_inclusive = torch.flip(
                torch.cumprod(torch.flip(one_minus_alpha, dims=(1,)), dim=1), dims=(1,)
            )
            suffix_after = torch.cat(
                (suffix_inclusive[:, 1:], torch.ones_like(suffix_inclusive[:, :1])), dim=1
            )
            contribution = (chunk.colors[:, :, :, None] * alpha * suffix_after).sum(dim=1)
            canvas = canvas * suffix_inclusive[:, 0] + contribution

        if sample_points is None:
            return canvas.reshape(raw_shapes.shape[0], 3, height, width)
        return canvas


def random_raw_scenes(
    batch_size: int,
    shape_count: int,
    device: torch.device,
    *,
    min_size: float = 0.01,
    max_size: float = 0.5,
    min_scene_size: float = 0.04,
    max_scene_size: float = 0.35,
) -> tuple[Tensor, Tensor]:
    """Generate SVG-parameter scenes for on-the-fly synthetic training."""
    raw = torch.zeros(batch_size, shape_count, PARAM_COUNT, device=device)
    shape_types = torch.randint(0, len(SHAPE_NAMES), (batch_size, shape_count), device=device)
    raw[..., 0:3] = -6.0
    raw[..., 0:3].scatter_(-1, shape_types[..., None], 6.0)

    centers = torch.empty(batch_size, shape_count, 2, device=device).uniform_(0.03, 0.97)
    raw[..., 3:5] = inverse_sigmoid(centers).to(device)

    lower = max(min_scene_size, min_size + 1e-4)
    upper = min(max_scene_size, max_size - 1e-4)
    sizes = torch.empty(batch_size, shape_count, 2, device=device).uniform_(lower, upper)
    size_unit = (sizes - min_size) / (max_size - min_size)
    raw[..., 5:7] = inverse_sigmoid(size_unit).to(device)

    normalized_angles = torch.empty(batch_size, shape_count, device=device).uniform_(-0.95, 0.95)
    raw[..., 7] = torch.atanh(normalized_angles)
    colors = torch.empty(batch_size, shape_count, 3, device=device).uniform_(0.03, 0.97)
    raw[..., 8:11] = inverse_sigmoid(colors).to(device)
    raw[..., 11] = 8.0

    background = torch.empty(batch_size, 3, device=device).uniform_(0.03, 0.97)
    return raw, inverse_sigmoid(background).to(device)
