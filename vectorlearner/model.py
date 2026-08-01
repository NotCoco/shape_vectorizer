from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .renderer import PARAM_COUNT, inverse_sigmoid


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(16, out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class ShapeVectorizer(nn.Module):
    """Variable-resolution image encoder with SVG-object query decoding."""

    def __init__(
        self,
        max_shapes: int = 450,
        hidden_dim: int = 128,
        decoder_layers: int = 2,
        attention_heads: int = 4,
        max_feature_side: int = 16,
        min_size: float = 0.01,
        max_size: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.max_shapes = max_shapes
        self.hidden_dim = hidden_dim
        self.max_feature_side = max_feature_side
        self.min_size = min_size
        self.max_size = max_size

        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
        )
        self.feature_projection = nn.Conv2d(192, hidden_dim, kernel_size=1)
        self.position_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.queries = nn.Embedding(max_shapes, hidden_dim)
        self.register_buffer("base_shape_params", self._base_shape_parameters(max_shapes))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.shape_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, PARAM_COUNT),
        )
        self.background_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self._initialize_heads()

    @staticmethod
    def _radical_inverse(index: int, base: int) -> float:
        result = 0.0
        fraction = 1.0 / base
        while index:
            result += fraction * (index % base)
            index //= base
            fraction /= base
        return result

    def _base_shape_parameters(self, count: int) -> Tensor:
        base = torch.zeros(count, PARAM_COUNT)
        shape_indices = torch.arange(count)
        base[:, 0:3] = -1.5
        base[shape_indices, shape_indices % 3] = 1.5
        centers = torch.tensor(
            [
                (self._radical_inverse(index + 1, 2), self._radical_inverse(index + 1, 3))
                for index in range(count)
            ],
            dtype=torch.float32,
        ).clamp(0.02, 0.98)
        base[:, 3:5] = inverse_sigmoid(centers)
        base[:, 11] = 8.0
        return base

    def _initialize_heads(self) -> None:
        final = self.shape_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)
        final.bias.data[5:7] = 0.0
        final.bias.data[11] = 0.0

        background_final = self.background_head[-1]
        assert isinstance(background_final, nn.Linear)
        nn.init.zeros_(background_final.weight)
        nn.init.zeros_(background_final.bias)

    def _memory(self, image: Tensor) -> Tensor:
        features = self.feature_projection(self.backbone(image))
        target_height = min(features.shape[-2], self.max_feature_side)
        target_width = min(features.shape[-1], self.max_feature_side)
        if (target_height, target_width) != features.shape[-2:]:
            features = F.adaptive_avg_pool2d(features, (target_height, target_width))

        ys = torch.linspace(-1.0, 1.0, target_height, device=image.device, dtype=image.dtype)
        xs = torch.linspace(-1.0, 1.0, target_width, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        position_features = self.position_projection(positions)

        memory = features.flatten(2).transpose(1, 2)
        return memory + position_features[None, :, :]

    def forward(self, image: Tensor, shape_count: int | None = None) -> tuple[Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        count = self.max_shapes if shape_count is None else shape_count
        if not 1 <= count <= self.max_shapes:
            raise ValueError(f"shape_count must be within 1..{self.max_shapes}")

        memory = self._memory(image)
        queries = self.queries.weight[:count][None, :, :].expand(image.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        base_parameters = self.base_shape_params[:count].clone()
        desired_size = min(0.34, max(0.045, math.sqrt(1.6 / count)))
        size_unit = (desired_size - self.min_size) / (self.max_size - self.min_size)
        base_parameters[:, 5:7] = inverse_sigmoid(size_unit).to(base_parameters)
        raw_shapes = self.shape_head(decoded) + base_parameters[None]

        base_centers = base_parameters[:, 3:5].sigmoid()
        sample_grid = base_centers.mul(2.0).sub(1.0)[None, None].expand(image.shape[0], -1, -1, -1)
        sampled_colors = F.grid_sample(
            image,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, :, 0, :].transpose(1, 2).clamp(1e-4, 1.0 - 1e-4)
        color_offset = torch.zeros_like(raw_shapes)
        color_offset[..., 8:11] = torch.logit(sampled_colors)
        raw_shapes = raw_shapes + color_offset
        image_mean = image.mean(dim=(2, 3)).clamp(1e-4, 1.0 - 1e-4)
        raw_background = self.background_head(memory.mean(dim=1)) + torch.logit(image_mean)
        return raw_shapes, raw_background

    def config(self) -> dict[str, int | float]:
        decoder_layers = len(self.decoder.layers)
        attention_heads = self.decoder.layers[0].self_attn.num_heads
        return {
            "max_shapes": self.max_shapes,
            "hidden_dim": self.hidden_dim,
            "decoder_layers": decoder_layers,
            "attention_heads": attention_heads,
            "max_feature_side": self.max_feature_side,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }
