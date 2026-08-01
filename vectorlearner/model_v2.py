from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .renderer import PARAM_COUNT, inverse_sigmoid


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.main(value) + self.skip(value))


@dataclass(frozen=True)
class FeaturePyramid:
    maps: tuple[Tensor, Tensor, Tensor]
    global_context: Tensor


class MultiScaleEncoder(nn.Module):
    """Small feature pyramid that keeps local detail instead of pooling to one 16x16 map."""

    def __init__(self, hidden_dim: int, base_channels: int) -> None:
        super().__init__()
        self.stem = ConvNormAct(3, base_channels, kernel_size=5, stride=2)
        self.stage_4 = ResidualDownBlock(base_channels, base_channels)
        self.stage_8 = ResidualDownBlock(base_channels, base_channels * 2)
        self.stage_16 = ResidualDownBlock(base_channels * 2, base_channels * 4)
        self.projections = nn.ModuleList(
            (
                nn.Conv2d(base_channels, hidden_dim, kernel_size=1),
                nn.Conv2d(base_channels * 2, hidden_dim, kernel_size=1),
                nn.Conv2d(base_channels * 4, hidden_dim, kernel_size=1),
            )
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, image: Tensor) -> FeaturePyramid:
        value = self.stem(image)
        level_4 = self.stage_4(value)
        level_8 = self.stage_8(level_4)
        level_16 = self.stage_16(level_8)
        maps = tuple(
            projection(level)
            for projection, level in zip(self.projections, (level_4, level_8, level_16))
        )
        assert len(maps) == 3
        global_context = self.global_projection(maps[-1].mean(dim=(2, 3)))
        return FeaturePyramid((maps[0], maps[1], maps[2]), global_context)


class SlotInteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )

    def forward(self, slots: Tensor) -> Tensor:
        normalized = self.attention_norm(slots)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + attended
        return slots + self.feedforward(self.feedforward_norm(slots))


class CorrectionEncoder(nn.Module):
    """Encodes target, current render, and signed error for a learned correction pass."""

    def __init__(self, hidden_dim: int, base_channels: int) -> None:
        super().__init__()
        middle = max(base_channels, hidden_dim // 2)
        self.network = nn.Sequential(
            ConvNormAct(9, base_channels, stride=2),
            ConvNormAct(base_channels, middle, stride=2),
            ConvNormAct(middle, hidden_dim, stride=2),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, target: Tensor, rendered: Tensor) -> tuple[Tensor, Tensor]:
        if target.shape != rendered.shape:
            raise ValueError("target and rendered images must have identical shapes")
        features = self.network(torch.cat((target, rendered, target - rendered), dim=1))
        return features, self.global_projection(features.mean(dim=(2, 3)))


@dataclass(frozen=True)
class V2Prediction:
    """A renderer-compatible prediction plus hidden state for another correction pass."""

    raw_shapes: Tensor
    raw_background: Tensor
    slot_features: Tensor


class ShapeVectorizerV2(nn.Module):
    """Coarse-to-detail, multi-resolution image-to-SVG initializer.

    The regular ``forward`` result has exactly the same raw 12-parameter shape
    representation as :class:`SoftShapeRenderer` and the existing SVG exporter.
    Supplying a renderer enables a small, fixed number of learned correction
    passes; these are neural-network passes rather than per-image optimization.
    """

    def __init__(
        self,
        max_shapes: int = 450,
        hidden_dim: int = 128,
        decoder_layers: int = 2,
        attention_heads: int = 4,
        encoder_base_channels: int = 32,
        correction_steps: int = 2,
        correction_side: int = 128,
        min_size: float = 0.01,
        max_size: float = 0.5,
    ) -> None:
        super().__init__()
        if max_shapes < 1:
            raise ValueError("max_shapes must be positive")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be positive")
        if correction_steps < 0:
            raise ValueError("correction_steps cannot be negative")
        if correction_side < 16:
            raise ValueError("correction_side must be at least 16")
        if not 0.0 < min_size < max_size <= 1.0:
            raise ValueError("Expected 0 < min_size < max_size <= 1")

        self.max_shapes = max_shapes
        self.hidden_dim = hidden_dim
        self.decoder_layers = decoder_layers
        self.attention_heads = attention_heads
        self.encoder_base_channels = encoder_base_channels
        self.correction_steps = correction_steps
        self.correction_side = correction_side
        self.min_size = min_size
        self.max_size = max_size

        anchor_centers, anchor_sizes, anchor_tiers = self._make_anchor_layout(max_shapes)
        self.register_buffer("anchor_centers", anchor_centers)
        self.register_buffer("anchor_sizes", anchor_sizes)
        self.register_buffer("anchor_tiers", anchor_tiers)
        self.register_buffer(
            "base_shape_params",
            self._make_base_shape_parameters(anchor_centers, anchor_sizes),
        )

        self.encoder = MultiScaleEncoder(hidden_dim, encoder_base_channels)
        self.local_feature_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
        )
        self.anchor_projection = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_embeddings = nn.Embedding(max_shapes, hidden_dim)
        self.slot_decoder = nn.ModuleList(
            SlotInteractionBlock(hidden_dim, attention_heads) for _ in range(decoder_layers)
        )
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

        self.correction_encoder = CorrectionEncoder(hidden_dim, encoder_base_channels)
        self.parameter_projection = nn.Sequential(
            nn.Linear(PARAM_COUNT, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.correction_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
        )
        self.correction_interaction = SlotInteractionBlock(hidden_dim, attention_heads)
        self.correction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, PARAM_COUNT),
        )
        self.background_correction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer(
            "correction_delta_scales",
            torch.tensor(
                (0.30, 0.30, 0.30, 0.50, 0.50, 0.35, 0.35, 0.25, 0.50, 0.50, 0.50, 0.35),
                dtype=torch.float32,
            ),
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

    @staticmethod
    def _tier_counts(count: int) -> tuple[int, int, int]:
        if count == 1:
            return 1, 0, 0
        if count == 2:
            return 1, 1, 0
        coarse = max(1, round(count * 32 / 450))
        medium = max(1, round(count * 96 / 450))
        if coarse + medium >= count:
            medium = max(1, count - coarse - 1)
        fine = count - coarse - medium
        return coarse, medium, fine

    def _make_anchor_layout(self, count: int) -> tuple[Tensor, Tensor, Tensor]:
        tier_counts = self._tier_counts(count)
        tier_sizes = (0.30, 0.13, 0.055)
        centers: list[tuple[float, float]] = []
        sizes: list[tuple[float, float]] = []
        tiers: list[int] = []
        for tier, tier_count in enumerate(tier_counts):
            offset = tier * 997
            for local_index in range(tier_count):
                sequence_index = offset + local_index + 1
                centers.append(
                    (
                        self._radical_inverse(sequence_index, 2),
                        self._radical_inverse(sequence_index, 3),
                    )
                )
                prior = min(self.max_size - 1e-4, max(self.min_size + 1e-4, tier_sizes[tier]))
                sizes.append((prior, prior))
                tiers.append(tier)
        return (
            torch.tensor(centers, dtype=torch.float32).clamp(0.02, 0.98),
            torch.tensor(sizes, dtype=torch.float32),
            torch.tensor(tiers, dtype=torch.long),
        )

    def _make_base_shape_parameters(self, centers: Tensor, sizes: Tensor) -> Tensor:
        count = centers.shape[0]
        base = torch.zeros(count, PARAM_COUNT, dtype=torch.float32)
        shape_indices = torch.arange(count)
        base[:, 0:3] = -1.5
        base[shape_indices, shape_indices % 3] = 1.5
        base[:, 3:5] = inverse_sigmoid(centers).to(base)
        size_unit = (sizes - self.min_size) / (self.max_size - self.min_size)
        base[:, 5:7] = inverse_sigmoid(size_unit).to(base)
        base[:, 11] = 2.5
        return base

    def _initialize_heads(self) -> None:
        nn.init.normal_(self.slot_embeddings.weight, std=0.02)
        for sequential in (self.shape_head, self.correction_head):
            final = sequential[-1]
            assert isinstance(final, nn.Linear)
            nn.init.normal_(final.weight, std=0.002)
            nn.init.zeros_(final.bias)
        for sequential in (self.background_head, self.background_correction_head):
            final = sequential[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _shape_count(self, shape_count: int | None) -> int:
        count = self.max_shapes if shape_count is None else shape_count
        if not 1 <= count <= self.max_shapes:
            raise ValueError(f"shape_count must be within 1..{self.max_shapes}")
        return count

    @staticmethod
    def _sample_features(features: Tensor, normalized_points: Tensor) -> Tensor:
        batch_size = features.shape[0]
        if normalized_points.ndim == 2:
            grid = normalized_points.mul(2.0).sub(1.0)[None, :, None, :]
            grid = grid.expand(batch_size, -1, -1, -1)
        elif normalized_points.ndim == 3:
            if normalized_points.shape[0] != batch_size:
                raise ValueError("batched points must match the feature batch size")
            grid = normalized_points.mul(2.0).sub(1.0)[:, :, None, :]
        else:
            raise ValueError("normalized_points must have shape [points, 2] or [batch, points, 2]")
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled[..., 0].transpose(1, 2)

    def _anchor_metadata(self, count: int, dtype: torch.dtype) -> Tensor:
        centers = self.anchor_centers[:count].to(dtype=dtype)
        sizes = self.anchor_sizes[:count].to(dtype=dtype)
        tiers = F.one_hot(self.anchor_tiers[:count], num_classes=3).to(dtype=dtype)
        return torch.cat((centers.mul(2.0).sub(1.0), sizes.log(), tiers), dim=-1)

    def predict_initial(self, image: Tensor, shape_count: int | None = None) -> V2Prediction:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        count = self._shape_count(shape_count)
        pyramid = self.encoder(image)
        centers = self.anchor_centers[:count].to(dtype=image.dtype)
        local_features = self.local_feature_fusion(
            torch.cat(
                tuple(self._sample_features(level, centers) for level in pyramid.maps),
                dim=-1,
            )
        )
        anchor_features = self.anchor_projection(self._anchor_metadata(count, image.dtype))
        slots = (
            local_features
            + anchor_features[None]
            + self.slot_embeddings.weight[:count][None]
            + pyramid.global_context[:, None, :]
        )
        for decoder_block in self.slot_decoder:
            slots = decoder_block(slots)

        raw_shapes = self.shape_head(slots) + self.base_shape_params[:count][None]
        color_grid = centers.mul(2.0).sub(1.0)[None, :, None, :]
        color_grid = color_grid.expand(image.shape[0], -1, -1, -1)
        sampled_colors = F.grid_sample(
            image,
            color_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[..., 0].transpose(1, 2).clamp(1e-4, 1.0 - 1e-4)
        color_offset = torch.zeros_like(raw_shapes)
        color_offset[..., 8:11] = torch.logit(sampled_colors)
        raw_shapes = raw_shapes + color_offset

        image_mean = image.mean(dim=(2, 3)).clamp(1e-4, 1.0 - 1e-4)
        raw_background = self.background_head(pyramid.global_context) + torch.logit(image_mean)
        return V2Prediction(raw_shapes, raw_background, slots)

    def prepare_correction_target(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        longest = max(height, width)
        if longest <= self.correction_side:
            return image
        scale = self.correction_side / longest
        target_size = (max(8, round(height * scale)), max(8, round(width * scale)))
        return F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)

    def refine_prediction(
        self,
        target: Tensor,
        rendered: Tensor,
        prediction: V2Prediction,
    ) -> V2Prediction:
        if prediction.raw_shapes.shape[:2] != prediction.slot_features.shape[:2]:
            raise ValueError("prediction slot state does not match its shape parameters")
        error_features, global_error = self.correction_encoder(target, rendered)
        centers = prediction.raw_shapes[..., 3:5].sigmoid()
        local_error = self._sample_features(error_features, centers)
        parameter_features = self.parameter_projection(
            torch.tanh(prediction.raw_shapes / 4.0)
        )
        repeated_global = global_error[:, None, :].expand(-1, centers.shape[1], -1)
        correction = self.correction_fusion(
            torch.cat(
                (prediction.slot_features, local_error, parameter_features, repeated_global),
                dim=-1,
            )
        )
        slot_features = self.correction_interaction(prediction.slot_features + correction)
        delta = torch.tanh(self.correction_head(slot_features))
        delta = delta * self.correction_delta_scales.to(dtype=delta.dtype)
        raw_shapes = prediction.raw_shapes + delta

        background_input = torch.cat(
            (global_error, torch.tanh(prediction.raw_background / 4.0)),
            dim=-1,
        )
        background_delta = 0.5 * torch.tanh(
            self.background_correction_head(background_input)
        )
        return V2Prediction(
            raw_shapes,
            prediction.raw_background + background_delta,
            slot_features,
        )

    def prediction_history(
        self,
        image: Tensor,
        shape_count: int | None = None,
        *,
        renderer: nn.Module | None = None,
        correction_steps: int | None = None,
    ) -> list[V2Prediction]:
        prediction = self.predict_initial(image, shape_count)
        history = [prediction]
        steps = (self.correction_steps if renderer is not None else 0) if correction_steps is None else correction_steps
        if steps < 0:
            raise ValueError("correction_steps cannot be negative")
        if steps > self.correction_steps:
            raise ValueError(
                f"This model is configured for at most {self.correction_steps} correction steps"
            )
        if steps and renderer is None:
            raise ValueError("renderer is required when correction_steps is positive")
        if not steps:
            return history

        target = self.prepare_correction_target(image)
        assert renderer is not None
        for _ in range(steps):
            rendered = renderer(
                prediction.raw_shapes,
                prediction.raw_background,
                target.shape[-2],
                target.shape[-1],
                use_presence=True,
            )
            prediction = self.refine_prediction(target, rendered, prediction)
            history.append(prediction)
        return history

    def forward(
        self,
        image: Tensor,
        shape_count: int | None = None,
        *,
        renderer: nn.Module | None = None,
        correction_steps: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        prediction = self.prediction_history(
            image,
            shape_count,
            renderer=renderer,
            correction_steps=correction_steps,
        )[-1]
        return prediction.raw_shapes, prediction.raw_background

    def config(self) -> dict[str, int | float]:
        return {
            "max_shapes": self.max_shapes,
            "hidden_dim": self.hidden_dim,
            "decoder_layers": self.decoder_layers,
            "attention_heads": self.attention_heads,
            "encoder_base_channels": self.encoder_base_channels,
            "correction_steps": self.correction_steps,
            "correction_side": self.correction_side,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }


def circular_angle_loss(predicted_radians: Tensor, target_radians: Tensor) -> Tensor:
    """Smooth angular distance helper for supervised synthetic pretraining."""
    return 1.0 - torch.cos(predicted_radians - target_radians)
