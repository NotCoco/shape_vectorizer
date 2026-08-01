"""GPU-first raster-to-SVG learning tools."""

from .model import ShapeVectorizer
from .renderer import PARAM_COUNT, SoftShapeRenderer

__all__ = ["PARAM_COUNT", "ShapeVectorizer", "SoftShapeRenderer"]

