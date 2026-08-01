from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from vectorlearner.data import RasterImageDataset
from vectorlearner.fit import refine_initialized
from vectorlearner.renderer import PARAM_COUNT


class OptimizedPipelineTests(unittest.TestCase):
    def test_full_image_training_sample_preserves_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (64, 32), (30, 80, 140)).save(root / "wide.png")
            dataset = RasterImageDataset(
                root,
                crop_size=32,
                full_image_probability=1.0,
            )
            self.assertEqual(tuple(dataset[0].shape), (3, 16, 32))

    def test_initialized_refinement_writes_a_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.png"
            output = root / "output"
            Image.new("RGB", (48, 32), (70, 120, 180)).save(source)
            raw_shapes = torch.zeros(1, 3, PARAM_COUNT)
            raw_background = torch.zeros(1, 3)
            summary = refine_initialized(
                source,
                output,
                raw_shapes,
                raw_background,
                steps=1,
                sample_count=64,
                error_refresh=1,
                preview_side=32,
            )
            self.assertEqual(summary["active_shapes"], 3)
            self.assertEqual(summary["sampled_backend"], "torch")
            self.assertTrue((output / "result.svg").is_file())
            self.assertTrue((output / "fit.pt").is_file())
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
