from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import torch

from vectorlearner.model import ShapeVectorizer
from vectorlearner.renderer import PARAM_COUNT, SoftShapeRenderer, decode_shapes, random_raw_scenes
from vectorlearner.svg import svg_string


class RendererTests(unittest.TestCase):
    def test_full_render_has_finite_gradients(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = SoftShapeRenderer(chunk_size=4).to(device)
        raw, background = random_raw_scenes(1, 6, device)
        raw.requires_grad_(True)
        background.requires_grad_(True)
        image = renderer(raw, background, 32, 48, use_presence=True)
        self.assertEqual(tuple(image.shape), (1, 3, 32, 48))
        image.mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue(torch.isfinite(background.grad).all())

    def test_batched_compositing_matches_layer_order(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = SoftShapeRenderer(chunk_size=4).to(device)
        raw, background = random_raw_scenes(1, 9, device)
        points = torch.rand(300, 2, device=device)
        actual = renderer(
            raw,
            background,
            sample_points=points,
            canvas_size=(720, 1280),
            use_presence=True,
        )
        decoded = decode_shapes(raw)
        masks = renderer._masks(decoded, points, (720, 1280), renderer.softness_px)
        masks = masks * decoded.presence[:, :, None]
        expected = background.sigmoid()[:, :, None].expand(-1, -1, points.shape[0])
        for layer in range(raw.shape[1]):
            alpha = masks[:, layer : layer + 1]
            expected = expected * (1.0 - alpha) + decoded.colors[:, layer, :, None] * alpha
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_450_shapes_support_sampled_native_loss(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = SoftShapeRenderer(chunk_size=16).to(device)
        raw, background = random_raw_scenes(1, 450, device)
        points = torch.rand(256, 2, device=device)
        with torch.no_grad():
            image = renderer(
                raw,
                background,
                sample_points=points,
                canvas_size=(2160, 3840),
                use_presence=False,
            )
        self.assertEqual(tuple(image.shape), (1, 3, 256))
        self.assertTrue(torch.isfinite(image).all())

    def test_fixed_type_renderer_matches_learnable_type_forward(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw, background = random_raw_scenes(1, 21, device)
        points = torch.rand(512, 2, device=device)
        learned = SoftShapeRenderer(chunk_size=8, learn_shape_types=True).to(device)
        fixed = SoftShapeRenderer(chunk_size=8, learn_shape_types=False).to(device)
        learned_image = learned(
            raw,
            background,
            sample_points=points,
            canvas_size=(1080, 1920),
            use_presence=False,
        )
        fixed_image = fixed(
            raw,
            background,
            sample_points=points,
            canvas_size=(1080, 1920),
            use_presence=False,
        )
        torch.testing.assert_close(fixed_image, learned_image, atol=2e-6, rtol=2e-6)

    def test_constraints_are_applied(self) -> None:
        raw = torch.randn(2, 7, PARAM_COUNT) * 100.0
        decoded = decode_shapes(raw, min_size=0.02, max_size=0.5)
        self.assertGreaterEqual(float(decoded.sizes.min()), 0.02 - 1e-6)
        self.assertLessEqual(float(decoded.sizes.max()), 0.5 + 1e-6)
        self.assertGreaterEqual(float(decoded.centers.min()), 0.0)
        self.assertLessEqual(float(decoded.centers.max()), 1.0)

    def test_svg_is_valid_xml(self) -> None:
        raw, background = random_raw_scenes(1, 9, torch.device("cpu"))
        document = svg_string(raw, background, 1920, 1080)
        root = ET.fromstring(document)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertEqual(root.attrib["viewBox"], "0 0 1920 1080")


class ModelTests(unittest.TestCase):
    def test_variable_resolution_model_output(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ShapeVectorizer(max_shapes=450, hidden_dim=64, decoder_layers=1).to(device)
        image = torch.rand(1, 3, 96, 128, device=device)
        with torch.no_grad():
            shapes, background = model(image, 9)
        self.assertEqual(tuple(shapes.shape), (1, 9, PARAM_COUNT))
        self.assertEqual(tuple(background.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
