from __future__ import annotations

import unittest

import torch

from vectorlearner.cuda_renderer import (
    CudaRendererUnavailable,
    CudaSampledShapeRenderer,
    cuda_renderer_available,
    cuda_renderer_unavailable_reason,
)
from vectorlearner.renderer import SoftShapeRenderer, random_raw_scenes


class CudaRendererFallbackTests(unittest.TestCase):
    def test_availability_helpers_are_safe_without_cupy(self) -> None:
        available = cuda_renderer_available()
        reason = cuda_renderer_unavailable_reason()
        self.assertIsInstance(available, bool)
        if available:
            self.assertIsNone(reason)
        else:
            self.assertIsInstance(reason, str)
            self.assertTrue(reason)

    def test_cpu_fallback_matches_fixed_type_renderer(self) -> None:
        torch.manual_seed(5)
        raw, background = random_raw_scenes(1, 9, torch.device("cpu"))
        points = torch.rand(257, 2)
        raw.requires_grad_(True)
        background.requires_grad_(True)
        accelerated = CudaSampledShapeRenderer(allow_fallback=True)
        reference = SoftShapeRenderer(learn_shape_types=False)

        actual = accelerated(
            raw,
            background,
            sample_points=points,
            canvas_size=(240, 320),
            use_presence=False,
        )
        expected = reference(
            raw,
            background,
            sample_points=points,
            canvas_size=(240, 320),
            use_presence=False,
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(accelerated.last_backend, "torch")
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue(torch.isfinite(background.grad).all())

    def test_fallback_can_be_disabled(self) -> None:
        raw, background = random_raw_scenes(1, 2, torch.device("cpu"))
        renderer = CudaSampledShapeRenderer(allow_fallback=False)
        with self.assertRaises(CudaRendererUnavailable):
            renderer(
                raw,
                background,
                sample_points=torch.rand(4, 2),
                canvas_size=(32, 32),
            )

    def test_presence_gradients_and_drop_in_grid_match_fallback(self) -> None:
        torch.manual_seed(13)
        raw, background = random_raw_scenes(1, 7, torch.device("cpu"))
        raw[..., 11] = torch.linspace(-1.5, 1.5, raw.shape[1])
        weights = torch.randn(1, 3, 18, 25)

        custom_raw = raw.detach().clone().requires_grad_(True)
        custom_background = background.detach().clone().requires_grad_(True)
        custom = CudaSampledShapeRenderer(allow_fallback=True)
        custom_image = custom(
            custom_raw,
            custom_background,
            18,
            25,
            use_presence=True,
        )
        (custom_image * weights).sum().backward()

        reference_raw = raw.detach().clone().requires_grad_(True)
        reference_background = background.detach().clone().requires_grad_(True)
        reference = SoftShapeRenderer(learn_shape_types=False)
        reference_image = reference(
            reference_raw,
            reference_background,
            18,
            25,
            use_presence=True,
        )
        (reference_image * weights).sum().backward()

        self.assertEqual(tuple(custom_image.shape), (1, 3, 18, 25))
        torch.testing.assert_close(custom_image, reference_image, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            custom_raw.grad[..., 3:12],
            reference_raw.grad[..., 3:12],
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            custom_background.grad,
            reference_background.grad,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_batched_grid_uses_compatible_fallback(self) -> None:
        raw, background = random_raw_scenes(2, 5, torch.device("cpu"))
        custom = CudaSampledShapeRenderer(allow_fallback=True)
        reference = SoftShapeRenderer(learn_shape_types=False)

        custom_image = custom(raw, background, 12, 17, use_presence=True)
        reference_image = reference(raw, background, 12, 17, use_presence=True)

        self.assertEqual(tuple(custom_image.shape), (2, 3, 12, 17))
        torch.testing.assert_close(custom_image, reference_image, atol=2e-6, rtol=2e-6)


class CudaRendererKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("PyTorch CUDA is unavailable")
        if not cuda_renderer_available(probe_kernel=True):
            self.skipTest(cuda_renderer_unavailable_reason() or "Custom CUDA renderer unavailable")

    def test_forward_and_gradients_match_pytorch_renderer(self) -> None:
        torch.manual_seed(17)
        device = torch.device("cuda")
        # Cross the 16-shape checkpoint boundary used by the custom backward.
        raw, background = random_raw_scenes(1, 21, device)
        raw[..., 11] = torch.linspace(-1.5, 1.5, raw.shape[1], device=device)
        points = torch.rand(257, 2, device=device)
        weights = torch.randn(1, 3, points.shape[0], device=device)

        custom_raw = raw.detach().clone().requires_grad_(True)
        custom_background = background.detach().clone().requires_grad_(True)
        custom = CudaSampledShapeRenderer(allow_fallback=False).to(device)
        custom_image = custom(
            custom_raw,
            custom_background,
            sample_points=points,
            canvas_size=(720, 1280),
            use_presence=True,
            softness_px=1.0,
        )
        (custom_image * weights).sum().backward()

        reference_raw = raw.detach().clone().requires_grad_(True)
        reference_background = background.detach().clone().requires_grad_(True)
        reference = SoftShapeRenderer(learn_shape_types=False).to(device)
        reference_image = reference(
            reference_raw,
            reference_background,
            sample_points=points,
            canvas_size=(720, 1280),
            use_presence=True,
            softness_px=1.0,
        )
        (reference_image * weights).sum().backward()

        self.assertEqual(custom.last_backend, "cuda")
        torch.testing.assert_close(custom_image, reference_image, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            custom_raw.grad[..., 3:12],
            reference_raw.grad[..., 3:12],
            atol=5e-4,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            custom_background.grad,
            reference_background.grad,
            atol=2e-4,
            rtol=2e-4,
        )

    def test_cuda_grid_matches_full_pytorch_render(self) -> None:
        torch.manual_seed(23)
        device = torch.device("cuda")
        raw, background = random_raw_scenes(1, 7, device)
        raw[..., 11] = torch.linspace(-1.0, 1.0, raw.shape[1], device=device)
        custom = CudaSampledShapeRenderer(allow_fallback=False).to(device)
        reference = SoftShapeRenderer(learn_shape_types=False).to(device)

        custom_image = custom(raw, background, 18, 25, use_presence=True)
        reference_image = reference(raw, background, 18, 25, use_presence=True)

        self.assertEqual(tuple(custom_image.shape), (1, 3, 18, 25))
        torch.testing.assert_close(custom_image, reference_image, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
