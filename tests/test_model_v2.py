from __future__ import annotations

import unittest

import torch

from vectorlearner.model_v2 import ShapeVectorizerV2
from vectorlearner.renderer import PARAM_COUNT, SoftShapeRenderer
from vectorlearner.train_v2 import make_synthetic_slot_batch, synthetic_parameter_losses


class ModelV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.device = torch.device("cpu")

    def _small_model(self, *, max_shapes: int = 12, correction_steps: int = 1) -> ShapeVectorizerV2:
        return ShapeVectorizerV2(
            max_shapes=max_shapes,
            hidden_dim=32,
            decoder_layers=1,
            attention_heads=4,
            encoder_base_channels=16,
            correction_steps=correction_steps,
            correction_side=32,
        ).to(self.device)

    def test_all_450_slots_are_renderer_compatible(self) -> None:
        model = self._small_model(max_shapes=450, correction_steps=0).eval()
        image = torch.rand(1, 3, 28, 40, device=self.device)
        with torch.no_grad():
            raw_shapes, raw_background = model(image, 450)
            sampled = SoftShapeRenderer(chunk_size=16).to(self.device)(
                raw_shapes,
                raw_background,
                sample_points=torch.rand(24, 2, device=self.device),
                canvas_size=(1440, 2560),
                use_presence=True,
            )
        self.assertEqual(tuple(raw_shapes.shape), (1, 450, PARAM_COUNT))
        self.assertEqual(tuple(raw_background.shape), (1, 3))
        self.assertEqual(tuple(sampled.shape), (1, 3, 24))
        self.assertEqual(torch.bincount(model.anchor_tiers).tolist(), [32, 96, 322])
        self.assertTrue(torch.isfinite(sampled).all())

    def test_learned_correction_pass_has_finite_gradients(self) -> None:
        model = self._small_model()
        renderer = SoftShapeRenderer(chunk_size=4).to(self.device)
        image = torch.rand(1, 3, 28, 36, device=self.device)
        history = model.prediction_history(
            image,
            12,
            renderer=renderer,
            correction_steps=1,
        )
        self.assertEqual(len(history), 2)
        final = history[-1]
        rendered = renderer(
            final.raw_shapes,
            final.raw_background,
            28,
            36,
            use_presence=True,
        )
        loss = (rendered - image).square().mean()
        loss.backward()
        final_layer = model.correction_head[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        self.assertIsNotNone(final_layer.weight.grad)
        assert final_layer.weight.grad is not None
        self.assertTrue(torch.isfinite(final_layer.weight.grad).all())
        self.assertGreater(float(final_layer.weight.grad.abs().sum()), 0.0)

    def test_synthetic_pretraining_batch_and_slot_loss(self) -> None:
        model = self._small_model(max_shapes=16, correction_steps=0)
        renderer = SoftShapeRenderer(chunk_size=4).to(self.device)
        batch = make_synthetic_slot_batch(
            model,
            renderer,
            batch_size=2,
            shape_count=9,
            image_size=28,
            device=self.device,
        )
        prediction = model.predict_initial(batch.image, 9)
        losses = synthetic_parameter_losses(
            prediction,
            batch.raw_shapes,
            batch.raw_background,
            min_size=model.min_size,
            max_size=model.max_size,
        )
        self.assertEqual(tuple(batch.image.shape), (2, 3, 28, 28))
        self.assertEqual(tuple(batch.raw_shapes.shape), (2, 9, PARAM_COUNT))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(model.shape_head[-1].weight.grad)

    def test_checkpoint_config_round_trip(self) -> None:
        original = self._small_model(max_shapes=21, correction_steps=1)
        restored = ShapeVectorizerV2(**original.config())
        self.assertEqual(restored.config(), original.config())
        self.assertEqual(tuple(restored.anchor_centers.shape), (21, 2))


if __name__ == "__main__":
    unittest.main()
