from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from vectorlearner.checkpoint import (
    detect_model_version,
    load_vectorizer_checkpoint,
    newest_checkpoint,
)
from vectorlearner.infer import main as infer_main
from vectorlearner.model import ShapeVectorizer
from vectorlearner.model_v2 import ShapeVectorizerV2
from vectorlearner.ui import FastVectorizer, preferred_default_checkpoint


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.device = torch.device("cpu")

    def _v1(self) -> ShapeVectorizer:
        return ShapeVectorizer(
            max_shapes=8,
            hidden_dim=32,
            decoder_layers=1,
            attention_heads=4,
            max_feature_side=8,
        )

    def _v2(self) -> ShapeVectorizerV2:
        return ShapeVectorizerV2(
            max_shapes=8,
            hidden_dim=32,
            decoder_layers=1,
            attention_heads=4,
            encoder_base_channels=16,
            correction_steps=1,
            correction_side=16,
        )

    def test_loader_preserves_v1_and_loads_explicit_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1 = self._v1()
            v1_path = root / "v1.pt"
            torch.save(
                {
                    "model": v1.state_dict(),
                    "model_config": v1.config(),
                    "active_shapes": 6,
                },
                v1_path,
            )
            loaded_v1 = load_vectorizer_checkpoint(v1_path, self.device)
            self.assertEqual(loaded_v1.model_version, "v1")
            self.assertIsInstance(loaded_v1.model, ShapeVectorizer)
            self.assertEqual(loaded_v1.trained_shapes, 6)

            v2 = self._v2()
            v2_path = root / "v2.pt"
            torch.save(
                {
                    "model_version": "v2",
                    "model": v2.state_dict(),
                    "model_config": v2.config(),
                    "active_shapes": 8,
                },
                v2_path,
            )
            loaded_v2 = load_vectorizer_checkpoint(v2_path, self.device)
            self.assertEqual(loaded_v2.model_version, "v2")
            self.assertIsInstance(loaded_v2.model, ShapeVectorizerV2)
            self.assertEqual(loaded_v2.trained_shapes, 8)

    def test_legacy_v2_config_is_detected_without_marker(self) -> None:
        model = self._v2()
        checkpoint = {"model": model.state_dict(), "model_config": model.config()}
        self.assertEqual(detect_model_version(checkpoint), "v2")

    def test_newest_checkpoint_prefers_latest_file_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / "fallback.pt"
            fallback.write_bytes(b"fallback")
            checkpoint_dir = root / "v2-optimized"
            checkpoint_dir.mkdir()
            older = checkpoint_dir / "checkpoint_0000010.pt"
            newer = checkpoint_dir / "checkpoint_0000020.pt"
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")
            os.utime(older, ns=(1_000_000_000, 1_000_000_000))
            os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(newest_checkpoint(checkpoint_dir, fallback), newer)
            self.assertEqual(newest_checkpoint(root / "missing", fallback), fallback)
            with (
                patch("vectorlearner.ui.V2_CHECKPOINT_DIR", checkpoint_dir),
                patch("vectorlearner.ui.BUNDLED_CHECKPOINT", root / "missing-bundled.pt"),
                patch("vectorlearner.ui.DEFAULT_CHECKPOINT", fallback),
            ):
                self.assertEqual(preferred_default_checkpoint(), newer)

            bundled = root / "bundled.pt"
            bundled.write_bytes(b"bundled")
            with (
                patch("vectorlearner.ui.V2_CHECKPOINT_DIR", root / "missing-checkpoints"),
                patch("vectorlearner.ui.BUNDLED_CHECKPOINT", bundled),
                patch("vectorlearner.ui.DEFAULT_CHECKPOINT", fallback),
            ):
                self.assertEqual(preferred_default_checkpoint(), bundled)

    def test_ui_vectorizer_runs_v2_learned_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._v2()
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_version": "v2",
                    "model": model.state_dict(),
                    "model_config": model.config(),
                    "active_shapes": 8,
                },
                checkpoint_path,
            )
            source = root / "input.png"
            destination = root / "result.svg"
            Image.new("RGB", (24, 20), (72, 128, 190)).save(source)

            vectorizer = FastVectorizer(checkpoint_path, self.device)
            metadata = vectorizer.vectorize(source, destination, 8)

            self.assertTrue(destination.is_file())
            self.assertEqual(metadata["model_version"], "v2")
            self.assertEqual(metadata["correction_steps"], 1)
            self.assertEqual(metadata["shapes"], 8)

    def test_ui_vectorizer_preserves_v1_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._v1()
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model.config(),
                    "active_shapes": 8,
                },
                checkpoint_path,
            )
            source = root / "input.png"
            destination = root / "result.svg"
            Image.new("RGB", (24, 20), (72, 128, 190)).save(source)

            vectorizer = FastVectorizer(checkpoint_path, self.device)
            metadata = vectorizer.vectorize(source, destination, 8)

            self.assertTrue(destination.is_file())
            self.assertEqual(metadata["model_version"], "v1")
            self.assertEqual(metadata["correction_steps"], 0)

    def test_cli_inference_runs_v2_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._v2()
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_version": "v2",
                    "model": model.state_dict(),
                    "model_config": model.config(),
                    "active_shapes": 8,
                },
                checkpoint_path,
            )
            source = root / "input.png"
            output = root / "output"
            Image.new("RGB", (24, 20), (72, 128, 190)).save(source)
            arguments = [
                "vectorlearner.infer",
                str(checkpoint_path),
                str(source),
                "--output-dir",
                str(output),
                "--shapes",
                "8",
                "--encoder-max-side",
                "32",
                "--preview-max-side",
                "32",
                "--correction-steps",
                "1",
                "--device",
                "cpu",
            ]
            with patch.object(sys, "argv", arguments):
                infer_main()

            metadata = json.loads((output / "inference.json").read_text(encoding="utf-8"))
            self.assertTrue((output / "result.svg").is_file())
            self.assertEqual(metadata["model_version"], "v2")
            self.assertEqual(metadata["correction_steps"], 1)


if __name__ == "__main__":
    unittest.main()
