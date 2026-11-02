from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.dataset import ActivitySample
from activity_recognition.evaluate import (
    evaluate_all_models,
    validate_checkpoint_labels,
)
from activity_recognition.labels import ACTIVITY_LABELS, LABEL_TO_INDEX
from activity_recognition.models import (
    ActivityClassifier,
    build_mobilenet_extractor,
    build_s3d_extractor,
)
from activity_recognition.train import cache_backbone_features


class ActivityModelTest(unittest.TestCase):
    def setUp(self):
        weights = patch(
            "activity_recognition.train.backbone_weight_fingerprint", return_value="mock-backbone"
        )
        weights.start()
        self.addCleanup(weights.stop)

    def test_mlp_returns_four_logits(self) -> None:
        model = ActivityClassifier(input_dim=16 * 17 * 3, hidden_dim=32)

        output = model(torch.zeros(2, 16 * 17 * 3))

        self.assertEqual(tuple(output.shape), (2, 4))

    def test_frozen_mobilenet_returns_feature_vectors(self) -> None:
        model = build_mobilenet_extractor(pretrained=False)

        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))

        self.assertEqual(tuple(output.shape), (1, 1280))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_frozen_s3d_returns_video_feature_vectors(self) -> None:
        model = build_s3d_extractor(pretrained=False)

        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 16, 224, 224))

        self.assertEqual(tuple(output.shape), (1, 1024))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_checkpoint_labels_must_match_canonical_order(self) -> None:
        validate_checkpoint_labels(
            {
                "labels": list(ACTIVITY_LABELS),
                "label_to_index": dict(LABEL_TO_INDEX),
            },
            Path("valid.pt"),
        )
        with self.assertRaises(ValueError):
            validate_checkpoint_labels(
                {
                    "labels": list(reversed(ACTIVITY_LABELS)),
                    "label_to_index": dict(LABEL_TO_INDEX),
                },
                Path("invalid.pt"),
            )

    def test_cached_features_do_not_load_the_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_root = Path(directory)
            sample = ActivitySample(
                key="cached",
                video_path="clip.avi",
                cache_path=str(feature_root / "clip.npz"),
                label="walking",
                label_index=0,
                split="train",
                hmdb51_class="walk",
            )
            np.savez_compressed(
                sample.cache_path,
                crops=np.zeros((2, 8, 8, 3), dtype=np.uint8),
            )
            tiny_extractor = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
            )
            with patch(
                "activity_recognition.train.build_mobilenet_extractor",
                return_value=tiny_extractor,
            ) as first_build:
                cache_backbone_features([sample], feature_root, "cnn", torch.device("cpu"))
            first_build.assert_called_once_with()
            output_path = feature_root / "cnn" / "cached.npy"
            original_vector = output_path.read_bytes()

            with patch("activity_recognition.train.build_mobilenet_extractor") as build_extractor:
                cache_backbone_features(
                    [sample],
                    feature_root,
                    "cnn",
                    torch.device("cpu"),
                )
            self.assertEqual(output_path.read_bytes(), original_vector)

        build_extractor.assert_not_called()

    def test_evaluation_rejects_zero_latency_samples_before_loading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "latency_samples must be positive"):
            evaluate_all_models(
                Path("missing-manifest.json"),
                Path("features"),
                Path("models"),
                Path("results"),
                latency_samples=0,
            )


if __name__ == "__main__":
    unittest.main()
