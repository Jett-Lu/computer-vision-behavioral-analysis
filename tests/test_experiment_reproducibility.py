from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.dataset import ActivitySample
from activity_recognition.preprocessing import cache_activity_samples, validate_cached_samples
from activity_recognition.train import train_classifier


class ProvenanceTest(unittest.TestCase):
    def test_manifest_cannot_silently_drop_samples_or_use_unsafe_keys(self):
        metadata = self.prepare()
        metadata["cache_provenance"]["omitted"] = dict(metadata["cache_provenance"]["one"])
        with self.assertRaisesRegex(ValueError, "partition|registry"):
            validate_cached_samples([self.sample], metadata)
        with self.assertRaisesRegex(ValueError, "key"):
            self.prepare([replace(self.sample, key="../outside")])

    def test_split_definition_hashes_and_quality_report(self):
        from activity_recognition.provenance import annotation_provenance, cache_quality_report

        self.prepare()
        annotation = self.root / "walk_test_split1.txt"
        annotation.write_text("video.avi 1\n")
        first = annotation_provenance([annotation], "official-split-files")
        annotation.write_text("video.avi 2\n")
        self.assertNotEqual(first, annotation_provenance([annotation], "official-split-files"))
        report = cache_quality_report([self.sample])
        self.assertEqual(report["by_partition"]["train"]["walking"]["samples"], 1)
        self.assertEqual(
            report["by_partition"]["train"]["walking"]["missing_keypoint_fraction"], 1.0
        )
        self.assertEqual(
            report["by_partition"]["train"]["walking"]["no_visible_pose_frame_fraction"], 1.0
        )

    def test_backbone_weight_changes_invalidate_feature_cache(self):
        from activity_recognition.train import cache_backbone_features

        self.prepare()
        extractor = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten())
        with (
            patch(
                "activity_recognition.train.backbone_weight_fingerprint", return_value="first"
            ) as fingerprint,
            patch(
                "activity_recognition.train.build_mobilenet_extractor", return_value=extractor
            ) as build,
        ):
            cache_backbone_features([self.sample], self.root, "cnn", torch.device("cpu"))
            cache_backbone_features([self.sample], self.root, "cnn", torch.device("cpu"))
            self.assertEqual(build.call_count, 1)
            fingerprint.return_value = "changed"
            cache_backbone_features([self.sample], self.root, "cnn", torch.device("cpu"))
            self.assertEqual(build.call_count, 2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "video.avi"
        source.write_bytes(b"video one")
        self.sample = ActivitySample(
            "one", str(source), str(self.root / "one.npz"), "walking", 0, "train", "walk"
        )
        self.contract = {"fixture": "pose-v1", "weights_sha256": "a" * 64}

    def prepare(self, samples=None):
        samples = samples or [self.sample]
        analyzer = Mock()
        analyzer.analyze.return_value = []
        with patch(
            "activity_recognition.preprocessing.sample_video_frames",
            return_value=[np.zeros((8, 8, 3), np.uint8)] * 2,
        ):
            cache_activity_samples(
                samples,
                frames_per_sample=2,
                pose_analyzer=analyzer,
                extraction_provenance=self.contract,
            )
        from activity_recognition.provenance import dataset_provenance

        return {
            "frames_per_sample": 2,
            "sampling_interval_seconds": 0.1,
            "extraction": self.contract,
            "cache_provenance": dataset_provenance(samples),
        }

    def test_swapped_split_source_and_modified_content_are_rejected(self):
        metadata = self.prepare()
        validate_cached_samples([self.sample], metadata)
        for change in (
            {"split": "test"},
            {"key": "different"},
            {"video_path": str(self.root / "other.avi")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cached_samples([replace(self.sample, **change)], metadata)
        Path(self.sample.video_path).write_bytes(b"changed video")
        with self.assertRaises(ValueError):
            validate_cached_samples([self.sample], metadata)

    def test_duplicate_content_across_partitions_rejected_before_extraction(self):
        other = self.root / "duplicate.avi"
        other.write_bytes(Path(self.sample.video_path).read_bytes())
        duplicate = replace(
            self.sample,
            key="two",
            video_path=str(other),
            cache_path=str(self.root / "two.npz"),
            split="test",
        )
        with self.assertRaisesRegex(ValueError, "content.*partition"):
            self.prepare([self.sample, duplicate])

    def test_changed_extraction_invalidates_but_same_contract_preserves_cache(self):
        self.prepare()
        original = Path(self.sample.cache_path).read_bytes()
        self.prepare()
        self.assertEqual(original, Path(self.sample.cache_path).read_bytes())
        self.contract = {**self.contract, "threshold": 0.5}
        self.prepare()
        self.assertNotEqual(original, Path(self.sample.cache_path).read_bytes())

    def test_corrupted_finite_arrays_rejected_even_if_zip_is_readable(self):
        metadata = self.prepare()
        with np.load(self.sample.cache_path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["pose"][0, 0, 0] = 0.5
        np.savez_compressed(self.sample.cache_path, **arrays)
        with self.assertRaises(ValueError):
            validate_cached_samples([self.sample], metadata)


class RNGTest(unittest.TestCase):
    def test_mps_rng_restored_on_success_and_failure(self):
        from activity_recognition.provenance import isolated_rng

        for fail in (False, True):
            state = {"value": torch.tensor([42], dtype=torch.uint8)}
            with (
                patch("torch.backends.mps.is_available", return_value=True),
                patch("torch.mps.get_rng_state", side_effect=lambda: state["value"].clone()),
                patch(
                    "torch.mps.set_rng_state", side_effect=lambda value: state.update(value=value)
                ),
                patch(
                    "torch.mps.manual_seed",
                    side_effect=lambda seed: state.update(
                        value=torch.tensor([seed], dtype=torch.uint8)
                    ),
                ),
            ):
                try:
                    with isolated_rng(123):
                        self.assertEqual(state["value"].item(), 123)
                        if fail:
                            raise RuntimeError("fixture")
                except RuntimeError:
                    if not fail:
                        raise
                self.assertEqual(state["value"].item(), 42)

    def test_rng_restores_numpy_and_deterministic_flags_after_failure(self):
        from activity_recognition.provenance import isolated_rng

        before = np.random.get_state()
        deterministic = torch.are_deterministic_algorithms_enabled()
        with self.assertRaises(RuntimeError):
            with isolated_rng(10):
                np.random.random(4)
                raise RuntimeError("synthetic failure")
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())

    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.pt"
            path.write_bytes(b"preserve history")
            with self.assertRaises(FileExistsError):
                train_classifier(
                    "cnn",
                    np.ones((2, 3)),
                    np.array([0, 1]),
                    np.ones((2, 3)),
                    np.array([0, 1]),
                    path,
                    epochs=1,
                    patience=1,
                    batch_size=2,
                    learning_rate=0.001,
                    seed=7,
                    device=torch.device("cpu"),
                )
            self.assertEqual(path.read_bytes(), b"preserve history")

    def test_classifier_reproducible_despite_prior_rng_consumption_and_restores_state(self):
        with tempfile.TemporaryDirectory() as directory:
            features = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
            labels = np.array([0, 1, 2, 3] * 2)
            states = []
            histories = []
            for i in range(2):
                torch.rand(100 + i * 93)
                before = torch.random.get_rng_state().clone()
                py_before = random.getstate()
                path = Path(directory) / f"{i}.pt"
                result = train_classifier(
                    "cnn",
                    features,
                    labels,
                    features,
                    labels,
                    path,
                    epochs=2,
                    patience=2,
                    batch_size=4,
                    learning_rate=0.001,
                    seed=7,
                    device=torch.device("cpu"),
                )
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                self.assertEqual(py_before, random.getstate())
                states.append(torch.load(path, weights_only=True)["state_dict"])
                histories.append(result.history)
            self.assertEqual(histories[0], histories[1])
            self.assertTrue(all(torch.equal(states[0][k], states[1][k]) for k in states[0]))


class EvidenceTest(unittest.TestCase):
    def test_latency_synchronizes_selected_cuda_device_and_mps(self):
        from activity_recognition.evaluate import _synchronize

        with patch("torch.cuda.synchronize") as cuda, patch("torch.mps.synchronize") as mps:
            _synchronize(torch.device("cuda:1"))
            cuda.assert_called_once_with(torch.device("cuda:1"))
            _synchronize(torch.device("mps"))
            mps.assert_called_once_with()

    def test_source_latency_reports_repeated_lazy_predictor_initialization(self):
        from activity_recognition.evaluate import measure_source_latency

        class LazyAnalyzer:
            def __init__(self):
                self.predictor = None
                self.setups = 0

            def reset_tracking(self):
                self.predictor = None

            def analyze(self, frame):
                if self.predictor is None:
                    self.setups += 1
                    self.predictor = object()
                return []

        analyzer = LazyAnalyzer()
        with patch(
            "activity_recognition.preprocessing.sample_video_frames",
            return_value=[np.zeros((8, 8, 3), np.uint8)],
        ):
            report = measure_source_latency(
                [ActivitySample("k", "fake.avi", "cache", "walking", 0, "test", "walk")],
                analyzer,
                lambda pose, crops: None,
                torch.device("cpu"),
                frames_per_sample=1,
                interval=0.1,
                warmup=1,
                repeats=2,
            )
        self.assertEqual(analyzer.setups, 3)
        self.assertIn("predictor initialization", report["source_pipeline_scope"])
        self.assertIn("warmup", report["source_pipeline_scope"])

    def test_full_synthetic_experiment_warm_cold_cache_and_independent_metrics(self):
        from contextlib import ExitStack
        from activity_recognition.dataset import write_manifest
        from activity_recognition.provenance import file_hash
        from activity_recognition.train import train_all_models
        from activity_recognition.evaluate import evaluate_all_models
        from activity_recognition.labels import ACTIVITY_LABELS

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            samples = []
            for i, split in enumerate(("train", "validation", "test")):
                video = root / f"{i}.avi"
                video.write_bytes(f"fixture {i}".encode())
                samples.append(
                    ActivitySample(
                        str(i), str(video), str(root / f"{i}.npz"), "walking", 0, split, "walk"
                    )
                )
            analyzer = Mock()
            analyzer.analyze.return_value = []
            stack.enter_context(
                patch(
                    "activity_recognition.preprocessing.sample_video_frames",
                    return_value=[np.zeros((8, 8, 3), np.uint8)] * 2,
                )
            )
            contract = {"fixture": True}
            cache_activity_samples(
                samples, frames_per_sample=2, pose_analyzer=analyzer, extraction_provenance=contract
            )
            manifest = root / "manifest.json"
            write_manifest(
                manifest,
                samples,
                dataset_root=root,
                annotations_root=root,
                fold=1,
                validation_fraction=0.2,
                seed=7,
                frames_per_sample=2,
                extraction_provenance=contract,
            )
            stack.enter_context(
                patch(
                    "activity_recognition.train.backbone_weight_fingerprint", return_value="fixture"
                )
            )
            for module in ("train", "evaluate"):
                stack.enter_context(
                    patch(
                        f"activity_recognition.{module}.build_mobilenet_extractor",
                        side_effect=lambda: torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
                        ),
                    )
                )
                stack.enter_context(
                    patch(
                        f"activity_recognition.{module}.build_s3d_extractor",
                        side_effect=lambda: torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten()
                        ),
                    )
                )
            for run in ("cold", "warm"):
                train_all_models(
                    manifest,
                    root / "features",
                    root / run / "models",
                    root / run / "results",
                    epochs=1,
                    patience=1,
                    batch_size=1,
                    seed=7,
                    device_name="cpu",
                )
            for model in ("mlp", "cnn", "advanced"):
                cold = torch.load(root / "cold" / "models" / f"{model}.pt", weights_only=True)
                warm = torch.load(root / "warm" / "models" / f"{model}.pt", weights_only=True)
                self.assertTrue(
                    all(
                        torch.equal(v, warm["state_dict"][key])
                        for key, v in cold["state_dict"].items()
                    )
                )
                self.assertEqual(cold["dataset_provenance"]["manifest_sha256"], file_hash(manifest))
                self.assertEqual(cold["labels"], list(ACTIVITY_LABELS))
            results = evaluate_all_models(
                manifest,
                root / "features",
                root / "cold" / "models",
                root / "cold" / "results",
                device_name="cpu",
                latency_warmup=1,
                latency_repeats=2,
            )
            report = json.loads(
                (root / "cold" / "results" / "reconstructed_metrics.json").read_text()
            )
            for model in results:
                for metric in ("accuracy", "macro_f1", "confusion_matrix"):
                    self.assertEqual(results[model][metric], report["models"][model][metric])
            self.assertTrue((root / "cold" / "results" / "latency.csv").is_file())
            run_manifest = json.loads((root / "cold" / "results" / "run_manifest.json").read_text())
            self.assertIn("dataset_quality", run_manifest)
            with self.assertRaises(FileExistsError):
                evaluate_all_models(
                    manifest,
                    root / "features",
                    root / "cold" / "models",
                    root / "cold" / "results",
                    device_name="cpu",
                )
            payload = json.loads(manifest.read_text())
            payload["seed"] = 99
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "manifest|provenance"):
                evaluate_all_models(
                    manifest,
                    root / "features",
                    root / "cold" / "models",
                    root / "changed" / "results",
                    device_name="cpu",
                )

    def test_optional_source_latency_measures_decode_pose_and_full_pipeline(self):
        from activity_recognition.evaluate import measure_source_latency

        analyzer = Mock()
        analyzer.analyze.return_value = []
        frames = [np.zeros((8, 8, 3), np.uint8)] * 2
        with patch("activity_recognition.preprocessing.sample_video_frames", return_value=frames):
            report = measure_source_latency(
                [ActivitySample("k", "fake.avi", "cache", "walking", 0, "test", "walk")],
                analyzer,
                lambda pose, crops: None,
                torch.device("cpu"),
                frames_per_sample=2,
                interval=0.1,
                warmup=1,
                repeats=2,
            )
        self.assertEqual(report["pose_extraction"]["count"], 2)
        self.assertEqual(report["end_to_end_video"]["count"], 2)
        self.assertEqual(analyzer.reset_tracking.call_count, 3)
        self.assertEqual(analyzer.analyze.call_count, 6)

    def test_prediction_reconstruction_rejects_duplicates_and_wrong_labels(self):
        from activity_recognition.evidence import reconstruct_predictions
        from activity_recognition.metrics import write_predictions_csv

        rows = [
            dict(
                model="mlp",
                sample_key=str(i),
                source_video=f"{i}.avi",
                split="test",
                expected_index=i,
                expected_label=label,
                predicted_index=0,
                predicted_label="walking",
            )
            for i, label in enumerate(("walking", "running", "standing", "sitting"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_predictions_csv(path, rows)
            result = reconstruct_predictions(path)
            self.assertEqual(result["models"]["mlp"]["accuracy"], 0.25)
            self.assertEqual(result["models"]["mlp"]["macro_f1"], 0.1)
            self.assertEqual(len(result["models"]["mlp"]["misclassifications"]), 3)
            write_predictions_csv(path, rows + rows[:1])
            with self.assertRaises(ValueError):
                reconstruct_predictions(path)
            rows[0]["expected_label"] = "running"
            write_predictions_csv(path, rows)
            with self.assertRaises(ValueError):
                reconstruct_predictions(path)

    def test_latency_reports_separate_boundaries_and_repeated_statistics(self):
        from activity_recognition.evaluate import measure_latency_breakdown

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(
                path, pose=np.ones((1, 17, 3), np.float32), crops=np.zeros((1, 8, 8, 3), np.uint8)
            )
            sample = ActivitySample("one", "unused", str(path), "walking", 0, "test", "walk")
            model = torch.nn.Linear(51, 4).eval()
            checkpoint = {"feature_mean": torch.zeros(51), "feature_std": torch.ones(51)}
            result = measure_latency_breakdown(
                "mlp", model, checkpoint, [sample], torch.device("cpu"), warmup=1, repeats=3
            )
            self.assertEqual(result["classifier_only"]["count"], 3)
            self.assertGreaterEqual(
                result["classifier_only"]["p95_ms"], result["classifier_only"]["median_ms"]
            )
            self.assertIsNone(result["pose_extraction"])
            self.assertIsNone(result["end_to_end_video"])
            self.assertIn("preprocessing", result)


if __name__ == "__main__":
    unittest.main()
