"""Regression tests for managed writes and privacy opt-outs."""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics import _check_writable_directory
from events import EventRecorder
import data_policy
from dataclasses import replace
from config import AppConfig
from enrollment import (
    validate_enrollment_frame,
    capture_enrollment_images,
    EnrollmentSession,
    MenuState,
)
from media_export import _export_image
import numpy as np
from unittest.mock import Mock


class ManagedWriteTest(unittest.TestCase):
    def test_video_releases_capture_when_private_temp_creation_fails(self):
        from media_export import _export_video

        with tempfile.TemporaryDirectory() as directory:
            capture = Mock()
            capture.isOpened.return_value = True
            with (
                patch("media_export.cv2.VideoCapture", return_value=capture),
                patch("data_policy._restrict_access", side_effect=OSError("ACL failed")),
            ):
                with self.assertRaises(OSError):
                    _export_video(Path("input.mp4"), Path(directory) / "out.mp4", Mock())
            capture.release.assert_called_once()

    def test_benchmark_report_rejects_hardlink(self):
        from activity_recognition.metrics import write_metrics_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("keep")
            output = root / "metrics.json"
            os.link(target, output)
            with self.assertRaises(ValueError):
                write_metrics_json(output, {"accuracy": 0})
            self.assertEqual(target.read_text(), "keep")

    def test_enrollment_delete_rejects_nested_hardlink(self):
        from enrollment import _remove_enrollment_folder

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"keep")
            folder = root / "enrolled"
            folder.mkdir()
            os.link(target, folder / "photo.jpg")
            with self.assertRaises(ValueError):
                _remove_enrollment_folder(folder, root)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_model_download_rejects_hardlinked_destination(self):
        from model_manager import ensure_model, ModelSpec

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"keep")
            output = root / "model.onnx"
            os.link(target, output)
            with patch("model_manager.urlopen", side_effect=AssertionError("network")):
                with self.assertRaises(ValueError):
                    ensure_model(
                        ModelSpec("test", output, "https://example.test", "bad"), AppConfig()
                    )
            self.assertEqual(target.read_bytes(), b"keep")

    def test_rotating_log_rejects_hardlinked_destination(self):
        from logging_setup import PrivateRotatingFileHandler

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"keep")
            path = root / "monitoring.log"
            os.link(target, path)
            with self.assertRaises(ValueError):
                PrivateRotatingFileHandler(path, maxBytes=1, backupCount=3)

    def test_export_rejects_hardlinked_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            import cv2

            cv2.imwrite(str(source), np.zeros((8, 8, 3), np.uint8))
            target = root / "keep"
            target.write_bytes(b"keep")
            output = root / "output.png"
            os.link(target, output)
            processor = Mock(config=AppConfig())
            processor.process_frame.side_effect = lambda frame, **kwargs: frame
            with self.assertRaises(ValueError):
                _export_image(source, output, processor)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_event_append_rejects_hardlink_without_changing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("private")
            destination = root / "events.jsonl"
            os.link(target, destination)
            with self.assertRaises(ValueError):
                EventRecorder(destination)
            self.assertEqual(target.read_text(), "private")

    def test_diagnostic_probe_does_not_overwrite_predictable_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / f".write-test-{os.getpid()}"
            probe.write_text("keep")
            self.assertTrue(_check_writable_directory(root, "test").passed)
            self.assertTrue(probe.exists())
            self.assertEqual(probe.read_text(), "keep")

    def test_diagnostics_use_private_directory_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "managed"
            with patch("data_policy._restrict_access", side_effect=OSError("ACL failed")):
                self.assertFalse(_check_writable_directory(root, "test").passed)
            self.assertFalse(root.exists())

    def test_atomic_output_preserves_old_file_on_failure_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.png"
            output.write_bytes(b"old")
            with self.assertRaisesRegex(RuntimeError, "encoder"):
                with data_policy.atomic_output(output) as temporary:
                    self.assertEqual(temporary.suffix, ".png")
                    temporary.write_bytes(b"partial")
                    raise RuntimeError("encoder")
            self.assertEqual(output.read_bytes(), b"old")
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_atomic_output_rechecks_destination_before_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"keep")
            destination = root / "out"
            with self.assertRaises(ValueError):
                with data_policy.atomic_output(destination) as temporary:
                    temporary.write_bytes(b"new")
                    os.link(target, destination)
            self.assertEqual(target.read_bytes(), b"keep")

    @unittest.skipIf(os.name == "nt", "POSIX modes; Windows ACL covered separately")
    def test_new_event_file_is_private_even_in_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.close()
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_event_writer_rejects_linked_file_and_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            linked = root / "linked"
            try:
                linked.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("Symlink creation unavailable")
            with self.assertRaises(ValueError):
                EventRecorder(linked / "events.jsonl")
            self.assertEqual(list(target.iterdir()), [])


class RecognitionPrivacyTest(unittest.TestCase):
    def test_injected_recorder_obeys_processor_privacy(self):
        from detection import MonitoringProcessor

        with tempfile.TemporaryDirectory() as directory:
            recorder = EventRecorder(Path(directory) / "events.jsonl")
            with patch("detection.configure_logging"):
                processor = MonitoringProcessor(
                    replace(AppConfig(), identity_enabled=False, expression_enabled=False),
                    pose_analyzer=Mock(),
                    event_recorder=recorder,
                )
            try:
                self.assertFalse(recorder.identity_enabled)
                self.assertFalse(recorder.expression_enabled)
            finally:
                processor.close()

    def test_switches_gate_supplied_analyzers_independently(self):
        from detection import MonitoringProcessor

        for identity_enabled, expression_enabled in ((False, True), (True, False)):
            identity, expression = Mock(), Mock()
            config = replace(
                AppConfig(),
                identity_enabled=identity_enabled,
                expression_enabled=expression_enabled,
            )
            with patch("detection.configure_logging"):
                processor = MonitoringProcessor(
                    config,
                    pose_analyzer=Mock(),
                    identity_matcher=identity,
                    emotion_analyzer=expression,
                    known_identities=[],
                    event_recorder=EventRecorder(None),
                )
            self.assertIs(processor.identity_matcher, identity if identity_enabled else None)
            self.assertIs(processor.emotion_analyzer, expression if expression_enabled else None)
            processor.close()

    def test_cli_pose_only_overrides_environment_and_activity(self):
        import main

        with (
            patch.dict(
                os.environ,
                {"MONITOR_IDENTITY_ENABLED": "true", "MONITOR_EXPRESSION_ENABLED": "true"},
            ),
            patch.object(
                sys, "argv", ["monitor", "--detect", "--pose-only", "--activity-model", "mlp"]
            ),
            patch("detection.run_detection", return_value=True) as run,
        ):
            main.main()
        config = run.call_args.kwargs["config"]
        self.assertFalse(config.identity_enabled)
        self.assertFalse(config.expression_enabled)
        self.assertEqual(config.activity_model, "none")
        self.assertFalse(config.allow_unchecked_enrollment)

    def test_switches_are_independent_and_defaults_preserve_features(self):
        with patch.dict(
            os.environ, {"MONITOR_IDENTITY_ENABLED": "true", "MONITOR_EXPRESSION_ENABLED": "true"}
        ):
            self.assertTrue(AppConfig.from_env().identity_enabled)
            self.assertTrue(AppConfig.from_env().expression_enabled)
        for identity, expression in (("false", "true"), ("true", "false")):
            with patch.dict(
                os.environ,
                {"MONITOR_IDENTITY_ENABLED": identity, "MONITOR_EXPRESSION_ENABLED": expression},
            ):
                config = AppConfig.from_env()
            self.assertEqual(config.identity_enabled, identity == "true")
            self.assertEqual(config.expression_enabled, expression == "true")

    def test_enrollment_detection_failure_rejects_frame(self):
        identifier = Mock()
        identifier.detect_faces.side_effect = RuntimeError("detector failed")
        accepted, _ = validate_enrollment_frame(identifier, None)
        self.assertFalse(accepted)

    def test_unavailable_enrollment_validation_rejects_image(self):
        accepted, _ = validate_enrollment_frame(None, np.zeros((8, 8, 3), np.uint8))
        self.assertFalse(accepted)

    def test_unchecked_enrollment_requires_explicit_opt_in(self):
        accepted, _ = validate_enrollment_frame(None, None, allow_unchecked=True)
        self.assertTrue(accepted)

    def test_failed_enrollment_initialization_never_opens_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            session = EnrollmentSession(folder=Path(directory), target_image_count=1)
            with (
                patch("enrollment._create_enrollment_identifier", return_value=None),
                patch(
                    "enrollment.prompt_camera_source", side_effect=AssertionError("camera prompt")
                ),
            ):
                self.assertEqual(
                    capture_enrollment_images(session, AppConfig()), MenuState.ENROLL_ABORT
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_disabled_modules_do_not_initialize_or_persist_identity(self):
        from detection import MonitoringProcessor
        from pose import PoseResult
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                AppConfig(),
                identity_enabled=False,
                expression_enabled=False,
                log_dir=root / "logs",
                event_dir=root / "events",
            )
            pose = Mock()
            pose.analyze.return_value = [PoseResult(7, (0, 0, 8, 8), {})]
            with (
                patch("detection.configure_logging"),
                patch("detection.ensure_models", side_effect=AssertionError("identity download")),
                patch("detection.ensure_model", side_effect=AssertionError("expression download")),
                patch(
                    "detection.OpenCVFaceIdentifier", side_effect=AssertionError("identity init")
                ),
            ):
                processor = MonitoringProcessor(config, pose_analyzer=pose)
                processor.process_frame(np.zeros((8, 8, 3), np.uint8), timestamp=0)
                processor.close()
            payload = json.loads(next(config.event_dir.glob("*.jsonl")).read_text().splitlines()[0])
            self.assertNotIn("identity_name", payload)
            self.assertIsNone(processor.identity_matcher)
            self.assertIsNone(processor.emotion_analyzer)

    def test_diagnostics_prepare_only_enabled_models(self):
        from diagnostics import run_diagnostics

        config = replace(AppConfig(), identity_enabled=False, expression_enabled=False)
        prepared = []
        with (
            patch("diagnostics._check_writable_directory"),
            patch(
                "diagnostics.ensure_model",
                side_effect=lambda spec, config: prepared.append(spec.name),
            ),
        ):
            run_diagnostics(config, prepare_models=True)
        self.assertEqual(prepared, ["YOLO pose model"])


if __name__ == "__main__":
    unittest.main()
