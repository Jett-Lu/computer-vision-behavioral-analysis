"""Application entrypoint for the integrated monitoring demo."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

from config import AppConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated live monitoring demo.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--secure-data",
        action="store_true",
        help="Restrict managed data directories to the current OS account.",
    )
    mode.add_argument(
        "--retention",
        action="store_true",
        help="Preview expired managed data. Stop monitoring before maintenance.",
    )
    parser.add_argument(
        "--apply-retention",
        action="store_true",
        help="Delete expired files; requires --retention. Default is preview only.",
    )
    mode.add_argument(
        "--detect",
        action="store_true",
        help="Start live monitoring directly instead of opening the menu.",
    )
    mode.add_argument(
        "--process-media",
        action="store_true",
        help="Write annotated photos or videos from the input folder to the output folder.",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run recorded validation scenarios from a JSON manifest.",
    )
    mode.add_argument(
        "--doctor",
        action="store_true",
        help="Check package, model, and filesystem readiness.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or stream URL used with --detect.",
    )
    parser.add_argument(
        "--input",
        default="input",
        help="Photo, video, or folder used with --process-media. Default: input",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder for annotated files created with --process-media. Default: output",
    )
    parser.add_argument(
        "--manifest",
        default="validation/manifest.json",
        help="Validation manifest used with --validate.",
    )
    parser.add_argument(
        "--report",
        default="validation/reports/latest.json",
        help="Validation report path used with --validate.",
    )
    parser.add_argument(
        "--prepare-models",
        action="store_true",
        help="Download and verify models while running --doctor.",
    )
    parser.add_argument(
        "--no-model-downloads",
        action="store_true",
        help="Run without downloading missing models.",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable JSONL operator event reports.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Override the MONITOR_LOG_LEVEL setting.",
    )
    parser.add_argument(
        "--activity-model",
        choices=("none", "mlp"),
        help="Optional live activity model. Default: none",
    )
    parser.add_argument(
        "--activity-checkpoint",
        type=Path,
        help="Path to the MLP activity checkpoint.",
    )
    parser.add_argument(
        "--activity-sequence-length",
        type=_positive_int,
        help="Number of valid tracked poses required for activity inference.",
    )
    parser.add_argument(
        "--activity-confidence-threshold",
        type=_unit_interval,
        help="Minimum smoothed confidence for a named activity.",
    )
    parser.add_argument(
        "--activity-inference-interval",
        type=_positive_int,
        help="Minimum frames between activity inferences for one track.",
    )
    parser.add_argument(
        "--activity-smoothing-window",
        type=_positive_int,
        help="Number of activity probability vectors to average.",
    )
    parser.add_argument(
        "--no-identity",
        action="store_true",
        help="Disable face identity models and identity reporting.",
    )
    parser.add_argument(
        "--no-expression",
        action="store_true",
        help="Disable facial-expression models and reporting.",
    )
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Anonymous pose mode: disable identity, expression, and activity models.",
    )
    parser.add_argument(
        "--allow-unchecked-enrollment",
        action="store_true",
        help="Explicitly permit enrollment without automatic face-quality checks.",
    )
    args = parser.parse_args()
    if args.apply_retention and not args.retention:
        parser.error("--apply-retention requires --retention")
    return args


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    if args.secure_data or args.retention:
        from data_policy import apply_retention, plan_retention, retention_rules, secure_data

        if args.secure_data:
            secure_data(config)
            print("Managed output, log, event and enrollment directories restricted.")
        else:
            plan = plan_retention(retention_rules(config))
            for item in plan:
                print(f"{item.category}: {item.path}")
            print(f"{len(plan)} expired files; {sum(item.size for item in plan)} bytes.")
            if args.apply_retention:
                print(f"Deleted {apply_retention(plan)} expired files.")
            else:
                print("Preview only. No data deleted. Use --retention --apply-retention to apply.")
        return
    config = replace(
        config,
        identity_enabled=config.identity_enabled and not (args.no_identity or args.pose_only),
        expression_enabled=config.expression_enabled and not (args.no_expression or args.pose_only),
        allow_unchecked_enrollment=args.allow_unchecked_enrollment,
        allow_model_downloads=(False if args.no_model_downloads else config.allow_model_downloads),
        event_logging_enabled=(False if args.no_events else config.event_logging_enabled),
        log_level=args.log_level or config.log_level,
        activity_model="none" if args.pose_only else (args.activity_model or config.activity_model),
        activity_checkpoint_path=(
            args.activity_checkpoint.resolve()
            if args.activity_checkpoint is not None
            else config.activity_checkpoint_path
        ),
        activity_sequence_length=(
            args.activity_sequence_length
            if args.activity_sequence_length is not None
            else config.activity_sequence_length
        ),
        activity_confidence_threshold=(
            args.activity_confidence_threshold
            if args.activity_confidence_threshold is not None
            else config.activity_confidence_threshold
        ),
        activity_inference_interval=(
            args.activity_inference_interval
            if args.activity_inference_interval is not None
            else config.activity_inference_interval
        ),
        activity_smoothing_window=(
            args.activity_smoothing_window
            if args.activity_smoothing_window is not None
            else config.activity_smoothing_window
        ),
    )
    if args.doctor:
        from diagnostics import print_diagnostics, run_diagnostics

        checks = run_diagnostics(config, prepare_models=args.prepare_models)
        print_diagnostics(checks)
        if not all(check.passed for check in checks):
            raise SystemExit(1)
        return
    if args.validate:
        from validation import print_validation_summary, run_validation

        results = run_validation(Path(args.manifest), Path(args.report), config)
        print_validation_summary(results)
        if not all(result.passed for result in results):
            raise SystemExit(1)
        return
    if args.process_media:
        from media_export import export_media

        export_media(Path(args.input), Path(args.output_dir), config=config, fail_on_error=True)
        return
    if args.detect:
        from camera import normalize_camera_source
        from detection import run_detection

        if not run_detection(source=normalize_camera_source(args.source), config=config):
            raise SystemExit(1)
        return

    from enrollment import main as menu_main

    menu_main(config)


if __name__ == "__main__":
    main()
