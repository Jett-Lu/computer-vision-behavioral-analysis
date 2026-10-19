"""Prepare HMDB51 pose and person-crop caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_recognition.dataset import (
    build_hmdb51_fold,
    build_hmdb51_metadata_fold,
    download_hmdb51_subset,
    write_manifest,
)
from activity_recognition.preprocessing import cache_activity_samples
from activity_recognition.sampling import DEFAULT_SAMPLING_INTERVAL_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare four-class HMDB51 activity caches.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path)
    parser.add_argument("--metadata-train", type=Path)
    parser.add_argument("--metadata-test", type=Path)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "activity_cache",
    )
    parser.add_argument("--fold", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--sampling-interval-seconds", type=float, default=DEFAULT_SAMPLING_INTERVAL_SECONDS
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.absolute()
    metadata_mode = args.metadata_train is not None or args.metadata_test is not None
    if metadata_mode:
        if args.metadata_train is None or args.metadata_test is None:
            raise SystemExit("--metadata-train and --metadata-test must be used together")
        if args.fold != 1:
            raise SystemExit("Metadata mode currently supports official fold 1 only")
        if args.download_videos:
            download_counts = download_hmdb51_subset(
                (args.metadata_train, args.metadata_test),
                args.dataset_root,
                overwrite=args.overwrite,
            )
            print(f"Download summary: {download_counts}")
            if download_counts["failed"]:
                raise SystemExit(1)
        samples = build_hmdb51_metadata_fold(
            args.dataset_root,
            args.metadata_train,
            args.metadata_test,
            cache_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        annotations_root = args.metadata_train.parent
    else:
        if args.annotations_root is None:
            raise SystemExit("Provide --annotations-root or both metadata CSV arguments")
        samples = build_hmdb51_fold(
            args.dataset_root,
            args.annotations_root,
            cache_dir,
            fold=args.fold,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        annotations_root = args.annotations_root
    manifest_path = cache_dir / "manifest.json"
    counts = cache_activity_samples(
        samples,
        frames_per_sample=args.frames,
        overwrite=args.overwrite,
        sampling_interval_seconds=args.sampling_interval_seconds,
    )
    print(f"Cache summary: {counts}")
    if counts["failed"]:
        raise SystemExit(1)
    write_manifest(
        manifest_path,
        samples,
        dataset_root=args.dataset_root,
        annotations_root=annotations_root,
        fold=args.fold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        frames_per_sample=args.frames,
        sampling_interval_seconds=args.sampling_interval_seconds,
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
