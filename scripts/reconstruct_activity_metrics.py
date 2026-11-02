"""Reconstruct saved prediction metrics without models, training or video access."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from activity_recognition.dataset import load_manifest
from activity_recognition.evidence import reconstruct_predictions
from activity_recognition.metrics import write_metrics_json
from activity_recognition.provenance import require_fresh_outputs
from data_policy import atomic_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    samples = load_manifest(args.manifest)[1] if args.manifest else None
    report = reconstruct_predictions(args.predictions, samples)
    require_fresh_outputs(args.output_dir, ["reconstructed.json", "reconstructed.csv"])
    write_metrics_json(args.output_dir / "reconstructed.json", report)
    with (
        atomic_output(args.output_dir / "reconstructed.csv") as temp,
        temp.open("w", newline="", encoding="utf-8") as stream,
    ):
        fields = (
            "model",
            "sample_count",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, metrics in report["models"].items():
            writer.writerow({"model": model, **{key: metrics[key] for key in fields[1:]}})
    print(f"Reconstructed {len(report['models'])} model reports; no inference performed.")


if __name__ == "__main__":
    main()
