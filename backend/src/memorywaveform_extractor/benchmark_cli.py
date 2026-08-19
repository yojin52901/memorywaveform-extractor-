"""Command-line entry point for local sample benchmark reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from memorywaveform_extractor.benchmark import run_benchmark, write_benchmark_report
from memorywaveform_extractor.domain.models import ExtractionMode


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured local extractor against one sample manifest."""

    parser = argparse.ArgumentParser(prog="memorywaveform-benchmark")
    parser.add_argument("samples", type=Path, help="directory containing manifest.json")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ExtractionMode],
        default=ExtractionMode.HYBRID.value,
        help="extraction mode (default: hybrid)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/benchmark-report.json"),
        help="JSON file to write (default: .artifacts/benchmark-report.json)",
    )
    parser.add_argument("--model-id", help="model identifier to record in the report")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="relations below this confidence count as review-needed (default: 0.7)",
    )
    arguments = parser.parse_args(argv)

    report = run_benchmark(
        arguments.samples,
        ExtractionMode(arguments.mode),
        model_id=arguments.model_id,
        confidence_threshold=arguments.confidence_threshold,
    )
    output_path = write_benchmark_report(report, arguments.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
