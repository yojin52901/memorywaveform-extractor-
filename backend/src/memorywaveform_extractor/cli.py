"""Command-line entry point sharing the same extraction service as the HTTP API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.application.settings import Settings, build_job_coordinator
from memorywaveform_extractor.domain.models import ExtractionMode, ExtractionResult, JobStatus
from memorywaveform_extractor.infrastructure.images import MAX_SOURCE_BYTES, ImageDecodingError


SUPPORTED_INPUT_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png"})


def extract_file(
    input_path: Path,
    mode: ExtractionMode,
    settings: Settings | None = None,
    *,
    coordinator: JobCoordinator | None = None,
) -> ExtractionResult:
    """Run one file through the canonical job lifecycle and return its JSON result."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    active_coordinator = _resolve_coordinator(settings, coordinator)
    return _extract_file_with_coordinator(input_path, mode, active_coordinator)


def extract_paths(
    input_path: Path,
    mode: ExtractionMode,
    settings: Settings | None = None,
    *,
    coordinator: JobCoordinator | None = None,
) -> list[ExtractionResult]:
    """Extract one supported image or every supported file in one directory."""

    active_coordinator = _resolve_coordinator(settings, coordinator)
    if input_path.is_file():
        return [_extract_file_with_coordinator(input_path, mode, active_coordinator)]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    image_paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No PNG, JPG, or GIF images were found in: {input_path}")
    return [
        _extract_file_with_coordinator(path, mode, active_coordinator)
        for path in image_paths
    ]


def _extract_file_with_coordinator(
    input_path: Path,
    mode: ExtractionMode,
    coordinator: JobCoordinator,
) -> ExtractionResult:
    job = coordinator.submit(_read_input_source(input_path), input_path.name, mode)
    if job.status is JobStatus.FAILED:
        raise RuntimeError(job.error_message or "Input validation failed without an error message.")
    completed_job = coordinator.run(job.id)
    if completed_job.status is JobStatus.FAILED:
        raise RuntimeError(
            completed_job.error_message or "Extraction failed without an error message."
        )
    if completed_job.result_path is None:
        raise RuntimeError("Extraction completed without a result artifact.")
    return ExtractionResult.model_validate_json(
        Path(completed_job.result_path).read_text("utf-8")
    )


def _read_input_source(input_path: Path) -> bytes:
    """Read no more than the shared upload byte limit from a CLI input path."""

    if input_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ImageDecodingError(
            f"The uploaded file exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MiB limit."
        )
    with input_path.open("rb") as source:
        contents = source.read(MAX_SOURCE_BYTES + 1)
    if len(contents) > MAX_SOURCE_BYTES:
        raise ImageDecodingError(
            f"The uploaded file exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MiB limit."
        )
    return contents


def _resolve_coordinator(
    settings: Settings | None,
    coordinator: JobCoordinator | None,
) -> JobCoordinator:
    if coordinator is not None and settings is not None:
        raise ValueError("Pass either settings or a coordinator, not both.")
    return coordinator or build_job_coordinator(settings or Settings.from_environment())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local CLI and print canonical JSON to stdout."""

    parser = argparse.ArgumentParser(prog="memorywaveform-extract")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser(
        "extract",
        help="extract timing relations from one image",
    )
    extract_parser.add_argument(
        "input",
        type=Path,
        help="PNG, JPG, or GIF timing-diagram image file or directory",
    )
    extract_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ExtractionMode],
        default=ExtractionMode.HYBRID.value,
        help="extraction mode (default: hybrid)",
    )
    extract_parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts"),
        help="directory for input, JSON, annotation, and local job metadata",
    )

    arguments = parser.parse_args(argv)
    settings = Settings.from_environment().with_artifact_root(arguments.output)
    results = extract_paths(
        arguments.input,
        ExtractionMode(arguments.mode),
        settings,
    )
    if len(results) == 1:
        print(results[0].model_dump_json(indent=2))
    else:
        print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
