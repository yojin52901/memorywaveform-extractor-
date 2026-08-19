"""Local runtime settings and factory functions for one extraction process."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

from memorywaveform_extractor.application.extract import ExtractionService
from memorywaveform_extractor.application.jobs import JobCoordinator
from memorywaveform_extractor.infrastructure.artifacts import ArtifactStore
from memorywaveform_extractor.infrastructure.sqlite_jobs import SQLiteJobRepository
from memorywaveform_extractor.providers.ollama import OllamaVisionProvider
from memorywaveform_extractor.providers.openai_compatible import OpenAICompatibleVisionProvider
from memorywaveform_extractor.providers.tesseract import TesseractOcrProvider
from memorywaveform_extractor.strategies.hybrid import HybridStrategy
from memorywaveform_extractor.strategies.vision import VisionStrategy


@dataclass(frozen=True)
class Settings:
    artifact_root: Path
    database_path: Path
    vision_provider: str
    cors_origins: tuple[str, ...]
    confidence_threshold: float

    @classmethod
    def from_environment(cls) -> Settings:
        artifact_root = Path(os.environ.get("ARTIFACT_ROOT", ".artifacts"))
        configured_database_path = os.environ.get("JOB_DATABASE_PATH")
        database_path = (
            Path(configured_database_path)
            if configured_database_path
            else artifact_root / "jobs.sqlite3"
        )
        return cls(
            artifact_root=artifact_root,
            database_path=database_path,
            vision_provider=os.environ.get("VISION_PROVIDER", "ollama"),
            cors_origins=_cors_origins_from_environment(),
            confidence_threshold=_confidence_threshold_from_environment(),
        )

    def with_artifact_root(self, artifact_root: Path) -> Settings:
        return replace(
            self,
            artifact_root=artifact_root,
            database_path=artifact_root / "jobs.sqlite3",
        )


def build_job_coordinator(settings: Settings) -> JobCoordinator:
    """Build the same local application service used by CLI and HTTP adapters."""

    vision_provider = _build_vision_provider(settings)
    vision_strategy = VisionStrategy(vision_provider)
    hybrid_strategy = HybridStrategy(
        TesseractOcrProvider(),
        vision_provider,
    )
    extraction_service = ExtractionService(
        vision_strategy,
        hybrid_strategy,
        confidence_threshold=settings.confidence_threshold,
    )
    return JobCoordinator(
        repository=SQLiteJobRepository(settings.database_path),
        artifacts=ArtifactStore(settings.artifact_root),
        extraction_service=extraction_service,
    )


def _build_vision_provider(
    settings: Settings,
) -> OllamaVisionProvider | OpenAICompatibleVisionProvider:
    if settings.vision_provider == "ollama":
        return OllamaVisionProvider.from_environment()
    if settings.vision_provider in {"openai-compatible", "openai_compatible"}:
        return OpenAICompatibleVisionProvider.from_environment()
    raise ValueError(
        "VISION_PROVIDER must be 'ollama' or 'openai-compatible', "
        f"not {settings.vision_provider!r}."
    )


def _cors_origins_from_environment() -> tuple[str, ...]:
    configured_origins = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return tuple(origin.strip() for origin in configured_origins.split(",") if origin.strip())


def _confidence_threshold_from_environment() -> float:
    raw_value = os.environ.get("RELATION_CONFIDENCE_THRESHOLD", "0.7")
    try:
        confidence_threshold = float(raw_value)
    except ValueError as error:
        raise ValueError("RELATION_CONFIDENCE_THRESHOLD must be a number.") from error
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("RELATION_CONFIDENCE_THRESHOLD must be between 0 and 1.")
    return confidence_threshold
