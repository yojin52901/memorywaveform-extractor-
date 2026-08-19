"""A small SQLite-backed repository for local extraction-job metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

from memorywaveform_extractor.domain.models import ExtractionJob


class SQLiteJobRepository:
    """Persists complete job snapshots without coupling callers to SQL rows."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )

    def create(self, job: ExtractionJob) -> ExtractionJob:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO extraction_jobs (id, payload) VALUES (?, ?)",
                    (str(job.id), job.model_dump_json()),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Job {job.id} already exists.") from error
        return job

    def get(self, job_id: UUID) -> ExtractionJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM extraction_jobs WHERE id = ?", (str(job_id),)
            ).fetchone()
        return None if row is None else ExtractionJob.model_validate_json(row[0])

    def update(self, job: ExtractionJob) -> ExtractionJob:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE extraction_jobs SET payload = ? WHERE id = ?",
                (job.model_dump_json(), str(job.id)),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Job {job.id} does not exist.")
        return job

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)
