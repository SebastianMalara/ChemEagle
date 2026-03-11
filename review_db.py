from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class ReviewRepository:
    SCHEMA_VERSION = 3

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    source_set_fingerprint TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    profile_label TEXT NOT NULL,
                    ingest_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    config_snapshot_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    main_provider TEXT NOT NULL DEFAULT '',
                    main_model TEXT NOT NULL DEFAULT '',
                    ocr_provider TEXT NOT NULL DEFAULT '',
                    ocr_model TEXT NOT NULL DEFAULT '',
                    pdf_model_size TEXT NOT NULL DEFAULT '',
                    run_mode TEXT NOT NULL DEFAULT '',
                    artifact_store_type TEXT NOT NULL DEFAULT '',
                    current_phase TEXT NOT NULL DEFAULT '',
                    current_run_source_id TEXT NOT NULL DEFAULT '',
                    current_source_name TEXT NOT NULL DEFAULT '',
                    status_message TEXT NOT NULL DEFAULT '',
                    last_error_summary TEXT NOT NULL DEFAULT '',
                    completed_sources INTEGER NOT NULL DEFAULT 0,
                    failed_sources INTEGER NOT NULL DEFAULT 0,
                    last_progress_at TEXT NOT NULL DEFAULT '',
                    preflight_status TEXT NOT NULL DEFAULT '',
                    preflight_summary TEXT NOT NULL DEFAULT '',
                    systemic_failure_kind TEXT NOT NULL DEFAULT '',
                    systemic_failure_count INTEGER NOT NULL DEFAULT 0,
                    abort_reason TEXT NOT NULL DEFAULT '',
                    log_artifact_key TEXT NOT NULL DEFAULT '',
                    stdout_artifact_key TEXT NOT NULL DEFAULT '',
                    stderr_artifact_key TEXT NOT NULL DEFAULT '',
                    failure_summary TEXT NOT NULL DEFAULT '',
                    last_event_ts TEXT NOT NULL DEFAULT '',
                    last_event_level TEXT NOT NULL DEFAULT '',
                    total_sources INTEGER NOT NULL DEFAULT 0,
                    total_derived_images INTEGER NOT NULL DEFAULT 0,
                    total_reactions INTEGER NOT NULL DEFAULT 0,
                    total_failures INTEGER NOT NULL DEFAULT 0,
                    total_redo INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL,
                    usage_completeness TEXT NOT NULL DEFAULT 'none',
                    elapsed_seconds REAL
                );

                CREATE TABLE IF NOT EXISTS source_assets (
                    source_asset_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    sha256 TEXT,
                    original_filename TEXT NOT NULL,
                    artifact_backend TEXT NOT NULL DEFAULT '',
                    artifact_key TEXT NOT NULL DEFAULT '',
                    artifact_status TEXT NOT NULL DEFAULT 'present'
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_source_assets_sha256
                    ON source_assets(sha256)
                    WHERE sha256 IS NOT NULL;

                CREATE TABLE IF NOT EXISTS run_sources (
                    run_source_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    source_asset_id TEXT NOT NULL REFERENCES source_assets(source_asset_id) ON DELETE CASCADE,
                    input_order INTEGER NOT NULL DEFAULT 0,
                    recovery_note TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    current_phase TEXT NOT NULL DEFAULT '',
                    status_message TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    expected_derived_images INTEGER NOT NULL DEFAULT 0,
                    completed_derived_images INTEGER NOT NULL DEFAULT 0,
                    successful_derived_images INTEGER NOT NULL DEFAULT 0,
                    failed_derived_images INTEGER NOT NULL DEFAULT 0,
                    reaction_count INTEGER NOT NULL DEFAULT 0,
                    redo_count INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL,
                    usage_completeness TEXT NOT NULL DEFAULT 'none',
                    error_summary TEXT NOT NULL DEFAULT '',
                    last_error_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS derived_images (
                    derived_image_id TEXT PRIMARY KEY,
                    run_source_id TEXT NOT NULL REFERENCES run_sources(run_source_id) ON DELETE CASCADE,
                    page_hint TEXT NOT NULL DEFAULT '',
                    image_index INTEGER NOT NULL DEFAULT 0,
                    artifact_backend TEXT NOT NULL DEFAULT '',
                    artifact_key TEXT NOT NULL DEFAULT '',
                    artifact_status TEXT NOT NULL DEFAULT 'present',
                    status TEXT NOT NULL DEFAULT 'queued',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    status_message TEXT NOT NULL DEFAULT '',
                    outcome_class TEXT NOT NULL DEFAULT '',
                    raw_artifact_key TEXT NOT NULL DEFAULT '',
                    error_text TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_id TEXT NOT NULL DEFAULT '',
                    accepted_reaction_count INTEGER NOT NULL DEFAULT 0,
                    rejected_reaction_count INTEGER NOT NULL DEFAULT 0,
                    normalization_status TEXT NOT NULL DEFAULT '',
                    normalization_summary TEXT NOT NULL DEFAULT '',
                    last_retry_reason TEXT NOT NULL DEFAULT '',
                    reaction_count INTEGER NOT NULL DEFAULT 0,
                    redo_count INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL
                );

                CREATE TABLE IF NOT EXISTS derived_image_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    derived_image_id TEXT NOT NULL REFERENCES derived_images(derived_image_id) ON DELETE CASCADE,
                    attempt_no INTEGER NOT NULL,
                    trigger TEXT NOT NULL DEFAULT '',
                    execution_mode TEXT NOT NULL DEFAULT 'normal',
                    status TEXT NOT NULL DEFAULT 'queued',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    failure_kind TEXT NOT NULL DEFAULT '',
                    error_summary TEXT NOT NULL DEFAULT '',
                    raw_artifact_key TEXT NOT NULL DEFAULT '',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL,
                    usage_completeness TEXT NOT NULL DEFAULT 'none',
                    config_snapshot_json TEXT NOT NULL DEFAULT '',
                    retry_of_attempt_id TEXT NOT NULL DEFAULT ''
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_derived_image_attempts_number
                    ON derived_image_attempts(derived_image_id, attempt_no);

                CREATE TABLE IF NOT EXISTS reactions (
                    reaction_uid TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    run_source_id TEXT NOT NULL REFERENCES run_sources(run_source_id) ON DELETE CASCADE,
                    derived_image_id TEXT NOT NULL REFERENCES derived_images(derived_image_id) ON DELETE CASCADE,
                    attempt_id TEXT NOT NULL DEFAULT '',
                    reaction_id TEXT NOT NULL,
                    reaction_fingerprint TEXT NOT NULL DEFAULT '',
                    outcome_class TEXT NOT NULL,
                    structure_quality TEXT NOT NULL DEFAULT '',
                    acceptance_reason TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'unchecked',
                    review_notes TEXT NOT NULL DEFAULT '',
                    render_artifact_key TEXT NOT NULL DEFAULT '',
                    raw_reaction_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reactions_run_id ON reactions(run_id);
                CREATE INDEX IF NOT EXISTS idx_reactions_fingerprint ON reactions(reaction_fingerprint);

                CREATE TABLE IF NOT EXISTS reaction_molecules (
                    molecule_id TEXT PRIMARY KEY,
                    reaction_uid TEXT NOT NULL REFERENCES reactions(reaction_uid) ON DELETE CASCADE,
                    side TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    smiles TEXT NOT NULL DEFAULT '',
                    label TEXT NOT NULL DEFAULT '',
                    valid_smiles INTEGER NOT NULL DEFAULT 0,
                    validation_kind TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS reaction_conditions (
                    condition_id TEXT PRIMARY KEY,
                    reaction_uid TEXT NOT NULL REFERENCES reactions(reaction_uid) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    smiles TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS reaction_additional_info (
                    info_id TEXT PRIMARY KEY,
                    reaction_uid TEXT NOT NULL REFERENCES reactions(reaction_uid) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT '',
                    key TEXT NOT NULL DEFAULT '',
                    value TEXT NOT NULL DEFAULT '',
                    raw_text TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS llm_call_metrics (
                    llm_call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    run_source_id TEXT DEFAULT '',
                    derived_image_id TEXT DEFAULT '',
                    phase TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    usage_prompt_tokens INTEGER,
                    usage_completion_tokens INTEGER,
                    usage_total_tokens INTEGER,
                    estimated_cost_usd REAL,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 1,
                    raw_usage_json TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_llm_call_metrics_run_id ON llm_call_metrics(run_id);
                """
            )
            conn.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(self.SCHEMA_VERSION),),
            )
            self._ensure_column(conn, "runs", "current_phase", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "current_run_source_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "current_source_name", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "status_message", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "last_error_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "completed_sources", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "runs", "failed_sources", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "runs", "last_progress_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "preflight_status", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "preflight_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "systemic_failure_kind", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "systemic_failure_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "runs", "abort_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "log_artifact_key", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "stdout_artifact_key", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "stderr_artifact_key", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "failure_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "last_event_ts", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "runs", "last_event_level", "TEXT NOT NULL DEFAULT ''")

            self._ensure_column(conn, "run_sources", "source_type", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "run_sources", "status", "TEXT NOT NULL DEFAULT 'queued'")
            self._ensure_column(conn, "run_sources", "current_phase", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "run_sources", "status_message", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "run_sources", "started_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "run_sources", "finished_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "run_sources", "expected_derived_images", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "completed_derived_images", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "successful_derived_images", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "failed_derived_images", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "reaction_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "redo_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "prompt_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "completion_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "total_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "run_sources", "estimated_cost_usd", "REAL")
            self._ensure_column(conn, "run_sources", "usage_completeness", "TEXT NOT NULL DEFAULT 'none'")
            self._ensure_column(conn, "run_sources", "error_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "run_sources", "last_error_at", "TEXT NOT NULL DEFAULT ''")

            self._ensure_column(conn, "derived_images", "status", "TEXT NOT NULL DEFAULT 'queued'")
            self._ensure_column(conn, "derived_images", "started_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "finished_at", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "status_message", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "reaction_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "redo_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "attempt_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "last_attempt_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "accepted_reaction_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "rejected_reaction_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "normalization_status", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "normalization_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "last_retry_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "derived_images", "prompt_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "completion_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "total_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "derived_images", "estimated_cost_usd", "REAL")
            self._ensure_column(conn, "reactions", "attempt_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "reactions", "structure_quality", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "reactions", "acceptance_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "reaction_molecules", "validation_kind", "TEXT NOT NULL DEFAULT ''")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column in columns:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def create_experiment(
        self,
        *,
        name: str,
        notes: str = "",
        source_set_fingerprint: str = "",
        status: str = "queued",
    ) -> str:
        experiment_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(experiment_id, name, created_at, notes, source_set_fingerprint, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (experiment_id, name, utcnow_iso(), notes, source_set_fingerprint, status),
            )
            conn.commit()
        return experiment_id

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE experiments SET status = ? WHERE experiment_id = ?", (status, experiment_id))
            conn.commit()

    def create_run(
        self,
        *,
        experiment_id: str,
        profile_label: str,
        ingest_mode: str,
        status: str,
        config_snapshot: Dict[str, Any],
        config_hash: str,
    ) -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, experiment_id, profile_label, ingest_mode, status, created_at,
                    config_snapshot_json, config_hash, main_provider, main_model, ocr_provider,
                    ocr_model, pdf_model_size, run_mode, artifact_store_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    profile_label,
                    ingest_mode,
                    status,
                    utcnow_iso(),
                    json.dumps(config_snapshot, ensure_ascii=False),
                    config_hash,
                    str(config_snapshot.get("llm_provider") or config_snapshot.get("LLM_PROVIDER") or ""),
                    str(config_snapshot.get("llm_model") or config_snapshot.get("LLM_MODEL") or ""),
                    str(config_snapshot.get("ocr_llm_provider") or config_snapshot.get("OCR_LLM_PROVIDER") or ""),
                    str(config_snapshot.get("ocr_llm_model") or config_snapshot.get("OCR_LLM_MODEL") or ""),
                    str(config_snapshot.get("pdf_model_size") or config_snapshot.get("PDF_MODEL_SIZE") or ""),
                    str(config_snapshot.get("mode") or config_snapshot.get("CHEMEAGLE_RUN_MODE") or ""),
                    str(config_snapshot.get("artifact_backend") or config_snapshot.get("ARTIFACT_BACKEND") or "filesystem"),
                ),
            )
            conn.commit()
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _dict_from_row(row) if row else None

    def update_run_status(self, run_id: str, status: str, *, started: bool = False, finished: bool = False) -> None:
        with self.connect() as conn:
            if started:
                now = utcnow_iso()
                conn.execute(
                    "UPDATE runs SET status = ?, started_at = ?, last_progress_at = ? WHERE run_id = ?",
                    (status, now, now, run_id),
                )
            elif finished:
                now = utcnow_iso()
                conn.execute(
                    "UPDATE runs SET status = ?, finished_at = ?, last_progress_at = ? WHERE run_id = ?",
                    (status, now, now, run_id),
                )
            else:
                conn.execute(
                    "UPDATE runs SET status = ?, last_progress_at = ? WHERE run_id = ?",
                    (status, utcnow_iso(), run_id),
                )
            conn.commit()

    def mark_running_runs_interrupted(
        self,
        *,
        reason: str = "Application stopped while the run was active.",
    ) -> List[Dict[str, str]]:
        recovered: List[Dict[str, str]] = []
        now = utcnow_iso()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, experiment_id, current_run_source_id
                FROM runs
                WHERE status = 'running'
                """
            ).fetchall()
            for row in rows:
                run_id = str(row["run_id"])
                experiment_id = str(row["experiment_id"])
                current_run_source_id = str(row["current_run_source_id"] or "")
                conn.execute(
                    """
                    UPDATE runs
                    SET status = 'interrupted',
                        finished_at = ?,
                        current_phase = 'interrupted',
                        status_message = ?,
                        abort_reason = ?,
                        failure_summary = CASE
                            WHEN failure_summary = '' THEN ?
                            ELSE failure_summary
                        END,
                        last_error_summary = ?,
                        last_progress_at = ?
                    WHERE run_id = ?
                    """,
                    (now, reason, reason, reason, reason, now, run_id),
                )
                if current_run_source_id:
                    conn.execute(
                        """
                        UPDATE run_sources
                        SET status = 'aborted',
                            current_phase = 'finalize',
                            status_message = ?,
                            error_summary = CASE
                                WHEN error_summary = '' THEN ?
                                ELSE error_summary
                            END,
                            finished_at = CASE
                                WHEN finished_at = '' THEN ?
                                ELSE finished_at
                            END,
                            last_error_at = CASE
                                WHEN last_error_at = '' THEN ?
                                ELSE last_error_at
                            END
                        WHERE run_source_id = ?
                          AND status NOT IN ('completed', 'failed', 'aborted', 'skipped')
                        """,
                        (reason, reason, now, now, current_run_source_id),
                    )
                    conn.execute(
                        """
                        UPDATE derived_images
                        SET status = CASE
                                WHEN status = 'processing' THEN 'failed'
                                ELSE 'skipped'
                            END,
                            status_message = ?,
                            error_text = CASE
                                WHEN status = 'processing' THEN ?
                                ELSE error_text
                            END,
                            finished_at = CASE
                                WHEN finished_at = '' THEN ?
                                ELSE finished_at
                            END
                        WHERE run_source_id = ?
                          AND status NOT IN ('completed', 'failed', 'skipped')
                        """,
                        (reason, reason, now, current_run_source_id),
                    )
                conn.execute(
                    """
                    UPDATE run_sources
                    SET status = 'skipped',
                        current_phase = 'finalize',
                        status_message = ?,
                        finished_at = CASE
                            WHEN finished_at = '' THEN ?
                            ELSE finished_at
                        END
                    WHERE run_id = ?
                      AND run_source_id != ?
                      AND status NOT IN ('completed', 'failed', 'aborted', 'skipped')
                    """,
                    (reason, now, run_id, current_run_source_id),
                )
                conn.execute(
                    """
                    UPDATE derived_images
                    SET status = 'skipped',
                        status_message = ?,
                        finished_at = CASE
                            WHEN finished_at = '' THEN ?
                            ELSE finished_at
                        END
                    WHERE run_source_id IN (
                        SELECT run_source_id
                        FROM run_sources
                        WHERE run_id = ?
                    )
                      AND status NOT IN ('completed', 'failed', 'skipped')
                    """,
                    (reason, now, run_id),
                )
                recovered.append({"run_id": run_id, "experiment_id": experiment_id})
            conn.commit()
        return recovered

    def update_run_live_state(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments: List[str] = []
        params: List[Any] = []
        for key, value in fields.items():
            if value is None:
                continue
            assignments.append(f"{key} = ?")
            params.append(value)
        if not assignments:
            return
        assignments.append("last_progress_at = ?")
        params.append(utcnow_iso())
        params.append(run_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?", tuple(params))
            conn.commit()

    def update_run_preflight(self, run_id: str, *, preflight_status: str, preflight_summary: str) -> None:
        self.update_run_live_state(
            run_id,
            preflight_status=preflight_status,
            preflight_summary=preflight_summary,
            status_message=preflight_summary,
        )

    def update_run_abort(
        self,
        run_id: str,
        *,
        abort_reason: str,
        failure_summary: str,
        systemic_failure_kind: str = "",
        systemic_failure_count: int = 0,
    ) -> None:
        self.update_run_live_state(
            run_id,
            abort_reason=abort_reason,
            failure_summary=failure_summary,
            last_error_summary=failure_summary,
            status_message=abort_reason,
            systemic_failure_kind=systemic_failure_kind,
            systemic_failure_count=systemic_failure_count,
        )

    def update_run_summary(self, run_id: str, summary: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET total_sources = ?,
                    total_derived_images = ?,
                    total_reactions = ?,
                    total_failures = ?,
                    total_redo = ?,
                    prompt_tokens = ?,
                    completion_tokens = ?,
                    total_tokens = ?,
                    estimated_cost_usd = ?,
                    usage_completeness = ?,
                    elapsed_seconds = ?
                WHERE run_id = ?
                """,
                (
                    int(summary.get("total_sources", 0)),
                    int(summary.get("total_derived_images", 0)),
                    int(summary.get("total_reactions", 0)),
                    int(summary.get("total_failures", 0)),
                    int(summary.get("total_redo", 0)),
                    int(summary.get("prompt_tokens", 0)),
                    int(summary.get("completion_tokens", 0)),
                    int(summary.get("total_tokens", 0)),
                    summary.get("estimated_cost_usd"),
                    str(summary.get("usage_completeness", "none")),
                    summary.get("elapsed_seconds"),
                    run_id,
                ),
            )
            conn.commit()

    def upsert_source_asset(
        self,
        *,
        source_type: str,
        original_filename: str,
        artifact_backend: str,
        artifact_key: str,
        artifact_status: str,
        sha256: Optional[str] = None,
        source_asset_id: str = "",
    ) -> str:
        with self.connect() as conn:
            if sha256:
                row = conn.execute(
                    "SELECT source_asset_id FROM source_assets WHERE sha256 = ?",
                    (sha256,),
                ).fetchone()
                if row:
                    return row["source_asset_id"]
            source_asset_id = source_asset_id or uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO source_assets(
                    source_asset_id, source_type, sha256, original_filename,
                    artifact_backend, artifact_key, artifact_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_asset_id,
                    source_type,
                    sha256,
                    original_filename,
                    artifact_backend,
                    artifact_key,
                    artifact_status,
                ),
            )
            conn.commit()
            return source_asset_id

    def get_source_asset(self, source_asset_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM source_assets WHERE source_asset_id = ?",
                (source_asset_id,),
            ).fetchone()
        return _dict_from_row(row) if row else None

    def create_run_source(
        self,
        *,
        run_id: str,
        source_asset_id: str,
        input_order: int,
        recovery_note: str = "",
        source_type: str = "",
    ) -> str:
        run_source_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO run_sources(run_source_id, run_id, source_asset_id, input_order, recovery_note, source_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_source_id, run_id, source_asset_id, input_order, recovery_note, source_type),
            )
            conn.commit()
        return run_source_id

    def update_run_source_status(self, run_source_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments: List[str] = []
        params: List[Any] = []
        if fields.get("started"):
            fields["started_at"] = utcnow_iso()
            del fields["started"]
        if fields.get("finished"):
            fields["finished_at"] = utcnow_iso()
            del fields["finished"]
        if "error_summary" in fields and fields["error_summary"]:
            fields.setdefault("last_error_at", utcnow_iso())
        for key, value in fields.items():
            if value is None:
                continue
            assignments.append(f"{key} = ?")
            params.append(value)
        if not assignments:
            return
        params.append(run_source_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE run_sources SET {', '.join(assignments)} WHERE run_source_id = ?", tuple(params))
            conn.commit()

    def update_run_source_progress(self, run_source_id: str, **fields: Any) -> None:
        self.update_run_source_status(run_source_id, **fields)

    def finalize_run_source_summary(self, run_source_id: str, summary: Dict[str, Any]) -> None:
        self.update_run_source_status(
            run_source_id,
            status=summary.get("status", "completed"),
            current_phase="finalize",
            status_message=summary.get("status_message", ""),
            expected_derived_images=int(summary.get("expected_derived_images", 0)),
            completed_derived_images=int(summary.get("completed_derived_images", 0)),
            successful_derived_images=int(summary.get("successful_derived_images", 0)),
            failed_derived_images=int(summary.get("failed_derived_images", 0)),
            reaction_count=int(summary.get("reaction_count", 0)),
            redo_count=int(summary.get("redo_count", 0)),
            prompt_tokens=int(summary.get("prompt_tokens", 0)),
            completion_tokens=int(summary.get("completion_tokens", 0)),
            total_tokens=int(summary.get("total_tokens", 0)),
            estimated_cost_usd=summary.get("estimated_cost_usd"),
            usage_completeness=str(summary.get("usage_completeness", "none")),
            error_summary=str(summary.get("error_summary") or ""),
            finished=True,
        )

    def create_derived_image(
        self,
        *,
        run_source_id: str,
        page_hint: str,
        image_index: int,
        artifact_backend: str,
        artifact_key: str,
        artifact_status: str,
        outcome_class: str,
        raw_artifact_key: str,
        error_text: str = "",
        status: str = "queued",
    ) -> str:
        derived_image_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO derived_images(
                    derived_image_id, run_source_id, page_hint, image_index,
                    artifact_backend, artifact_key, artifact_status,
                    status, outcome_class, raw_artifact_key, error_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    derived_image_id,
                    run_source_id,
                    page_hint,
                    image_index,
                    artifact_backend,
                    artifact_key,
                    artifact_status,
                    status,
                    outcome_class,
                    raw_artifact_key,
                    error_text,
                ),
            )
            conn.commit()
        return derived_image_id

    def update_derived_image_status(self, derived_image_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments: List[str] = []
        params: List[Any] = []
        if fields.get("started"):
            fields["started_at"] = utcnow_iso()
            del fields["started"]
        if fields.get("finished"):
            fields["finished_at"] = utcnow_iso()
            del fields["finished"]
        for key, value in fields.items():
            if value is None:
                continue
            assignments.append(f"{key} = ?")
            params.append(value)
        if not assignments:
            return
        params.append(derived_image_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE derived_images SET {', '.join(assignments)} WHERE derived_image_id = ?", tuple(params))
            conn.commit()

    def finalize_derived_image_summary(self, derived_image_id: str, summary: Dict[str, Any]) -> None:
        self.update_derived_image_status(
            derived_image_id,
            status=summary.get("status", "completed"),
            status_message=summary.get("status_message", ""),
            outcome_class=summary.get("outcome_class", ""),
            raw_artifact_key=summary.get("raw_artifact_key", ""),
            error_text=summary.get("error_text", ""),
            artifact_status=summary.get("artifact_status"),
            artifact_backend=summary.get("artifact_backend"),
            artifact_key=summary.get("artifact_key"),
            reaction_count=int(summary.get("reaction_count", 0)),
            accepted_reaction_count=int(summary.get("accepted_reaction_count", summary.get("reaction_count", 0))),
            rejected_reaction_count=int(summary.get("rejected_reaction_count", 0)),
            normalization_status=str(summary.get("normalization_status", "")),
            normalization_summary=str(summary.get("normalization_summary", "")),
            attempt_count=int(summary.get("attempt_count", 0)),
            last_attempt_id=str(summary.get("last_attempt_id", "")),
            last_retry_reason=str(summary.get("last_retry_reason", "")),
            redo_count=int(summary.get("redo_count", 0)),
            prompt_tokens=int(summary.get("prompt_tokens", 0)),
            completion_tokens=int(summary.get("completion_tokens", 0)),
            total_tokens=int(summary.get("total_tokens", 0)),
            estimated_cost_usd=summary.get("estimated_cost_usd"),
            finished=True,
        )

    def update_derived_image(
        self,
        derived_image_id: str,
        *,
        outcome_class: str,
        raw_artifact_key: str,
        error_text: str = "",
        artifact_status: Optional[str] = None,
        artifact_backend: Optional[str] = None,
        artifact_key: Optional[str] = None,
    ) -> None:
        self.update_derived_image_status(
            derived_image_id,
            outcome_class=outcome_class,
            raw_artifact_key=raw_artifact_key,
            error_text=error_text,
            artifact_status=artifact_status,
            artifact_backend=artifact_backend,
            artifact_key=artifact_key,
        )

    def create_reaction(
        self,
        *,
        run_id: str,
        run_source_id: str,
        derived_image_id: str,
        attempt_id: str,
        reaction_id: str,
        reaction_fingerprint: str,
        outcome_class: str,
        structure_quality: str,
        acceptance_reason: str,
        render_artifact_key: str,
        raw_reaction_json: str,
        review_status: str = "unchecked",
        review_notes: str = "",
    ) -> str:
        reaction_uid = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO reactions(
                    reaction_uid, run_id, run_source_id, derived_image_id, attempt_id, reaction_id,
                    reaction_fingerprint, outcome_class, structure_quality, acceptance_reason, review_status, review_notes,
                    render_artifact_key, raw_reaction_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reaction_uid,
                    run_id,
                    run_source_id,
                    derived_image_id,
                    attempt_id,
                    reaction_id,
                    reaction_fingerprint,
                    outcome_class,
                    structure_quality,
                    acceptance_reason,
                    review_status,
                    review_notes,
                    render_artifact_key,
                    raw_reaction_json,
                ),
            )
            conn.commit()
        return reaction_uid

    def add_reaction_molecules(self, reaction_uid: str, molecules: Iterable[Dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO reaction_molecules(molecule_id, reaction_uid, side, ordinal, smiles, label, valid_smiles, validation_kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        uuid.uuid4().hex,
                        reaction_uid,
                        str(item.get("side") or ""),
                        int(item.get("ordinal", 0)),
                        str(item.get("smiles") or ""),
                        str(item.get("label") or ""),
                        1 if item.get("valid_smiles") else 0,
                        str(item.get("validation_kind") or ""),
                    )
                    for item in molecules
                ],
            )
            conn.commit()

    def add_reaction_conditions(self, reaction_uid: str, conditions: Iterable[Dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO reaction_conditions(condition_id, reaction_uid, ordinal, role, text, smiles)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        uuid.uuid4().hex,
                        reaction_uid,
                        int(item.get("ordinal", 0)),
                        str(item.get("role") or ""),
                        str(item.get("text") or ""),
                        str(item.get("smiles") or ""),
                    )
                    for item in conditions
                ],
            )
            conn.commit()

    def add_reaction_additional_info(self, reaction_uid: str, items: Iterable[Dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO reaction_additional_info(info_id, reaction_uid, ordinal, kind, key, value, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        uuid.uuid4().hex,
                        reaction_uid,
                        int(item.get("ordinal", 0)),
                        str(item.get("kind") or ""),
                        str(item.get("key") or ""),
                        str(item.get("value") or ""),
                        str(item.get("raw_text") or ""),
                    )
                    for item in items
                ],
            )
            conn.commit()

    def purge_canonical_reactions_for_derived_image(self, derived_image_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM reactions WHERE derived_image_id = ?", (derived_image_id,))
            conn.commit()

    def create_derived_image_attempt(
        self,
        *,
        derived_image_id: str,
        trigger: str,
        execution_mode: str,
        status: str = "queued",
        config_snapshot_json: str = "",
        retry_of_attempt_id: str = "",
    ) -> Dict[str, Any]:
        attempt_id = uuid.uuid4().hex
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) AS max_attempt_no FROM derived_image_attempts WHERE derived_image_id = ?",
                (derived_image_id,),
            ).fetchone()
            attempt_no = int(row["max_attempt_no"] or 0) + 1
            conn.execute(
                """
                INSERT INTO derived_image_attempts(
                    attempt_id, derived_image_id, attempt_no, trigger, execution_mode, status,
                    config_snapshot_json, retry_of_attempt_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    derived_image_id,
                    attempt_no,
                    trigger,
                    execution_mode,
                    status,
                    config_snapshot_json,
                    retry_of_attempt_id,
                ),
            )
            conn.execute(
                """
                UPDATE derived_images
                SET attempt_count = ?, last_attempt_id = ?, last_retry_reason = ?
                WHERE derived_image_id = ?
                """,
                (attempt_no, attempt_id, trigger, derived_image_id),
            )
            conn.commit()
        return {
            "attempt_id": attempt_id,
            "attempt_no": attempt_no,
            "derived_image_id": derived_image_id,
            "trigger": trigger,
            "execution_mode": execution_mode,
            "status": status,
        }

    def update_derived_image_attempt(self, attempt_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments: List[str] = []
        params: List[Any] = []
        if fields.get("started"):
            fields["started_at"] = utcnow_iso()
            del fields["started"]
        if fields.get("finished"):
            fields["finished_at"] = utcnow_iso()
            del fields["finished"]
        for key, value in fields.items():
            if value is None:
                continue
            assignments.append(f"{key} = ?")
            params.append(value)
        if not assignments:
            return
        params.append(attempt_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE derived_image_attempts SET {', '.join(assignments)} WHERE attempt_id = ?", tuple(params))
            conn.commit()

    def finalize_derived_image_attempt(self, attempt_id: str, summary: Dict[str, Any]) -> None:
        self.update_derived_image_attempt(
            attempt_id,
            status=summary.get("status", "completed"),
            failure_kind=str(summary.get("failure_kind", "")),
            error_summary=str(summary.get("error_summary", "")),
            raw_artifact_key=str(summary.get("raw_artifact_key", "")),
            prompt_tokens=int(summary.get("prompt_tokens", 0)),
            completion_tokens=int(summary.get("completion_tokens", 0)),
            total_tokens=int(summary.get("total_tokens", 0)),
            estimated_cost_usd=summary.get("estimated_cost_usd"),
            usage_completeness=str(summary.get("usage_completeness", "none")),
            finished=True,
        )

    def list_derived_image_attempts(self, derived_image_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM derived_image_attempts
                WHERE derived_image_id = ?
                ORDER BY attempt_no
                """,
                (derived_image_id,),
            ).fetchall()
        return [_dict_from_row(row) for row in rows]

    def add_llm_call_metrics(
        self,
        *,
        run_id: str,
        call_metrics: Iterable[Dict[str, Any]],
        run_source_id: str = "",
        derived_image_id: str = "",
    ) -> None:
        records = list(call_metrics)
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO llm_call_metrics(
                    llm_call_id, run_id, run_source_id, derived_image_id, phase, provider, model,
                    usage_prompt_tokens, usage_completion_tokens, usage_total_tokens,
                    estimated_cost_usd, latency_ms, success, raw_usage_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        uuid.uuid4().hex,
                        run_id,
                        run_source_id,
                        derived_image_id,
                        str(item.get("phase") or ""),
                        str(item.get("provider") or ""),
                        str(item.get("model") or ""),
                        item.get("usage_prompt_tokens"),
                        item.get("usage_completion_tokens"),
                        item.get("usage_total_tokens"),
                        item.get("estimated_cost_usd"),
                        int(item.get("latency_ms") or 0),
                        1 if item.get("success", True) else 0,
                        str(item.get("raw_usage_json") or ""),
                    )
                    for item in records
                ],
            )
            conn.commit()

    def list_experiments(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*,
                       COUNT(r.run_id) AS run_count
                FROM experiments e
                LEFT JOIN runs r ON r.experiment_id = e.experiment_id
                GROUP BY e.experiment_id
                ORDER BY e.created_at DESC
                """
            ).fetchall()
        return [_dict_from_row(row) for row in rows]

    def list_runs(self, experiment_id: str = "") -> List[Dict[str, Any]]:
        query = """
            SELECT runs.*, experiments.name AS experiment_name, experiments.status AS experiment_status
            FROM runs
            JOIN experiments ON experiments.experiment_id = runs.experiment_id
        """
        params: tuple[Any, ...] = ()
        if experiment_id:
            query += " WHERE runs.experiment_id = ?"
            params = (experiment_id,)
        query += " ORDER BY runs.created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_dict_from_row(row) for row in rows]

    def list_run_sources(self, run_id: str) -> List[Dict[str, Any]]:
        query = """
            SELECT rs.*, sa.original_filename, sa.artifact_backend AS source_artifact_backend,
                   sa.artifact_key AS source_artifact_key, sa.artifact_status AS source_artifact_status
            FROM run_sources rs
            JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
            WHERE rs.run_id = ?
            ORDER BY rs.input_order, sa.original_filename
        """
        with self.connect() as conn:
            rows = conn.execute(query, (run_id,)).fetchall()
        return [_dict_from_row(row) for row in rows]

    def get_run_source_detail(self, run_source_id: str) -> Dict[str, Any]:
        with self.connect() as conn:
            source = conn.execute(
                """
                SELECT rs.*, sa.original_filename, sa.artifact_backend AS source_artifact_backend,
                       sa.artifact_key AS source_artifact_key, sa.artifact_status AS source_artifact_status
                FROM run_sources rs
                JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
                WHERE rs.run_source_id = ?
                """,
                (run_source_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"Unknown run_source_id: {run_source_id}")
            derived_images = conn.execute(
                """
                SELECT *
                FROM derived_images
                WHERE run_source_id = ?
                ORDER BY image_index
                """,
                (run_source_id,),
            ).fetchall()
        payload = _dict_from_row(source)
        payload["derived_images"] = []
        for row in derived_images:
            record = _dict_from_row(row)
            record["attempts"] = self.list_derived_image_attempts(str(record["derived_image_id"]))
            payload["derived_images"].append(record)
        return payload

    def get_derived_image(self, derived_image_id: str) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT di.*, rs.run_id, rs.run_source_id, rs.source_asset_id, sa.original_filename
                FROM derived_images di
                JOIN run_sources rs ON rs.run_source_id = di.run_source_id
                JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
                WHERE di.derived_image_id = ?
                """,
                (derived_image_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown derived_image_id: {derived_image_id}")
        payload = _dict_from_row(row)
        payload["attempts"] = self.list_derived_image_attempts(derived_image_id)
        return payload

    def list_retry_candidates(self, run_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT di.*, rs.run_id, rs.run_source_id, sa.original_filename
                FROM derived_images di
                JOIN run_sources rs ON rs.run_source_id = di.run_source_id
                JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
                WHERE rs.run_id = ?
                  AND (
                        di.status = 'failed'
                        OR di.outcome_class = 'needs_redo'
                        OR di.accepted_reaction_count = 0
                        OR di.normalization_status IN ('redo_pending', 'rejected_invalid_smiles', 'rejected_missing_smiles')
                  )
                ORDER BY sa.original_filename, di.image_index
                """,
                (run_id,),
            ).fetchall()
        return [_dict_from_row(row) for row in rows]

    def list_reactions(
        self,
        *,
        experiment_id: str = "",
        run_id: str = "",
        review_status: str = "",
        outcome_class: str = "",
    ) -> List[Dict[str, Any]]:
        filters: List[str] = []
        params: List[Any] = []
        if experiment_id:
            filters.append("r.run_id IN (SELECT run_id FROM runs WHERE experiment_id = ?)")
            params.append(experiment_id)
        if run_id:
            filters.append("r.run_id = ?")
            params.append(run_id)
        if review_status:
            filters.append("r.review_status = ?")
            params.append(review_status)
        if outcome_class:
            filters.append("r.outcome_class = ?")
            params.append(outcome_class)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"""
            SELECT r.reaction_uid, r.run_id, r.reaction_id, r.reaction_fingerprint, r.outcome_class,
                   r.structure_quality, r.acceptance_reason,
                   r.review_status, r.review_notes, r.render_artifact_key,
                   runs.profile_label, runs.main_provider, runs.main_model, runs.estimated_cost_usd,
                   sa.original_filename, di.page_hint, di.artifact_backend AS derived_backend,
                   di.artifact_key AS derived_artifact_key, di.artifact_status AS derived_artifact_status,
                   sa.source_asset_id
            FROM reactions r
            JOIN runs ON runs.run_id = r.run_id
            JOIN run_sources rs ON rs.run_source_id = r.run_source_id
            JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
            JOIN derived_images di ON di.derived_image_id = r.derived_image_id
            {where_clause}
            ORDER BY runs.created_at DESC, sa.original_filename, di.image_index, r.reaction_id
        """
        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [_dict_from_row(row) for row in rows]

    def get_reaction_detail(self, reaction_uid: str) -> Dict[str, Any]:
        with self.connect() as conn:
            reaction = conn.execute(
                """
                SELECT r.*,
                       runs.profile_label, runs.main_provider, runs.main_model, runs.estimated_cost_usd,
                       sa.original_filename, sa.artifact_backend AS source_artifact_backend,
                       sa.artifact_key AS source_artifact_key, sa.artifact_status AS source_artifact_status,
                       di.page_hint, di.image_index, di.artifact_backend AS derived_backend,
                       di.artifact_key AS derived_artifact_key, di.artifact_status AS derived_artifact_status
                FROM reactions r
                JOIN runs ON runs.run_id = r.run_id
                JOIN run_sources rs ON rs.run_source_id = r.run_source_id
                JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
                JOIN derived_images di ON di.derived_image_id = r.derived_image_id
                WHERE r.reaction_uid = ?
                """,
                (reaction_uid,),
            ).fetchone()
            if reaction is None:
                raise KeyError(f"Unknown reaction_uid: {reaction_uid}")
            molecules = conn.execute(
                "SELECT * FROM reaction_molecules WHERE reaction_uid = ? ORDER BY side, ordinal",
                (reaction_uid,),
            ).fetchall()
            conditions = conn.execute(
                "SELECT * FROM reaction_conditions WHERE reaction_uid = ? ORDER BY ordinal",
                (reaction_uid,),
            ).fetchall()
            infos = conn.execute(
                "SELECT * FROM reaction_additional_info WHERE reaction_uid = ? ORDER BY ordinal",
                (reaction_uid,),
            ).fetchall()
        payload = _dict_from_row(reaction)
        payload["molecules"] = [_dict_from_row(row) for row in molecules]
        payload["conditions"] = [_dict_from_row(row) for row in conditions]
        payload["additional_info"] = [_dict_from_row(row) for row in infos]
        return payload

    def update_reaction_review(self, reaction_uid: str, *, review_status: str, review_notes: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE reactions SET review_status = ?, review_notes = ? WHERE reaction_uid = ?",
                (review_status, review_notes, reaction_uid),
            )
            conn.commit()

    def export_run_to_parquet(self, run_id: str, output_dir: Path) -> Dict[str, str]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "reactions": output_dir / f"{run_id}_reactions.parquet",
            "reaction_molecules": output_dir / f"{run_id}_reaction_molecules.parquet",
            "reaction_conditions": output_dir / f"{run_id}_reaction_conditions.parquet",
            "reaction_additional_info": output_dir / f"{run_id}_reaction_additional_info.parquet",
        }
        with self.connect() as conn:
            pd.read_sql_query("SELECT * FROM reactions WHERE run_id = ?", conn, params=(run_id,)).to_parquet(files["reactions"], index=False)
            pd.read_sql_query(
                """
                SELECT rm.*
                FROM reaction_molecules rm
                JOIN reactions r ON r.reaction_uid = rm.reaction_uid
                WHERE r.run_id = ?
                """,
                conn,
                params=(run_id,),
            ).to_parquet(files["reaction_molecules"], index=False)
            pd.read_sql_query(
                """
                SELECT rc.*
                FROM reaction_conditions rc
                JOIN reactions r ON r.reaction_uid = rc.reaction_uid
                WHERE r.run_id = ?
                """,
                conn,
                params=(run_id,),
            ).to_parquet(files["reaction_conditions"], index=False)
            pd.read_sql_query(
                """
                SELECT rai.*
                FROM reaction_additional_info rai
                JOIN reactions r ON r.reaction_uid = rai.reaction_uid
                WHERE r.run_id = ?
                """,
                conn,
                params=(run_id,),
            ).to_parquet(files["reaction_additional_info"], index=False)
        return {name: str(path) for name, path in files.items()}
