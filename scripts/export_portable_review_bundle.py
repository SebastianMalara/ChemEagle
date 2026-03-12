#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ARTIFACT_SUBDIRS: Sequence[str] = ("sources", "derived", "renders")
SENSITIVE_CONFIG_KEYS = {
    "API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VLLM_API_KEY",
    "OCR_API_KEY",
    "OCR_OPENAI_API_KEY",
    "OCR_ANTHROPIC_API_KEY",
    "OCR_VLLM_API_KEY",
    "ARTIFACT_S3_ACCESS_KEY_ID",
    "ARTIFACT_S3_SECRET_ACCESS_KEY",
}
RELATIVE_PATH_OVERRIDES = {
    "ARTIFACT_FILESYSTEM_ROOT": "./data/artifacts",
    "REVIEW_DB_PATH": "./data/review_dataset.sqlite3",
    "CHEMEAGLE_ASSET_ROOT": "./data/artifacts",
    "PDF_PERSIST_DIR": "./data/artifacts/derived",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_loads(text: str) -> Any:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"unparsed_text": text}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def _rows_as_dicts(cursor: sqlite3.Cursor) -> List[Dict[str, Any]]:
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _artifact_href(artifact_key: str) -> str:
    key = (artifact_key or "").strip().replace("\\", "/")
    return f"./data/artifacts/{key}" if key else ""


def _sanitize_config_snapshot(snapshot_text: str) -> str:
    payload = _json_loads(snapshot_text)
    if not isinstance(payload, dict):
        return _json_dumps({})

    for key in SENSITIVE_CONFIG_KEYS:
        if key in payload:
            payload[key] = ""
    for key, value in RELATIVE_PATH_OVERRIDES.items():
        if key in payload:
            payload[key] = value
    if "ARTIFACT_BACKEND" in payload:
        payload["ARTIFACT_BACKEND"] = "filesystem"
    return _json_dumps(payload)


def _copy_artifacts(source_root: Path, output_root: Path) -> None:
    for subdir in ARTIFACT_SUBDIRS:
        source_dir = source_root / subdir
        if not source_dir.exists():
            continue
        shutil.copytree(source_dir, output_root / subdir)


def _copy_viewer_assets(repo_root: Path, output_root: Path) -> None:
    viewer_asset_root = repo_root / "assets" / "review_bundle_viewer"
    viewer_output_root = output_root / "viewer"
    viewer_output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(viewer_asset_root / "index.html", output_root / "index.html")
    shutil.copy2(viewer_asset_root / "app.css", viewer_output_root / "app.css")
    shutil.copy2(viewer_asset_root / "app.js", viewer_output_root / "app.js")


def _write_readme(output_root: Path, *, title: str, reaction_count: int, run_count: int) -> None:
    readme = f"""\
{title}

Open `index.html` in a browser to browse the reactions without installing anything.

Bundle contents:
- `index.html`: static viewer entrypoint
- `viewer/`: CSS, JavaScript, and generated dataset payload
- `data/review_dataset.sqlite3`: sanitized SQLite copy
- `data/artifacts/sources`: source PDFs
- `data/artifacts/derived`: extracted crop images
- `data/artifacts/renders`: RDKit render images

Notes:
- Runtime logs and raw retry payload artifacts are intentionally omitted.
- API keys and machine-local paths are stripped from database config snapshots.
- This bundle includes {reaction_count} accepted reactions across {run_count} run(s).
"""
    (output_root / "README.txt").write_text(readme, encoding="utf-8")


def _sanitize_database(source_db: Path, output_db: Path) -> None:
    shutil.copy2(source_db, output_db)
    with sqlite3.connect(output_db) as conn:
        run_rows = conn.execute("SELECT run_id, config_snapshot_json FROM runs").fetchall()
        for run_id, snapshot in run_rows:
            conn.execute(
                """
                UPDATE runs
                SET config_snapshot_json = ?,
                    log_artifact_key = '',
                    stdout_artifact_key = '',
                    stderr_artifact_key = ''
                WHERE run_id = ?
                """,
                (_sanitize_config_snapshot(snapshot), run_id),
            )

        attempt_rows = conn.execute("SELECT attempt_id, config_snapshot_json FROM derived_image_attempts").fetchall()
        for attempt_id, snapshot in attempt_rows:
            conn.execute(
                """
                UPDATE derived_image_attempts
                SET config_snapshot_json = ?,
                    raw_artifact_key = ''
                WHERE attempt_id = ?
                """,
                (_sanitize_config_snapshot(snapshot), attempt_id),
            )

        conn.execute("UPDATE derived_images SET raw_artifact_key = ''")
        conn.commit()


def _group_rows(rows: Iterable[Dict[str, Any]], key_name: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key_name, "")), []).append(row)
    return grouped


def _load_bundle_dataset(db_path: Path, *, title: str) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        run_rows = _rows_as_dicts(
            conn.execute(
                """
                SELECT runs.run_id,
                       runs.experiment_id,
                       experiments.name AS experiment_name,
                       runs.profile_label,
                       runs.ingest_mode,
                       runs.status,
                       runs.created_at,
                       runs.started_at,
                       runs.finished_at,
                       runs.total_sources,
                       runs.total_derived_images,
                       runs.total_reactions,
                       runs.total_failures,
                       runs.total_redo,
                       runs.estimated_cost_usd
                FROM runs
                JOIN experiments ON experiments.experiment_id = runs.experiment_id
                ORDER BY runs.created_at DESC
                """
            )
        )
        reaction_rows = _rows_as_dicts(
            conn.execute(
                """
                SELECT r.reaction_uid,
                       r.run_id,
                       r.run_source_id,
                       r.derived_image_id,
                       r.reaction_id,
                       r.reaction_fingerprint,
                       r.outcome_class,
                       r.structure_quality,
                       r.acceptance_reason,
                       r.review_status,
                       r.review_notes,
                       r.render_artifact_key,
                       r.raw_reaction_json,
                       runs.profile_label,
                       runs.status AS run_status,
                       runs.created_at AS run_created_at,
                       runs.estimated_cost_usd,
                       sa.source_asset_id,
                       sa.original_filename,
                       sa.artifact_key AS source_artifact_key,
                       di.page_hint,
                       di.image_index,
                       di.artifact_key AS derived_artifact_key,
                       di.status AS derived_status,
                       di.artifact_status AS derived_artifact_status
                FROM reactions r
                JOIN runs ON runs.run_id = r.run_id
                JOIN run_sources rs ON rs.run_source_id = r.run_source_id
                JOIN source_assets sa ON sa.source_asset_id = rs.source_asset_id
                JOIN derived_images di ON di.derived_image_id = r.derived_image_id
                ORDER BY runs.created_at DESC, sa.original_filename, di.image_index, r.reaction_id
                """
            )
        )
        molecule_rows = _rows_as_dicts(
            conn.execute(
                """
                SELECT reaction_uid,
                       side,
                       ordinal,
                       smiles,
                       label,
                       valid_smiles,
                       validation_kind
                FROM reaction_molecules
                ORDER BY reaction_uid, side, ordinal
                """
            )
        )
        condition_rows = _rows_as_dicts(
            conn.execute(
                """
                SELECT reaction_uid,
                       ordinal,
                       role,
                       text,
                       smiles
                FROM reaction_conditions
                ORDER BY reaction_uid, ordinal
                """
            )
        )
        info_rows = _rows_as_dicts(
            conn.execute(
                """
                SELECT reaction_uid,
                       ordinal,
                       kind,
                       key,
                       value,
                       raw_text
                FROM reaction_additional_info
                ORDER BY reaction_uid, ordinal
                """
            )
        )

    molecules_by_reaction = _group_rows(molecule_rows, "reaction_uid")
    conditions_by_reaction = _group_rows(condition_rows, "reaction_uid")
    info_by_reaction = _group_rows(info_rows, "reaction_uid")

    reactions: List[Dict[str, Any]] = []
    for row in reaction_rows:
        reaction_uid = str(row["reaction_uid"])
        raw_reaction = _json_loads(str(row.get("raw_reaction_json") or ""))
        conditions = conditions_by_reaction.get(reaction_uid, [])
        additional_info = info_by_reaction.get(reaction_uid, [])
        reactions.append(
            {
                "reaction_uid": reaction_uid,
                "run_id": row["run_id"],
                "run_status": row["run_status"],
                "run_created_at": row["run_created_at"],
                "profile_label": row["profile_label"],
                "estimated_cost_usd": row["estimated_cost_usd"],
                "source_asset_id": row["source_asset_id"],
                "source_filename": row["original_filename"],
                "original_filename": row["original_filename"],
                "source_href": _artifact_href(str(row.get("source_artifact_key") or "")),
                "source_file": _artifact_href(str(row.get("source_artifact_key") or "")),
                "page_hint": row["page_hint"],
                "image_index": row["image_index"],
                "reaction_id": row["reaction_id"],
                "reaction_fingerprint": row["reaction_fingerprint"],
                "outcome_class": row["outcome_class"],
                "structure_quality": row["structure_quality"],
                "acceptance_reason": row["acceptance_reason"],
                "review_status": row["review_status"],
                "review_notes": row["review_notes"],
                "crop_href": _artifact_href(str(row.get("derived_artifact_key") or "")),
                "crop_file": _artifact_href(str(row.get("derived_artifact_key") or "")),
                "crop_status": row["derived_artifact_status"],
                "render_href": _artifact_href(str(row.get("render_artifact_key") or "")),
                "render_file": _artifact_href(str(row.get("render_artifact_key") or "")),
                "molecules": molecules_by_reaction.get(reaction_uid, []),
                "conditions": [
                    {
                        **item,
                        "condition_type": item.get("role", ""),
                        "value_text": item.get("text", "") or item.get("smiles", ""),
                    }
                    for item in conditions
                ],
                "additional_info": [
                    {
                        **item,
                        "info_type": item.get("kind", "") or item.get("key", ""),
                        "value_text": item.get("value", "") or item.get("raw_text", ""),
                    }
                    for item in additional_info
                ],
                "raw_reaction": raw_reaction,
                "raw_reaction_json": raw_reaction,
            }
        )

    source_names = sorted({str(row["source_filename"]) for row in reactions if row.get("source_filename")})
    run_ids = [str(row["run_id"]) for row in run_rows]
    return {
        "meta": {
            "title": title,
            "generated_at": _utc_now_iso(),
            "run_count": len(run_rows),
            "reaction_count": len(reactions),
            "source_count": len(source_names),
        },
        "runs": run_rows,
        "filters": {
            "run_ids": run_ids,
            "source_filenames": source_names,
        },
        "reactions": reactions,
    }


def build_bundle(
    *,
    db_path: Path,
    artifact_root: Path,
    output_dir: Path,
    title: str,
    make_zip: bool,
) -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    output_data_dir = output_dir / "data"
    output_artifact_dir = output_data_dir / "artifacts"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_dir.mkdir(parents=True, exist_ok=True)

    sanitized_db_path = output_data_dir / "review_dataset.sqlite3"
    _sanitize_database(db_path, sanitized_db_path)
    _copy_artifacts(artifact_root, output_artifact_dir)
    _copy_viewer_assets(repo_root, output_dir)

    dataset = _load_bundle_dataset(sanitized_db_path, title=title)
    (output_dir / "viewer" / "dataset.js").write_text(
        "window.CHEMEAGLE_DATA = " + _json_dumps(dataset) + ";\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir,
        title=title,
        reaction_count=int(dataset["meta"]["reaction_count"]),
        run_count=int(dataset["meta"]["run_count"]),
    )

    result = {"bundle_dir": str(output_dir), "database": str(sanitized_db_path)}
    if make_zip:
        archive_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
        result["zip_path"] = archive_path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable static review bundle for CNR delivery.")
    parser.add_argument("--db-path", default="data/review_dataset.sqlite3", help="Path to the source SQLite review dataset.")
    parser.add_argument("--artifact-root", default="data/artifacts", help="Path to the source artifact directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the bundle should be created.")
    parser.add_argument("--title", default="ChemEagle Reaction Review Bundle", help="Title shown in the viewer.")
    parser.add_argument("--zip", action="store_true", help="Also create a zip archive next to the output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_bundle(
        db_path=Path(args.db_path).expanduser().resolve(),
        artifact_root=Path(args.artifact_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        title=args.title,
        make_zip=bool(args.zip),
    )
    print(_json_dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
