from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from review_db import ReviewRepository


class ReviewRepositoryTests(unittest.TestCase):
    def test_create_and_query_reaction_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ReviewRepository(Path(tmpdir) / "review.sqlite3")
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="queued",
                config_snapshot={"LLM_PROVIDER": "azure", "LLM_MODEL": "gpt-5-mini"},
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="image",
                sha256="abc123",
                original_filename="source.png",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0)
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="source.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
                outcome_class="succeeded",
                raw_artifact_key="raw/run/source.json",
            )
            reaction_uid = repo.create_reaction(
                run_id=run_id,
                run_source_id=run_source_id,
                derived_image_id=derived_image_id,
                attempt_id="attempt-1",
                reaction_id="0_1",
                reaction_fingerprint="fingerprint",
                outcome_class="succeeded",
                structure_quality="rdkit_valid",
                acceptance_reason="accepted_structure_gate",
                render_artifact_key="renders/r1.png",
                raw_reaction_json='{"reaction_id":"0_1"}',
            )
            repo.add_reaction_molecules(
                reaction_uid,
                [{"side": "reactant", "ordinal": 0, "smiles": "CCO", "label": "A", "valid_smiles": True, "validation_kind": "rdkit_valid"}],
            )
            repo.add_reaction_conditions(
                reaction_uid,
                [{"ordinal": 0, "role": "temperature", "text": "rt", "smiles": ""}],
            )
            repo.add_reaction_additional_info(
                reaction_uid,
                [{"ordinal": 0, "kind": "text", "key": "", "value": "", "raw_text": "Yield: 80%"}],
            )
            repo.update_reaction_review(reaction_uid, review_status="ok", review_notes="looks good")

            rows = repo.list_reactions(run_id=run_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reaction_uid"], reaction_uid)
            self.assertEqual(rows[0]["structure_quality"], "rdkit_valid")

            detail = repo.get_reaction_detail(reaction_uid)
            self.assertEqual(detail["review_status"], "ok")
            self.assertEqual(detail["review_notes"], "looks good")
            self.assertEqual(len(detail["molecules"]), 1)
            self.assertEqual(detail["molecules"][0]["validation_kind"], "rdkit_valid")
            self.assertEqual(len(detail["conditions"]), 1)
            self.assertEqual(len(detail["additional_info"]), 1)

    def test_run_source_and_derived_monitor_fields_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ReviewRepository(Path(tmpdir) / "review.sqlite3")
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="queued",
                config_snapshot={"LLM_PROVIDER": "azure", "LLM_MODEL": "gpt-5-mini"},
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="pdf",
                sha256="abc123",
                original_filename="paper.pdf",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.pdf",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(
                run_id=run_id,
                source_asset_id=source_asset_id,
                input_order=0,
                source_type="pdf",
            )
            repo.update_run_source_status(
                run_source_id,
                status="processing",
                current_phase="process_image",
                expected_derived_images=3,
                completed_derived_images=1,
                failed_derived_images=1,
                reaction_count=2,
                error_summary="first crop failed",
                started=True,
            )
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="paper_image_2_1.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key="derived/run-source/0.png",
                artifact_status="present",
                status="processing",
                outcome_class="queued",
                raw_artifact_key="",
            )
            repo.finalize_derived_image_summary(
                derived_image_id,
                {
                    "status": "failed",
                    "status_message": "crop failed",
                    "outcome_class": "failed",
                    "raw_artifact_key": "raw/run/source/image_0.json",
                    "error_text": "Connection error.",
                    "reaction_count": 0,
                    "redo_count": 0,
                },
            )
            repo.update_run_live_state(
                run_id,
                current_phase="process_image",
                current_run_source_id=run_source_id,
                current_source_name="paper.pdf",
                status_message="Processing crop 1",
                completed_sources=0,
                failed_sources=0,
            )

            run_row = repo.get_run(run_id)
            self.assertEqual(run_row["current_run_source_id"], run_source_id)
            self.assertEqual(run_row["current_source_name"], "paper.pdf")
            self.assertEqual(run_row["current_phase"], "process_image")

            sources = repo.list_run_sources(run_id)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["expected_derived_images"], 3)
            self.assertEqual(sources[0]["completed_derived_images"], 1)
            self.assertEqual(sources[0]["error_summary"], "first crop failed")

            detail = repo.get_run_source_detail(run_source_id)
            self.assertEqual(detail["original_filename"], "paper.pdf")
            self.assertEqual(len(detail["derived_images"]), 1)
            self.assertEqual(detail["derived_images"][0]["status"], "failed")
            self.assertIn("Connection error.", detail["derived_images"][0]["error_text"])

    def test_mark_running_runs_interrupted_updates_live_entities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ReviewRepository(Path(tmpdir) / "review.sqlite3")
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="running",
                config_snapshot={"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5-mini"},
                config_hash="hash",
            )
            source_a = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="pdf",
                sha256="sha-a",
                original_filename="paper-a.pdf",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.pdf",
                artifact_status="present",
            )
            source_b = repo.upsert_source_asset(
                source_asset_id="source-b",
                source_type="pdf",
                sha256="sha-b",
                original_filename="paper-b.pdf",
                artifact_backend="filesystem",
                artifact_key="sources/source-b/original.pdf",
                artifact_status="present",
            )
            run_source_a = repo.create_run_source(run_id=run_id, source_asset_id=source_a, input_order=0, source_type="pdf")
            run_source_b = repo.create_run_source(run_id=run_id, source_asset_id=source_b, input_order=1, source_type="pdf")
            repo.update_run_live_state(
                run_id,
                current_run_source_id=run_source_a,
                current_source_name="paper-a.pdf",
                current_phase="process_image",
                status_message="Processing crop 1",
            )
            repo.update_run_source_status(run_source_a, status="processing", current_phase="process_image", started=True)
            repo.update_run_source_status(run_source_b, status="queued", current_phase="prepare_source")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_a,
                page_hint="paper-a_image_2_1.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key="derived/source-a/0.png",
                artifact_status="present",
                status="processing",
                outcome_class="queued",
                raw_artifact_key="",
            )

            recovered = repo.mark_running_runs_interrupted(reason="Process ended.")

            self.assertEqual(recovered, [{"run_id": run_id, "experiment_id": experiment_id}])
            run_row = repo.get_run(run_id)
            self.assertEqual(run_row["status"], "interrupted")
            self.assertEqual(run_row["current_phase"], "interrupted")
            self.assertEqual(run_row["abort_reason"], "Process ended.")

            sources = repo.list_run_sources(run_id)
            by_id = {row["run_source_id"]: row for row in sources}
            self.assertEqual(by_id[run_source_a]["status"], "aborted")
            self.assertEqual(by_id[run_source_b]["status"], "skipped")

            detail = repo.get_run_source_detail(run_source_a)
            self.assertEqual(len(detail["derived_images"]), 1)
            self.assertEqual(detail["derived_images"][0]["derived_image_id"], derived_image_id)
            self.assertEqual(detail["derived_images"][0]["status"], "failed")
            self.assertIn("Process ended.", detail["derived_images"][0]["error_text"])

    def test_derived_image_attempts_and_retry_candidates_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ReviewRepository(Path(tmpdir) / "review.sqlite3")
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="completed",
                config_snapshot={"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-5-mini"},
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="image",
                sha256="abc123",
                original_filename="scheme.png",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0)
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="scheme.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key="derived/run-source/0.png",
                artifact_status="present",
                status="completed",
                outcome_class="needs_redo",
                raw_artifact_key="raw/run/source/image_0.json",
            )
            attempt = repo.create_derived_image_attempt(
                derived_image_id=derived_image_id,
                trigger="initial",
                execution_mode="normal",
                status="running",
                config_snapshot_json="{}",
            )
            repo.finalize_derived_image_attempt(
                attempt["attempt_id"],
                {"status": "completed", "raw_artifact_key": "raw/run/source/image_0.json"},
            )
            repo.finalize_derived_image_summary(
                derived_image_id,
                {
                    "status": "completed",
                    "outcome_class": "needs_redo",
                    "raw_artifact_key": "raw/run/source/image_0.json",
                    "reaction_count": 0,
                    "accepted_reaction_count": 0,
                    "rejected_reaction_count": 2,
                    "normalization_status": "redo_pending",
                    "normalization_summary": "candidates=2 | accepted=0 | rejected=2",
                    "attempt_count": 1,
                    "last_attempt_id": attempt["attempt_id"],
                    "last_retry_reason": "initial",
                },
            )

            detail = repo.get_run_source_detail(run_source_id)
            self.assertEqual(detail["derived_images"][0]["attempts"][0]["attempt_id"], attempt["attempt_id"])

            candidates = repo.list_retry_candidates(run_id)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["derived_image_id"], derived_image_id)


if __name__ == "__main__":
    unittest.main()
