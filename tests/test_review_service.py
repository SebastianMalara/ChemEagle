from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import review_service as review_service_module
from review_db import ReviewRepository
from review_service import ReviewDatasetService, _apply_runtime_env, _rewrite_payload_image_refs
from review_tracking import RunMetricsCollector
from review_artifacts import create_artifact_store_from_config


class ReviewServiceTests(unittest.TestCase):
    def test_apply_runtime_env_removes_empty_values(self) -> None:
        with mock.patch.dict("os.environ", {"OPENAI_BASE_URL": "https://stale.example/v1"}, clear=False):
            _apply_runtime_env(
                {
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_KEY": "sk-test",
                    "CHEMEAGLE_RUN_MODE": "cloud",
                }
            )
            self.assertNotIn("OPENAI_BASE_URL", os.environ)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-test")

    def test_rewrite_payload_image_refs_uses_persisted_artifact_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_root = tmp / "artifacts"
            image_path = tmp / "source.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            store = create_artifact_store_from_config(
                {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
            )
            artifact_key = "derived/run-source/0.png"
            store.put_file(artifact_key, str(image_path), content_type="image/png")
            payload = {
                "plan": [{"arguments": {"image_path": "/tmp/worker/source.png"}}],
                "execution_logs": [{"arguments": {"image_path": "/tmp/worker/source.png"}}],
            }
            rewritten = _rewrite_payload_image_refs(
                payload,
                store=store,
                artifact_key=artifact_key,
                artifact_backend="filesystem",
            )
            expected_ref = str((artifact_root / artifact_key).resolve())
            self.assertEqual(rewritten["plan"][0]["arguments"]["image_path"], expected_ref)
            self.assertEqual(rewritten["plan"][0]["arguments"]["original_image_path"], "/tmp/worker/source.png")
            self.assertEqual(rewritten["plan"][0]["arguments"]["image_artifact_key"], artifact_key)

    def test_sideload_import_recovers_image_and_persists_reaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "scheme_1.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            json_path = tmp / "run.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "image_name": "scheme_1.png",
                            "reactions": [
                                {
                                    "reaction_id": "0_1",
                                    "reactants": [{"smiles": "CCO", "label": "A"}],
                                    "conditions": [{"role": "temperature", "text": "rt"}],
                                    "products": [{"smiles": "CC=O", "label": "B"}],
                                    "additional_info": [{"text": "Yield: 80%"}],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            service = ReviewDatasetService(ReviewRepository(db_path))
            result = service.submit_sideload_experiment(
                experiment_name="sideload",
                notes="",
                json_paths=[str(json_path)],
                recovery_roots=[str(tmp)],
                config_snapshot={
                    "ARTIFACT_BACKEND": "filesystem",
                    "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                },
            )
            service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")

            reactions = service.list_reactions(run_id=runs[0]["run_id"])
            self.assertEqual(len(reactions), 1)
            detail = service.get_reaction_detail(reactions[0]["reaction_uid"])
            self.assertEqual(detail["reaction_id"], "0_1")
            self.assertEqual(detail["review_status"], "unchecked")

    def test_sideload_import_persists_nested_reactions_from_redo_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "scheme_2.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            json_path = tmp / "redo_run.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "image_name": "scheme_2.png",
                            "redo": True,
                            "plan": [{"id": "tool_1", "name": "text_extraction_agent", "arguments": {"image_path": str(image_path)}}],
                            "execution_logs": [
                                {
                                    "id": "tool_1",
                                    "name": "text_extraction_agent",
                                    "arguments": {"image_path": str(image_path)},
                                    "result": {
                                        "reactions": [
                                            {
                                                "reaction_id": "0_1",
                                                "reactants": [{"smiles": "CCO", "label": "A"}],
                                                "products": [{"smiles": "CC=O", "label": "B"}],
                                                "conditions": [{"role": "temperature", "text": "rt"}],
                                                "additional_info": [{"text": "Nested reaction"}],
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            service = ReviewDatasetService(ReviewRepository(db_path))
            result = service.submit_sideload_experiment(
                experiment_name="sideload-redo",
                notes="",
                json_paths=[str(json_path)],
                recovery_roots=[str(tmp)],
                config_snapshot={
                    "ARTIFACT_BACKEND": "filesystem",
                    "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                },
            )
            service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")
            self.assertEqual(runs[0]["total_reactions"], 1)
            self.assertEqual(runs[0]["total_redo"], 0)

            reactions = service.list_reactions(run_id=runs[0]["run_id"])
            self.assertEqual(len(reactions), 1)
            self.assertEqual(reactions[0]["outcome_class"], "succeeded")

    def test_sideload_rejects_reactions_without_structural_smiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "scheme_3.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            json_path = tmp / "invalid_run.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "image_name": "scheme_3.png",
                            "reactions": [
                                {
                                    "reaction_id": "0_1",
                                    "reactants": [{"smiles": "None", "label": "A"}],
                                    "products": [{"smiles": None, "label": "B"}],
                                    "conditions": [{"role": "solvent", "text": "water", "smiles": "O"}],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            service = ReviewDatasetService(ReviewRepository(db_path))
            result = service.submit_sideload_experiment(
                experiment_name="sideload-invalid",
                notes="",
                json_paths=[str(json_path)],
                recovery_roots=[str(tmp)],
                config_snapshot={
                    "ARTIFACT_BACKEND": "filesystem",
                    "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                },
            )
            service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(runs[0]["total_reactions"], 0)
            reactions = service.list_reactions(run_id=runs[0]["run_id"])
            self.assertEqual(reactions, [])

            source = service.list_run_sources(runs[0]["run_id"])[0]
            detail = service.get_run_source_monitor(source["run_source_id"])
            derived = detail["derived_images"][0]
            self.assertEqual(derived["accepted_reaction_count"], 0)
            self.assertEqual(derived["rejected_reaction_count"], 1)
            self.assertEqual(derived["normalization_status"], "rejected_missing_smiles")

    def test_retry_derived_image_recovers_local_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "retry.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            store = create_artifact_store_from_config(
                {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
            )
            artifact_key = "derived/run-source/0.png"
            store.put_file(artifact_key, str(image_path), content_type="image/png")

            repo = ReviewRepository(db_path)
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="completed",
                config_snapshot={
                    "ARTIFACT_BACKEND": "filesystem",
                    "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                    "CHEMEAGLE_RUN_MODE": "cloud",
                },
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="image",
                sha256="abc123",
                original_filename="retry.png",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0, source_type="image")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="retry.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key=artifact_key,
                artifact_status="present",
                status="failed",
                outcome_class="failed",
                raw_artifact_key="",
                error_text="IndexError: list index out of range",
            )
            service = ReviewDatasetService(repo)
            valid_payload = {
                "reactions": [
                    {
                        "reaction_id": "rxn-1",
                        "reactants": [{"smiles": "CCO", "label": "A"}],
                        "products": [{"smiles": "CC=O", "label": "B"}],
                        "conditions": [],
                    }
                ]
            }
            with mock.patch.object(service, "_execute_image_pipeline", side_effect=IndexError("boom")), mock.patch.object(
                service,
                "_execute_image_pipeline_recovery_subprocess",
                return_value=(valid_payload, RunMetricsCollector()),
            ):
                status = service.retry_derived_image(derived_image_id)

            self.assertIn("reactions=1", status)
            reactions = service.list_reactions(run_id=run_id)
            self.assertEqual(len(reactions), 1)
            detail = service.get_run_source_monitor(run_source_id)
            derived = detail["derived_images"][0]
            self.assertEqual(derived["attempt_count"], 3)
            self.assertEqual(derived["accepted_reaction_count"], 1)
            self.assertEqual(derived["normalization_status"], "accepted")
            self.assertEqual(derived["attempts"][-1]["execution_mode"], "recovery")

    def test_retry_derived_image_manual_no_agents_records_execution_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "retry_no_agents.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            store = create_artifact_store_from_config(
                {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
            )
            artifact_key = "derived/run-source/0.png"
            store.put_file(artifact_key, str(image_path), content_type="image/png")

            repo = ReviewRepository(db_path)
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="completed",
                config_snapshot={
                    "ARTIFACT_BACKEND": "filesystem",
                    "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                    "CHEMEAGLE_RUN_MODE": "cloud",
                },
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="image",
                sha256="abc123",
                original_filename="retry_no_agents.png",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0, source_type="image")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="retry_no_agents.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key=artifact_key,
                artifact_status="present",
                status="failed",
                outcome_class="failed",
                raw_artifact_key="",
                error_text="observer returned empty tool list",
            )
            service = ReviewDatasetService(repo)
            execution_modes: list[str] = []
            valid_payload = {
                "reactions": [
                    {
                        "reaction_id": "rxn-1",
                        "reactants": [{"smiles": "CCO", "label": "A"}],
                        "products": [{"smiles": "CC=O", "label": "B"}],
                        "conditions": [],
                    }
                ]
            }

            def fake_execute(image_path: str, config: dict, *, execution_mode: str = "normal"):
                del image_path, config
                execution_modes.append(execution_mode)
                return valid_payload, RunMetricsCollector()

            with mock.patch.object(service, "_execute_image_pipeline", side_effect=fake_execute):
                status = service.retry_derived_image(derived_image_id, execution_mode="no_agents")

            self.assertIn("using no_agents", status)
            self.assertEqual(execution_modes, ["no_agents"])
            detail = service.get_run_source_monitor(run_source_id)
            attempts = detail["derived_images"][0]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["execution_mode"], "normal")
            self.assertEqual(attempts[0]["trigger"], "initial")
            self.assertEqual(attempts[-1]["execution_mode"], "no_agents")
            self.assertEqual(attempts[-1]["trigger"], "manual_no_agents_retry")

    def test_retry_derived_image_persists_partial_payload_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "retry_partial.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            store = create_artifact_store_from_config(
                {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
            )
            artifact_key = "derived/run-source/0.png"
            store.put_file(artifact_key, str(image_path), content_type="image/png")

            repo = ReviewRepository(db_path)
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="completed",
                config_snapshot={
                    "ARTIFACT_BACKEND": "filesystem",
                    "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                    "CHEMEAGLE_RUN_MODE": "cloud",
                },
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="image",
                sha256="abc123",
                original_filename="retry_partial.png",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0, source_type="image")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="retry_partial.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key=artifact_key,
                artifact_status="present",
                status="failed",
                outcome_class="failed",
                raw_artifact_key="",
                error_text="list index out of range",
            )
            service = ReviewDatasetService(repo)
            partial_payload = {
                "partial": True,
                "error": "list index out of range",
                "traceback": "Traceback...",
                "image_path": "/tmp/worker/retry_partial.png",
                "failed_tool_id": "tool_call_0",
                "failed_tool_name": "get_full_reaction_template",
                "failed_tool_arguments": {"image_path": "/tmp/worker/retry_partial.png"},
                "conversion_summary": {"converted_bbox_count": 0, "skipped_bbox_count": 1},
                "tool_warnings": [{"bbox_index": 0, "reason": "edge matrix must be square"}],
                "plan": [{"id": "tool_call_0", "name": "get_full_reaction_template", "arguments": {"image_path": "/tmp/worker/retry_partial.png"}}],
                "execution_logs": [
                    {
                        "id": "tool_call_0",
                        "name": "get_full_reaction_template",
                        "arguments": {"image_path": "/tmp/worker/retry_partial.png"},
                        "result": {"partial": True, "conversion_summary": {"converted_bbox_count": 0, "skipped_bbox_count": 1}},
                    }
                ],
            }

            with mock.patch.object(service, "_execute_image_pipeline", return_value=(partial_payload, RunMetricsCollector())):
                status = service.retry_derived_image(derived_image_id)

            self.assertIn("failures=1", status)
            detail = service.get_run_source_monitor(run_source_id)
            derived = detail["derived_images"][0]
            self.assertTrue(derived["raw_artifact_key"])
            stored_payload = json.loads(store.get_bytes(derived["raw_artifact_key"]).decode("utf-8"))
            expected_ref = str((artifact_root / artifact_key).resolve())
            self.assertEqual(stored_payload["image_path"], expected_ref)
            self.assertEqual(stored_payload["original_image_path"], "/tmp/worker/retry_partial.png")
            self.assertEqual(stored_payload["failed_tool_name"], "get_full_reaction_template")
            self.assertEqual(stored_payload["plan"][0]["arguments"]["image_path"], expected_ref)
            self.assertEqual(stored_payload["execution_logs"][0]["arguments"]["image_path"], expected_ref)

    def test_live_image_redo_auto_retries_with_no_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "scheme.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            service = ReviewDatasetService(ReviewRepository(db_path))
            execution_modes: list[str] = []
            redo_payload = {
                "redo": True,
                "plan": [],
                "execution_logs": [],
                "reactions": [],
            }
            valid_payload = {
                "reactions": [
                    {
                        "reaction_id": "rxn-1",
                        "reactants": [{"smiles": "CCO", "label": "A"}],
                        "products": [{"smiles": "CC=O", "label": "B"}],
                        "conditions": [],
                    }
                ]
            }

            def fake_execute(image_path: str, config: dict, *, execution_mode: str = "normal"):
                del image_path, config
                execution_modes.append(execution_mode)
                if execution_mode == "normal":
                    return redo_payload, RunMetricsCollector()
                if execution_mode == "no_agents":
                    return valid_payload, RunMetricsCollector()
                raise AssertionError(f"Unexpected execution mode: {execution_mode}")

            with mock.patch.object(service, "_execute_image_pipeline", side_effect=fake_execute):
                result = service.submit_live_experiment(
                    experiment_name="redo batch",
                    notes="",
                    source_paths=[str(image_path)],
                    profile_configs=[
                        {
                            "profile_label": "baseline",
                            "ARTIFACT_BACKEND": "filesystem",
                            "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                            "CHEMEAGLE_RUN_MODE": "cloud",
                        }
                    ],
                )
                service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")
            self.assertEqual(runs[0]["total_reactions"], 1)
            self.assertEqual(execution_modes, ["normal", "no_agents"])

            run_id = runs[0]["run_id"]
            sources = service.list_run_sources(run_id)
            self.assertEqual(len(sources), 1)
            detail = service.get_run_source_monitor(sources[0]["run_source_id"])
            derived = detail["derived_images"][0]
            attempts = derived["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["execution_mode"], "normal")
            self.assertEqual(attempts[1]["execution_mode"], "no_agents")
            self.assertEqual(attempts[1]["trigger"], "auto_no_agents_retry")
            self.assertEqual(derived["accepted_reaction_count"], 1)

    def test_reprocess_normalization_for_run_purges_invalid_canonical_reactions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            store = create_artifact_store_from_config(
                {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
            )
            repo = ReviewRepository(db_path)
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="sideload_json",
                status="completed",
                config_snapshot={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)},
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
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0, source_type="image")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="scheme.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key="derived/run-source/0.png",
                artifact_status="present",
                status="completed",
                outcome_class="needs_redo",
                raw_artifact_key="raw/run/source/image_0_attempt_1.json",
            )
            raw_payload = {
                "redo": True,
                "execution_logs": [
                    {
                        "result": {
                            "reactions": [
                                {
                                    "reaction_id": "rxn-1",
                                    "reactants": [{"smiles": "CCO", "label": "A"}],
                                    "products": [{"smiles": "CC=O", "label": "B"}],
                                    "conditions": [],
                                }
                            ]
                        }
                    }
                ],
            }
            store.put_bytes("raw/run/source/image_0_attempt_1.json", json.dumps(raw_payload).encode("utf-8"), "application/json")
            bad_reaction_uid = repo.create_reaction(
                run_id=run_id,
                run_source_id=run_source_id,
                derived_image_id=derived_image_id,
                attempt_id="attempt-old",
                reaction_id="bad",
                reaction_fingerprint="bad",
                outcome_class="failed",
                structure_quality="",
                acceptance_reason="",
                render_artifact_key="",
                raw_reaction_json=json.dumps({"reaction_id": "bad", "reactants": [], "products": [{"smiles": None, "label": "X"}]}),
            )
            repo.add_reaction_molecules(
                bad_reaction_uid,
                [{"side": "product", "ordinal": 0, "smiles": "", "label": "X", "valid_smiles": False, "validation_kind": "missing"}],
            )
            service = ReviewDatasetService(repo)
            summary = service.reprocess_normalization_for_run(run_id, only_invalid_reactions=False)

            self.assertEqual(summary["accepted_reactions"], 1)
            reactions = service.list_reactions(run_id=run_id)
            self.assertEqual(len(reactions), 1)
            self.assertEqual(reactions[0]["reaction_id"], "rxn-1")

    def test_live_pdf_persists_all_extracted_images_and_continues_after_crop_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            pdf_path = tmp / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%dummy\n")

            service = ReviewDatasetService(ReviewRepository(db_path))

            def fake_run_pdf(*, pdf_dir=None, image_dir=None, model_size="large", config_file=None):
                del pdf_dir, model_size, config_file
                out_dir = Path(image_dir)
                Image.new("RGB", (32, 32), "white").save(out_dir / "paper_image_2_1.png")
                Image.new("RGB", (32, 32), "white").save(out_dir / "paper_image_7_1.png")

            valid_payload = {
                "reactions": [
                    {
                        "reaction_id": "rxn-1",
                        "reactants": [{"smiles": "CCO", "label": "A"}],
                        "products": [{"smiles": "CC=O", "label": "B"}],
                        "conditions": [{"role": "temperature", "text": "rt"}],
                        "additional_info": [{"text": "Yield 80%"}],
                    }
                ]
            }

            with mock.patch("pdf_extraction.run_pdf", side_effect=fake_run_pdf), mock.patch.object(
                service,
                "_execute_image_pipeline",
                side_effect=[
                    RuntimeError("first crop exploded"),
                    (valid_payload, RunMetricsCollector()),
                ],
            ):
                result = service.submit_live_experiment(
                    experiment_name="pdf batch",
                    notes="",
                    source_paths=[str(pdf_path)],
                    profile_configs=[
                        {
                            "profile_label": "baseline",
                            "ARTIFACT_BACKEND": "filesystem",
                            "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                            "CHEMEAGLE_RUN_MODE": "cloud",
                        }
                    ],
                )
                service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")
            self.assertEqual(runs[0]["total_sources"], 1)
            self.assertEqual(runs[0]["total_derived_images"], 2)
            self.assertEqual(runs[0]["total_failures"], 1)
            self.assertEqual(runs[0]["total_reactions"], 1)

            derived_paths = sorted(path.relative_to(artifact_root).as_posix() for path in artifact_root.rglob("*.png"))
            self.assertTrue(any(path.endswith("/0.png") for path in derived_paths))
            self.assertTrue(any(path.endswith("/1.png") for path in derived_paths))

            with service.repository.connect() as conn:
                rows = conn.execute(
                    "SELECT image_index, outcome_class, error_text FROM derived_images ORDER BY image_index"
                ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["outcome_class"], "failed")
            self.assertIn("first crop exploded", rows[0]["error_text"])
            self.assertEqual(rows[1]["outcome_class"], "succeeded")

    def test_live_batch_aborts_early_on_systemic_connection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            pdf_a = tmp / "paper-a.pdf"
            pdf_b = tmp / "paper-b.pdf"
            pdf_a.write_bytes(b"%PDF-1.4\n%a\n")
            pdf_b.write_bytes(b"%PDF-1.4\n%b\n")

            service = ReviewDatasetService(ReviewRepository(db_path))

            def fake_run_pdf(*, pdf_dir=None, image_dir=None, model_size="large", config_file=None):
                del model_size, config_file
                out_dir = Path(image_dir)
                stem = Path(pdf_dir).stem
                Image.new("RGB", (32, 32), "white").save(out_dir / f"{stem}_image_2_1.png")

            with mock.patch("pdf_extraction.run_pdf", side_effect=fake_run_pdf), mock.patch.object(
                service,
                "_execute_image_pipeline",
                side_effect=RuntimeError("Connection error."),
            ):
                result = service.submit_live_experiment(
                    experiment_name="abort batch",
                    notes="",
                    source_paths=[str(pdf_a), str(pdf_b)],
                    profile_configs=[
                        {
                            "profile_label": "baseline",
                            "ARTIFACT_BACKEND": "filesystem",
                            "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                            "CHEMEAGLE_RUN_MODE": "cloud",
                        }
                    ],
                )
                service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "failed")
            self.assertEqual(runs[0]["total_sources"], 2)
            self.assertEqual(runs[0]["total_derived_images"], 1)
            self.assertEqual(runs[0]["total_failures"], 1)
            self.assertIn("Connection error.", runs[0]["failure_summary"])
            self.assertEqual(runs[0]["systemic_failure_kind"], "dns_or_connection_error")

            run_id = runs[0]["run_id"]
            monitor = service.get_run_monitor(run_id, tail_lines=50, min_level="INFO", raw=False)
            self.assertEqual(monitor["run"]["status"], "failed")
            self.assertEqual(len(monitor["sources"]), 2)
            self.assertEqual(monitor["sources"][0]["status"], "aborted")
            self.assertEqual(monitor["sources"][1]["status"], "queued")
            self.assertIn("run_aborted", monitor["log_tail"]["formatted"])
            self.assertFalse(monitor["progress"]["is_active"])
            self.assertEqual(monitor["progress"]["progress_total_sources"], 2)
            self.assertEqual(monitor["progress"]["progress_completed_sources"], 1)
            self.assertEqual(monitor["progress"]["current_phase_label"], "Failed")
            self.assertTrue(monitor["progress"]["has_troubleshooting_logs"])
            self.assertEqual(monitor["aggregates"]["systemic_provider_failures"], 1)

            source_detail = service.get_run_source_monitor(monitor["sources"][0]["run_source_id"])
            self.assertEqual(len(source_detail["derived_images"]), 1)
            self.assertEqual(source_detail["derived_images"][0]["status"], "failed")

    def test_dataset_maintenance_repairs_blank_failure_kind_and_normalization_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            repo = ReviewRepository(tmp / "review.sqlite3")
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="completed",
                config_snapshot={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(tmp / "artifacts")},
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="image",
                sha256="sha-a",
                original_filename="broken.png",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.png",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0, source_type="image")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="broken.png",
                image_index=0,
                artifact_backend="filesystem",
                artifact_key="derived/source-a/0.png",
                artifact_status="present",
                status="failed",
                outcome_class="failed",
                raw_artifact_key="",
                error_text="Connection error.",
            )
            attempt = repo.create_derived_image_attempt(
                derived_image_id=derived_image_id,
                trigger="initial",
                execution_mode="normal",
                status="queued",
                config_snapshot_json="{}",
            )
            repo.update_derived_image_status(derived_image_id, last_attempt_id=str(attempt["attempt_id"]), attempt_count=1)

            service = ReviewDatasetService(repo)
            summary = service.run_dataset_maintenance(run_id)

            detail = service.get_run_source_monitor(run_source_id)
            derived = detail["derived_images"][0]
            latest_attempt = derived["attempts"][-1]
            self.assertGreaterEqual(summary["repaired_attempts"], 1)
            self.assertEqual(latest_attempt["status"], "failed")
            self.assertEqual(latest_attempt["failure_kind"], "provider_systemic")
            self.assertEqual(latest_attempt["error_summary"], "Connection error.")
            self.assertEqual(derived["normalization_status"], "none_found")

    def test_dataset_maintenance_rescues_logged_partial_payload_into_raw_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime_root = tmp / "runtime_logs" / "active"
            repo = ReviewRepository(tmp / "review.sqlite3")
            artifact_root = tmp / "artifacts"
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="failed",
                config_snapshot={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)},
                config_hash="hash",
            )
            source_asset_id = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="pdf",
                sha256="sha-a",
                original_filename="paper.pdf",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.pdf",
                artifact_status="present",
            )
            run_source_id = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_id, input_order=0, source_type="pdf")
            store = create_artifact_store_from_config(
                {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
            )
            derived_source = tmp / "paper_image_5_1.png"
            Image.new("RGB", (32, 32), "white").save(derived_source)
            derived_key = "derived/run-source/5.png"
            store.put_file(derived_key, str(derived_source), content_type="image/png")
            derived_image_id = repo.create_derived_image(
                run_source_id=run_source_id,
                page_hint="paper_image_5_1.png",
                image_index=5,
                artifact_backend="filesystem",
                artifact_key=derived_key,
                artifact_status="present",
                status="failed",
                outcome_class="failed",
                raw_artifact_key="",
                error_text="list index out of range",
            )
            attempt = repo.create_derived_image_attempt(
                derived_image_id=derived_image_id,
                trigger="initial",
                execution_mode="normal",
                status="failed",
                config_snapshot_json="{}",
            )
            repo.update_derived_image_status(derived_image_id, last_attempt_id=str(attempt["attempt_id"]), attempt_count=1)

            partial_payload = {
                "partial": True,
                "error": "list index out of range",
                "traceback": "Traceback...",
                "image_path": "/tmp/worker/paper_image_5_1.png",
                "plan": [{"id": "tool_call_0", "name": "get_full_reaction_template", "arguments": {"image_path": "/tmp/worker/paper_image_5_1.png"}}],
                "execution_logs": [],
            }
            active_dir = runtime_root / run_id
            active_dir.mkdir(parents=True, exist_ok=True)
            (active_dir / "stdout.log").write_text(repr(partial_payload) + "\n", encoding="utf-8")

            service = ReviewDatasetService(repo)
            with mock.patch.object(review_service_module, "ACTIVE_RUN_LOG_ROOT", runtime_root):
                summary = service.run_dataset_maintenance(run_id)

            detail = service.get_run_source_monitor(run_source_id)
            derived = detail["derived_images"][0]
            self.assertEqual(summary["rescued_partial_artifacts"], 1)
            self.assertTrue(derived["raw_artifact_key"])
            stored_payload = json.loads(store.get_bytes(derived["raw_artifact_key"]).decode("utf-8"))
            self.assertEqual(stored_payload["original_image_path"], "/tmp/worker/paper_image_5_1.png")
            self.assertEqual(stored_payload["image_artifact_key"], derived_key)

    def test_service_recovers_stale_running_runs_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            repo = ReviewRepository(tmp / "review.sqlite3")
            experiment_id = repo.create_experiment(name="exp")
            run_id = repo.create_run(
                experiment_id=experiment_id,
                profile_label="baseline",
                ingest_mode="live_batch",
                status="running",
                config_snapshot={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(tmp / "artifacts")},
                config_hash="hash",
            )
            source_asset_a = repo.upsert_source_asset(
                source_asset_id="source-a",
                source_type="pdf",
                sha256="sha-a",
                original_filename="paper-a.pdf",
                artifact_backend="filesystem",
                artifact_key="sources/source-a/original.pdf",
                artifact_status="present",
            )
            source_asset_b = repo.upsert_source_asset(
                source_asset_id="source-b",
                source_type="pdf",
                sha256="sha-b",
                original_filename="paper-b.pdf",
                artifact_backend="filesystem",
                artifact_key="sources/source-b/original.pdf",
                artifact_status="present",
            )
            run_source_a = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_a, input_order=0, source_type="pdf")
            run_source_b = repo.create_run_source(run_id=run_id, source_asset_id=source_asset_b, input_order=1, source_type="pdf")
            repo.update_run_live_state(
                run_id,
                current_run_source_id=run_source_a,
                current_source_name="paper-a.pdf",
                current_phase="extract_pdf_images",
                status_message="Extracting page 3",
            )
            repo.update_run_source_status(run_source_a, status="extracting", current_phase="extract_pdf_images", started=True)
            repo.update_run_source_status(run_source_b, status="queued", current_phase="prepare_source")

            service = ReviewDatasetService(repo)

            run_row = service.repository.get_run(run_id)
            self.assertIsNotNone(run_row)
            self.assertEqual(run_row["status"], "interrupted")
            self.assertEqual(run_row["current_phase"], "interrupted")
            self.assertIn("Application stopped", run_row["failure_summary"])

            sources = service.list_run_sources(run_id)
            self.assertEqual(sources[0]["status"], "aborted")
            self.assertEqual(sources[1]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
