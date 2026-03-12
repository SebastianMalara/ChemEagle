from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from PIL import Image

import review_service as review_service_module
from review_db import ReviewRepository
from review_service import ReviewDatasetService, _apply_runtime_env, _cleanup_runtime_after_source, _rewrite_payload_image_refs
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

    def test_cleanup_runtime_after_source_clears_per_source_caches(self) -> None:
        fake_r_group = mock.Mock()
        fake_r_group._process_multi_molecular_cache = {"a": object(), "b": object()}
        fake_r_group._raw_results_cache = {"c": object()}
        fake_r_group._RUNTIME_TOOLKIT = object()
        fake_r_group._RUNTIME_RXNIM = object()
        fake_r_group._RUNTIME_DEVICE_TYPE = "mps"
        fake_reaction_agent = mock.Mock()
        fake_reaction_agent._RUNTIME_RXNIM = object()
        fake_reaction_agent._RUNTIME_DEVICE_TYPE = "mps"
        fake_molecular_agent = mock.Mock()
        fake_molecular_agent._RUNTIME_TOOLKIT = object()
        fake_molecular_agent._RUNTIME_DEVICE_TYPE = "mps"
        fake_text_agent = mock.Mock()
        fake_text_agent._CHEMNER_MODELS = {"chemner": object()}
        fake_text_agent._RXN_EXTRACTORS = {"rxn": object()}
        fake_text_agent._EASYOCR_READERS = {("en", "cpu"): object()}
        fake_text_agent._OCR_TEXT_CACHE = {("img", "easyocr"): "text"}
        fake_cuda = mock.Mock()
        fake_cuda.is_available.return_value = False
        fake_mps = mock.Mock()
        fake_mps.is_available.return_value = True
        fake_torch = mock.Mock(cuda=fake_cuda, mps=fake_mps)
        fake_toolkit_cls = mock.Mock()
        fake_toolkit_cls.init_molnextr.cache_clear = mock.Mock()
        fake_toolkit_cls.init_rxnim.cache_clear = mock.Mock()
        fake_toolkit_cls.init_pdfparser.cache_clear = mock.Mock()
        fake_toolkit_cls.init_moldet.cache_clear = mock.Mock()
        fake_toolkit_cls.init_coref.cache_clear = mock.Mock()
        fake_toolkit_cls.init_chemrxnextractor.cache_clear = mock.Mock()
        fake_toolkit_cls.init_chemner.cache_clear = mock.Mock()
        fake_toolkit_interface = mock.Mock(ChemIEToolkit=fake_toolkit_cls)
        logger = mock.Mock()

        with mock.patch.dict(
            "sys.modules",
            {
                "get_R_group_sub_agent": fake_r_group,
                "get_reaction_agent": fake_reaction_agent,
                "get_molecular_agent": fake_molecular_agent,
                "get_text_agent": fake_text_agent,
                "chemietoolkit.interface": fake_toolkit_interface,
                "torch": fake_torch,
            },
            clear=False,
        ), mock.patch.object(review_service_module.gc, "collect", return_value=7):
            _cleanup_runtime_after_source("batch-1.pdf", logger)

        self.assertEqual(fake_r_group._process_multi_molecular_cache, {})
        self.assertEqual(fake_r_group._raw_results_cache, {})
        self.assertIsNone(fake_r_group._RUNTIME_TOOLKIT)
        self.assertIsNone(fake_r_group._RUNTIME_RXNIM)
        self.assertIsNone(fake_r_group._RUNTIME_DEVICE_TYPE)
        self.assertIsNone(fake_reaction_agent._RUNTIME_RXNIM)
        self.assertIsNone(fake_reaction_agent._RUNTIME_DEVICE_TYPE)
        self.assertIsNone(fake_molecular_agent._RUNTIME_TOOLKIT)
        self.assertIsNone(fake_molecular_agent._RUNTIME_DEVICE_TYPE)
        self.assertEqual(fake_text_agent._CHEMNER_MODELS, {})
        self.assertEqual(fake_text_agent._RXN_EXTRACTORS, {})
        self.assertEqual(fake_text_agent._EASYOCR_READERS, {})
        self.assertEqual(fake_text_agent._OCR_TEXT_CACHE, {})
        fake_toolkit_cls.init_molnextr.cache_clear.assert_called_once_with()
        fake_toolkit_cls.init_rxnim.cache_clear.assert_called_once_with()
        fake_toolkit_cls.init_pdfparser.cache_clear.assert_called_once_with()
        fake_toolkit_cls.init_moldet.cache_clear.assert_called_once_with()
        fake_toolkit_cls.init_coref.cache_clear.assert_called_once_with()
        fake_toolkit_cls.init_chemrxnextractor.cache_clear.assert_called_once_with()
        fake_toolkit_cls.init_chemner.cache_clear.assert_called_once_with()
        fake_mps.empty_cache.assert_called_once_with()
        logger.info.assert_called_once()
        _, message = logger.info.call_args.args[:2]
        self.assertIn("batch-1.pdf", message)
        self.assertEqual(logger.info.call_args.kwargs["gc_collected"], 7)
        self.assertEqual(logger.info.call_args.kwargs["allocator_caches_cleared"], "mps")

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

    def test_live_image_redo_with_allowlisted_observer_issue_auto_retries_with_no_agents(self) -> None:
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
                "observer_verdict": {
                    "redo": True,
                    "reason": "No reaction extracted from the tool outputs.",
                    "issue_codes": ["no_reaction_detected"],
                    "confidence": 0.94,
                },
                "plan": [],
                "execution_logs": [],
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

    def test_live_image_redo_with_accepted_reactions_does_not_auto_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "accepted.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            service = ReviewDatasetService(ReviewRepository(db_path))
            execution_modes: list[str] = []
            payload = {
                "redo": True,
                "observer_verdict": {
                    "redo": True,
                    "reason": "Potential chemistry issue detected.",
                    "issue_codes": ["suspect_smiles"],
                    "confidence": 0.72,
                },
                "reactions": [
                    {
                        "reaction_id": "rxn-1",
                        "reactants": [{"smiles": "CCO", "label": "A"}],
                        "products": [{"smiles": "CC=O", "label": "B"}],
                        "conditions": [],
                    }
                ],
            }

            def fake_execute(image_path: str, config: dict, *, execution_mode: str = "normal"):
                del image_path, config
                execution_modes.append(execution_mode)
                return payload, RunMetricsCollector()

            with mock.patch.object(service, "_execute_image_pipeline", side_effect=fake_execute):
                result = service.submit_live_experiment(
                    experiment_name="accepted redo batch",
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
            self.assertEqual(execution_modes, ["normal"])

            sources = service.list_run_sources(runs[0]["run_id"])
            detail = service.get_run_source_monitor(sources[0]["run_source_id"])
            derived = detail["derived_images"][0]
            self.assertEqual(derived["accepted_reaction_count"], 1)
            self.assertEqual(derived["normalization_status"], "accepted")
            self.assertEqual(len(derived["attempts"]), 1)

    def test_live_image_redo_with_rejected_candidates_does_not_auto_retry(self) -> None:
        cases = [
            (
                "missing",
                {
                    "reaction_id": "rxn-missing",
                    "reactants": [{"smiles": "None", "label": "A"}],
                    "products": [{"smiles": None, "label": "B"}],
                    "conditions": [],
                },
                "rejected_missing_smiles",
            ),
            (
                "invalid",
                {
                    "reaction_id": "rxn-invalid",
                    "reactants": [{"smiles": "C1=", "label": "A"}],
                    "products": [{"smiles": "C1=", "label": "B"}],
                    "conditions": [],
                },
                "rejected_invalid_smiles",
            ),
        ]
        for label, reaction, expected_status in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    db_path = tmp / "review.sqlite3"
                    artifact_root = tmp / "artifacts"
                    image_path = tmp / f"{label}.png"
                    Image.new("RGB", (64, 64), "white").save(image_path)

                    service = ReviewDatasetService(ReviewRepository(db_path))
                    execution_modes: list[str] = []
                    payload = {
                        "redo": True,
                        "observer_verdict": {
                            "redo": True,
                            "reason": "Extraction looked incomplete.",
                            "issue_codes": ["missing_core_reaction"],
                            "confidence": 0.88,
                        },
                        "reactions": [reaction],
                    }

                    def fake_execute(image_path: str, config: dict, *, execution_mode: str = "normal"):
                        del image_path, config
                        execution_modes.append(execution_mode)
                        return payload, RunMetricsCollector()

                    with mock.patch.object(service, "_execute_image_pipeline", side_effect=fake_execute):
                        result = service.submit_live_experiment(
                            experiment_name=f"{label} redo batch",
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
                    self.assertEqual(execution_modes, ["normal"])

                    sources = service.list_run_sources(runs[0]["run_id"])
                    detail = service.get_run_source_monitor(sources[0]["run_source_id"])
                    derived = detail["derived_images"][0]
                    self.assertEqual(derived["accepted_reaction_count"], 0)
                    self.assertEqual(derived["normalization_status"], expected_status)
                    self.assertEqual(len(derived["attempts"]), 1)

    def test_live_image_redo_with_malformed_or_empty_observer_verdict_does_not_auto_retry(self) -> None:
        cases = [
            ("empty_dict", {}),
            ("malformed_string", "not-json"),
        ]
        for label, verdict in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    db_path = tmp / "review.sqlite3"
                    artifact_root = tmp / "artifacts"
                    image_path = tmp / f"{label}.png"
                    Image.new("RGB", (64, 64), "white").save(image_path)

                    service = ReviewDatasetService(ReviewRepository(db_path))
                    execution_modes: list[str] = []
                    payload = {
                        "redo": True,
                        "observer_verdict": verdict,
                        "plan": [],
                        "execution_logs": [],
                    }

                    def fake_execute(image_path: str, config: dict, *, execution_mode: str = "normal"):
                        del image_path, config
                        execution_modes.append(execution_mode)
                        return payload, RunMetricsCollector()

                    with mock.patch.object(service, "_execute_image_pipeline", side_effect=fake_execute):
                        result = service.submit_live_experiment(
                            experiment_name=f"{label} redo batch",
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
                    self.assertEqual(execution_modes, ["normal"])

                    sources = service.list_run_sources(runs[0]["run_id"])
                    detail = service.get_run_source_monitor(sources[0]["run_source_id"])
                    derived = detail["derived_images"][0]
                    self.assertEqual(derived["normalization_status"], "none_found")
                    self.assertEqual(len(derived["attempts"]), 1)

    def test_live_image_redo_legacy_policy_auto_retries_with_no_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "legacy.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            service = ReviewDatasetService(ReviewRepository(db_path))
            execution_modes: list[str] = []
            redo_payload = {
                "redo": True,
                "observer_verdict": {},
                "plan": [],
                "execution_logs": [],
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
                    experiment_name="legacy redo batch",
                    notes="",
                    source_paths=[str(image_path)],
                    profile_configs=[
                        {
                            "profile_label": "baseline",
                            "ARTIFACT_BACKEND": "filesystem",
                            "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root),
                            "CHEMEAGLE_RUN_MODE": "cloud",
                            "CHEMEAGLE_REDO_POLICY": "legacy",
                        }
                    ],
                )
                service._queue.join()

            runs = service.list_runs(result["experiment_id"])
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "completed")
            self.assertEqual(execution_modes, ["normal", "no_agents"])

            sources = service.list_run_sources(runs[0]["run_id"])
            detail = service.get_run_source_monitor(sources[0]["run_source_id"])
            attempts = detail["derived_images"][0]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[1]["trigger"], "auto_no_agents_retry")

    def test_live_image_redo_after_no_agents_local_runtime_error_uses_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "review.sqlite3"
            artifact_root = tmp / "artifacts"
            image_path = tmp / "recovery.png"
            Image.new("RGB", (64, 64), "white").save(image_path)

            service = ReviewDatasetService(ReviewRepository(db_path))
            execution_modes: list[str] = []
            redo_payload = {
                "redo": True,
                "observer_verdict": {
                    "redo": True,
                    "reason": "No reaction detected.",
                    "issue_codes": ["no_reaction_detected"],
                    "confidence": 0.97,
                },
                "plan": [],
                "execution_logs": [],
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
                    raise IndexError("boom")
                raise AssertionError(f"Unexpected execution mode: {execution_mode}")

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(service, "_execute_image_pipeline", side_effect=fake_execute))
                stack.enter_context(
                    mock.patch.object(
                        service,
                        "_execute_image_pipeline_recovery_subprocess",
                        return_value=(valid_payload, RunMetricsCollector()),
                    )
                )
                result = service.submit_live_experiment(
                    experiment_name="recovery redo batch",
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
            self.assertEqual(execution_modes, ["normal", "no_agents"])

            sources = service.list_run_sources(runs[0]["run_id"])
            detail = service.get_run_source_monitor(sources[0]["run_source_id"])
            derived = detail["derived_images"][0]
            attempts = derived["attempts"]
            self.assertEqual(len(attempts), 3)
            self.assertEqual(attempts[1]["execution_mode"], "no_agents")
            self.assertEqual(attempts[2]["execution_mode"], "recovery")
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
