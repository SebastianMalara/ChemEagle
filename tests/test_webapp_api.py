from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from webapp.api.app import app


class _FakeRepository:
    def get_run(self, run_id: str):  # noqa: ANN001
        return {
            "run_id": run_id,
            "config_snapshot_json": '{"ARTIFACT_BACKEND":"filesystem","ARTIFACT_FILESYSTEM_ROOT":"/tmp/artifacts"}',
        }


class _FakeReviewService:
    def __init__(self) -> None:
        self.repository = _FakeRepository()
        self.saved_reviews = []

    def submit_live_experiment(self, **kwargs):  # noqa: ANN003
        self.submitted_live = kwargs
        return {"experiment_id": "exp-1", "run_ids": ["run-1"]}

    def list_experiments(self):
        return [{"experiment_id": "exp-1", "name": "Experiment", "run_count": 1}]

    def list_runs(self, experiment_id: str = ""):
        del experiment_id
        return [{"run_id": "run-1", "status": "running", "experiment_name": "Experiment"}]

    def get_run_monitor(self, run_id: str, **kwargs):  # noqa: ANN003
        del kwargs
        return {
            "run": {"run_id": run_id, "status": "running"},
            "progress": {"status_summary": "Working", "progress_fraction": 0.5},
            "sources": [{"run_source_id": "source-1", "original_filename": "paper.pdf"}],
            "log_tail": {"formatted": "tail", "raw": "tail", "events": []},
            "aggregates": {"planner_tool_empties": 0},
        }

    def get_log_download_ref(self, run_id: str) -> str:
        return f"/logs/{run_id}.log"

    def list_retry_candidates(self, run_id: str):
        del run_id
        return [{"derived_image_id": "derived-1", "status": "failed"}]

    def get_run_source_monitor(self, run_source_id: str):
        return {"run_source_id": run_source_id, "derived_images": [{"derived_image_id": "derived-1", "attempts": []}]}

    def export_run_to_parquet(self, run_id: str, output_dir: str):
        return {"reactions": f"{output_dir}/{run_id}_reactions.parquet"}

    def retry_failed_derived_images(self, run_id: str, **kwargs):  # noqa: ANN003
        del run_id, kwargs
        return ["derived-1"]

    def reprocess_normalization_for_run(self, run_id: str, **kwargs):  # noqa: ANN003
        del run_id, kwargs
        return {"derived_images": 1, "accepted_reactions": 2}

    def retry_derived_image(self, derived_image_id: str, **kwargs):  # noqa: ANN003
        del kwargs
        return f"Retried {derived_image_id}"

    def reprocess_normalization_for_derived_images(self, derived_image_ids, **kwargs):  # noqa: ANN003, ANN001
        del derived_image_ids, kwargs
        return {"accepted_reactions": 1, "rejected_reactions": 0}

    def list_reactions(self, **kwargs):  # noqa: ANN003
        del kwargs
        return [{"reaction_uid": "rxn-1", "reaction_id": "0_1"}]

    def get_reaction_detail(self, reaction_uid: str):
        return {
            "reaction_uid": reaction_uid,
            "reaction_id": "0_1",
            "run_id": "run-1",
            "raw_reaction_json": '{"reaction_id":"0_1"}',
            "review_status": "unchecked",
            "review_notes": "",
            "source_artifact_backend": "filesystem",
            "source_artifact_key": "source/paper.pdf",
            "derived_backend": "filesystem",
            "derived_artifact_key": "derived/image.png",
            "render_artifact_key": "render/reaction.png",
            "original_filename": "paper.pdf",
            "conditions": [],
            "additional_info": [],
            "molecules": [],
        }

    def update_reaction_review(self, reaction_uid: str, *, review_status: str, review_notes: str) -> None:
        self.saved_reviews.append((reaction_uid, review_status, review_notes))


class WebappApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_get_and_update_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = str(Path(tmpdir) / ".env.chemeagle")
            response = self.client.get("/api/config", params={"env_path": env_path})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["values"]["LLM_PROVIDER"], "azure")

            update = self.client.put(
                "/api/config",
                json={
                    "env_path": env_path,
                    "persist_to_env": True,
                    "values": {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "REVIEW_DB_PATH": "/tmp/review.sqlite3"},
                },
            )
            self.assertEqual(update.status_code, 200)
            payload = update.json()
            self.assertEqual(payload["values"]["LLM_PROVIDER"], "openai")
            self.assertIn("Saved", payload["save_status"])
            self.assertIn("LLM_PROVIDER=openai", Path(env_path).read_text(encoding="utf-8"))

    def test_model_refresh_and_preflight_routes(self) -> None:
        with mock.patch(
            "webapp.api.routers.config.refresh_model_catalog",
            return_value={"scope": "main", "selected_model": "gpt-5-mini", "models": ["gpt-5-mini"], "status": "ok"},
        ), mock.patch(
            "webapp.api.routers.config.collect_preflight_diagnostics",
            return_value={"blocking_errors": [], "warnings": [], "resolved_ocr_backend": "easyocr"},
        ):
            refresh = self.client.post(
                "/api/models/refresh",
                json={"scope": "main", "current_model": "", "values": {"LLM_PROVIDER": "openai"}},
            )
            self.assertEqual(refresh.status_code, 200)
            self.assertEqual(refresh.json()["selected_model"], "gpt-5-mini")

            preflight = self.client.post(
                "/api/preflight/runtime",
                json={
                    "file_path": "",
                    "mode": "cloud",
                    "include_pdf_section": False,
                    "values": {"LLM_PROVIDER": "openai"},
                },
            )
            self.assertEqual(preflight.status_code, 200)
            self.assertEqual(preflight.json()["diagnostics"]["resolved_ocr_backend"], "easyocr")

    def test_live_experiment_route(self) -> None:
        fake_service = _FakeReviewService()
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = str(Path(tmpdir) / ".env.chemeagle")
            image_path = str(Path(tmpdir) / "sample.png")
            Path(image_path).write_bytes(b"png")
            with mock.patch(
                "webapp.api.routers.experiments.review_service_for",
                return_value=fake_service,
            ), mock.patch(
                "webapp.api.routers.experiments.collect_batch_runtime_diagnostics",
                return_value={
                    "blocking_errors": [],
                    "warnings": [],
                    "runtime_provider_preflight": {"status": "ok"},
                },
            ):
                response = self.client.post(
                    "/api/experiments/live",
                    json={
                        "env_path": env_path,
                        "persist_to_env": False,
                        "values": {
                            "CHEMEAGLE_RUN_MODE": "cloud",
                            "REVIEW_DB_PATH": "/tmp/review.sqlite3",
                        },
                        "experiment_name": "Experiment",
                        "uploaded_paths": [image_path],
                        "comparison_profiles": [{"profile_label": "baseline"}],
                    },
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["experiment_id"], "exp-1")
            self.assertEqual(fake_service.submitted_live["source_paths"], [image_path])

    def test_run_monitor_and_run_actions(self) -> None:
        fake_service = _FakeReviewService()
        with mock.patch("webapp.api.routers.runs.review_service_for", return_value=fake_service):
            monitor = self.client.get("/api/runs/run-1/monitor", params={"review_db_path": "/tmp/review.sqlite3"})
            self.assertEqual(monitor.status_code, 200)
            self.assertEqual(monitor.json()["log_download_ref"], "/logs/run-1.log")

            export = self.client.post(
                "/api/runs/run-1/export",
                json={"output_dir": "/tmp/exports", "review_db_path": "/tmp/review.sqlite3"},
            )
            self.assertEqual(export.status_code, 200)
            self.assertIn("run-1_reactions.parquet", export.json()["files"]["reactions"])

            retry = self.client.post(
                "/api/derived-images/derived-1/retry",
                json={"review_db_path": "/tmp/review.sqlite3", "retry_mode": "recovery"},
            )
            self.assertEqual(retry.status_code, 200)
            self.assertIn("Retried derived-1", retry.json()["status_text"])

    def test_review_detail_and_save(self) -> None:
        fake_service = _FakeReviewService()
        with mock.patch("webapp.api.routers.review.review_service_for", return_value=fake_service):
            detail = self.client.get("/api/review/reactions/rxn-1", params={"review_db_path": "/tmp/review.sqlite3"})
            self.assertEqual(detail.status_code, 200)
            payload = detail.json()
            self.assertEqual(payload["raw_reaction_json"]["reaction_id"], "0_1")
            self.assertIn("/api/review/reactions/rxn-1/artifacts/source", payload["source_url"])

            saved = self.client.put(
                "/api/review/reactions/rxn-1",
                json={"review_db_path": "/tmp/review.sqlite3", "review_status": "ok", "review_notes": "looks good"},
            )
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(fake_service.saved_reviews, [("rxn-1", "ok", "looks good")])


if __name__ == "__main__":
    unittest.main()
