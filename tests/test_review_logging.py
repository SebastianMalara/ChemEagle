from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from review_artifacts import create_artifact_store_from_config
import review_logging
from review_logging import RunLogSession, read_log_tail


class ReviewLoggingTests(unittest.TestCase):
    def test_run_log_session_writes_tail_and_finalizes_to_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime_root = tmp / "runtime_logs"
            artifact_root = tmp / "artifacts"
            console_stream = io.StringIO()

            with mock.patch.object(review_logging, "RUNTIME_LOG_ROOT", runtime_root), mock.patch.object(
                review_logging,
                "ACTIVE_RUN_LOG_ROOT",
                runtime_root / "active",
            ), mock.patch.object(review_logging, "APP_LOG_PATH", runtime_root / "app.jsonl"), mock.patch.object(
                review_logging.sys,
                "__stderr__",
                console_stream,
            ):
                review_logging._APP_LOGGER_CONFIGURED = False
                session = RunLogSession(run_id="run-1", experiment_id="exp-1")
                session.logger.info("run_started", "Run started.")
                with session.capture_streams():
                    print("hello from stdout")
                    print("boom", file=__import__("sys").stderr)
                tail = read_log_tail(
                    run_id="run-1",
                    config={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)},
                    log_artifact_key="",
                    tail_lines=20,
                    min_level="INFO",
                    raw=False,
                )
                self.assertIn("run_started", tail["formatted"])
                self.assertIn("hello from stdout", tail["formatted"])
                self.assertIn("boom", tail["formatted"])

                store = create_artifact_store_from_config(
                    {"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(artifact_root)}
                )
                keys = session.finalize(store)
                session.close()

            console_output = console_stream.getvalue()
            self.assertIn("run_started", console_output)
            self.assertIn("hello from stdout", console_output)
            self.assertIn("boom", console_output)
            self.assertTrue((artifact_root / keys["log_artifact_key"]).exists())
            self.assertTrue((artifact_root / keys["stdout_artifact_key"]).exists())
            self.assertTrue((artifact_root / keys["stderr_artifact_key"]).exists())

    def test_stderr_userwarning_is_classified_as_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime_root = tmp / "runtime_logs"

            with mock.patch.object(review_logging, "RUNTIME_LOG_ROOT", runtime_root), mock.patch.object(
                review_logging,
                "ACTIVE_RUN_LOG_ROOT",
                runtime_root / "active",
            ), mock.patch.object(review_logging, "APP_LOG_PATH", runtime_root / "app.jsonl"):
                review_logging._APP_LOGGER_CONFIGURED = False
                session = RunLogSession(run_id="run-warning", experiment_id="exp-1")
                with session.capture_streams():
                    print("UserWarning: deprecated parameter", file=__import__("sys").stderr)
                tail = read_log_tail(
                    run_id="run-warning",
                    config={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(tmp / "artifacts")},
                    log_artifact_key="",
                    tail_lines=20,
                    min_level="DEBUG",
                    raw=False,
                )
                session.close()

            self.assertIn("WARNING", tail["formatted"])
            self.assertIn("UserWarning: deprecated parameter", tail["formatted"])

    def test_transformers_config_dump_is_grouped_and_hidden_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime_root = tmp / "runtime_logs"

            with mock.patch.object(review_logging, "RUNTIME_LOG_ROOT", runtime_root), mock.patch.object(
                review_logging,
                "ACTIVE_RUN_LOG_ROOT",
                runtime_root / "active",
            ), mock.patch.object(review_logging, "APP_LOG_PATH", runtime_root / "app.jsonl"):
                review_logging._APP_LOGGER_CONFIGURED = False
                session = RunLogSession(run_id="run-config", experiment_id="exp-1")
                with session.capture_streams():
                    print("Config of the encoder: <class 'BertModel'> is overwritten by shared encoder config: BertConfig {", file=__import__("sys").stderr)
                    print('  "hidden_size": 256,', file=__import__("sys").stderr)
                    print("}", file=__import__("sys").stderr)
                hidden_tail = read_log_tail(
                    run_id="run-config",
                    config={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(tmp / "artifacts")},
                    log_artifact_key="",
                    tail_lines=20,
                    min_level="DEBUG",
                    raw=False,
                )
                visible_tail = read_log_tail(
                    run_id="run-config",
                    config={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(tmp / "artifacts")},
                    log_artifact_key="",
                    tail_lines=20,
                    min_level="DEBUG",
                    raw=False,
                    include_suppressed=True,
                )
                stderr_raw = session.stderr_path.read_text(encoding="utf-8")
                session.close()

            self.assertEqual(hidden_tail["events"], [])
            self.assertEqual(len(visible_tail["events"]), 1)
            self.assertEqual(visible_tail["events"][0]["category"], "library_noise")
            self.assertTrue(visible_tail["events"][0]["suppressed"])
            self.assertIn("Config of the encoder", visible_tail["formatted"])
            self.assertIn('"hidden_size": 256', stderr_raw)

    def test_repeated_warning_collapses_to_single_event_with_repeat_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runtime_root = tmp / "runtime_logs"

            with mock.patch.object(review_logging, "RUNTIME_LOG_ROOT", runtime_root), mock.patch.object(
                review_logging,
                "ACTIVE_RUN_LOG_ROOT",
                runtime_root / "active",
            ), mock.patch.object(review_logging, "APP_LOG_PATH", runtime_root / "app.jsonl"):
                review_logging._APP_LOGGER_CONFIGURED = False
                session = RunLogSession(run_id="run-repeat", experiment_id="exp-1")
                with session.capture_streams():
                    print("UserWarning: repeated warning", file=__import__("sys").stderr)
                    print("  warnings.warn(", file=__import__("sys").stderr)
                    print("UserWarning: repeated warning", file=__import__("sys").stderr)
                    print("  warnings.warn(", file=__import__("sys").stderr)
                tail = read_log_tail(
                    run_id="run-repeat",
                    config={"ARTIFACT_BACKEND": "filesystem", "ARTIFACT_FILESYSTEM_ROOT": str(tmp / "artifacts")},
                    log_artifact_key="",
                    tail_lines=20,
                    min_level="DEBUG",
                    raw=False,
                    include_suppressed=True,
                )
                session.close()

            self.assertEqual(len(tail["events"]), 1)
            self.assertEqual(tail["events"][0]["repeat_count"], 2)
            self.assertIn("repeated x2", tail["formatted"])


if __name__ == "__main__":
    unittest.main()
