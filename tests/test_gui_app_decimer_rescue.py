from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from webapp.api import runtime as api_runtime


class GuiAppDecimerRescueTests(unittest.TestCase):
    def test_gui_source_contains_one_decimer_rescue_control_block(self) -> None:
        source = Path(__file__).resolve().parents[1].joinpath("gui_app.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('label="MOLECULE_SMILES_RESCUE"'), 1)
        self.assertEqual(source.count('label="MOLECULE_SMILES_RESCUE_CONFIDENCE"'), 1)

    def test_build_runtime_values_includes_rescue_fields(self) -> None:
        values = api_runtime.build_runtime_values(
            mode="cloud",
            llm_provider="openai",
            llm_model="gpt-5-mini",
            api_key="",
            azure_endpoint="",
            api_version="",
            openai_api_key="sk-test",
            openai_base_url="",
            anthropic_api_key="",
            vllm_base_url="",
            vllm_api_key="",
            chemeagle_device="auto",
            ocr_backend="easyocr",
            ocr_llm_inherit_main=True,
            ocr_llm_provider="openai",
            ocr_llm_model="gpt-5-mini",
            ocr_api_key="",
            ocr_azure_endpoint="",
            ocr_api_version="",
            ocr_openai_api_key="",
            ocr_openai_base_url="",
            ocr_anthropic_api_key="",
            ocr_vllm_base_url="",
            ocr_vllm_api_key="",
            ocr_lang="eng",
            ocr_config="",
            tesseract_cmd="",
            molecule_smiles_rescue="decimer",
            molecule_smiles_rescue_confidence="0.72",
            pdf_model_size="large",
            pdf_persist_images=False,
            pdf_persist_dir="",
        )
        self.assertEqual(values["MOLECULE_SMILES_RESCUE"], "decimer")
        self.assertEqual(values["MOLECULE_SMILES_RESCUE_CONFIDENCE"], "0.72")

    def test_save_and_merge_env_round_trip_preserves_rescue_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.chemeagle"
            message = api_runtime.save_env_file(
                env_path,
                {
                    "MOLECULE_SMILES_RESCUE": "decimer",
                    "MOLECULE_SMILES_RESCUE_CONFIDENCE": "0.91",
                },
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                merged = api_runtime.merged_env_values(env_path)

        self.assertIn("Saved", message)
        self.assertEqual(merged["MOLECULE_SMILES_RESCUE"], "decimer")
        self.assertEqual(merged["MOLECULE_SMILES_RESCUE_CONFIDENCE"], "0.91")

    def test_molecule_smiles_rescue_preflight_blocks_missing_decimer_import(self) -> None:
        with mock.patch.object(
            api_runtime,
            "probe_python_code",
            return_value={"ok": False, "stdout": "", "stderr": "No module named decimer"},
        ):
            report = api_runtime.molecule_smiles_rescue_preflight(
                {
                    "MOLECULE_SMILES_RESCUE": "decimer",
                    "MOLECULE_SMILES_RESCUE_CONFIDENCE": "0.85",
                }
            )

        self.assertIn("DECIMER failed to import in", "\n".join(report["blocking_errors"]))
        self.assertEqual(report["checks"]["requested_rescue"], "decimer")
        self.assertEqual(report["checks"]["confidence_threshold"], 0.85)
        self.assertFalse(report["checks"]["python_import_probe"]["ok"])
        self.assertTrue(report["checks"]["python_executable"])
        self.assertTrue(any("direct molecule extraction" in warning for warning in report["warnings"]))

    def test_molecule_smiles_rescue_preflight_accepts_fast_module_probe(self) -> None:
        with mock.patch.object(
            api_runtime,
            "probe_python_code",
            return_value={
                "ok": True,
                "stdout": '{"DECIMER": "/tmp/site-packages/DECIMER/__init__.py", "decimer": null}',
                "stderr": "",
            },
        ):
            report = api_runtime.molecule_smiles_rescue_preflight(
                {
                    "MOLECULE_SMILES_RESCUE": "decimer",
                    "MOLECULE_SMILES_RESCUE_CONFIDENCE": "0.85",
                }
            )

        self.assertEqual(report["blocking_errors"], [])
        self.assertIn("DECIMER", report["checks"]["module_probe"])

    def test_molecule_smiles_rescue_preflight_rejects_bad_confidence(self) -> None:
        report = api_runtime.molecule_smiles_rescue_preflight(
            {
                "MOLECULE_SMILES_RESCUE": "off",
                "MOLECULE_SMILES_RESCUE_CONFIDENCE": "not-a-number",
            }
        )
        self.assertIn("must be a float between 0 and 1", "\n".join(report["blocking_errors"]))

    def test_collect_preflight_diagnostics_blocks_when_torch_probe_fails_for_local_path(self) -> None:
        def fake_probe(code: str, env):  # noqa: ANN001
            del env
            if "import torch" in code:
                return {
                    "ok": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "OMP: Error #179: Function Can't open SHM2 failed",
                }
            return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}

        with mock.patch.object(api_runtime, "probe_python_code", side_effect=fake_probe), mock.patch.object(
            api_runtime,
            "profile_preflight",
            return_value={"blocking_errors": [], "warnings": [], "checks": {}},
        ), mock.patch.object(
            api_runtime,
            "model_catalog_preflight",
            return_value={"blocking_errors": [], "warnings": [], "checks": {}},
        ), mock.patch.object(
            api_runtime,
            "asset_preflight",
            return_value={"blocking_errors": [], "warnings": [], "checks": {}},
        ), mock.patch(
            "webapp.api.runtime.collect_runtime_provider_preflight",
            return_value={"blocking_errors": [], "warnings": [], "checks": {}, "status": "ok"},
        ):
            diagnostics = api_runtime.collect_preflight_diagnostics(
                "",
                "local_os",
                {
                    "CHEMEAGLE_DEVICE": "auto",
                    "OCR_BACKEND": "easyocr",
                    "MOLECULE_SMILES_RESCUE": "off",
                    "MOLECULE_SMILES_RESCUE_CONFIDENCE": "0.85",
                },
                include_pdf_section=False,
            )

        section = diagnostics["torch_runtime_preflight"]
        self.assertTrue(section["checks"]["required_for_current_run"])
        self.assertEqual(
            section["checks"]["python_import_probe"]["stderr"],
            "OMP: Error #179: Function Can't open SHM2 failed",
        )
        self.assertTrue(
            any("torch_runtime_preflight: PyTorch failed to import" in item for item in diagnostics["blocking_errors"])
        )

    def test_probe_python_code_returns_failure_on_timeout(self) -> None:
        with mock.patch.dict(os.environ, {"CHEMEAGLE_PREFLIGHT_PROBE_TIMEOUT_SECONDS": "3"}, clear=False):
            with mock.patch.object(
                api_runtime.subprocess,
                "run",
                side_effect=api_runtime.subprocess.TimeoutExpired(cmd=["python"], timeout=3, output="", stderr="hung import"),
            ):
                result = api_runtime.probe_python_code("import torch", {})

        self.assertFalse(result["ok"])
        self.assertIsNone(result["returncode"])
        self.assertIn("Probe timed out after 3.0s.", result["stderr"])


if __name__ == "__main__":
    unittest.main()
