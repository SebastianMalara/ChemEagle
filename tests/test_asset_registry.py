from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import asset_registry


class AssetRegistryTests(unittest.TestCase):
    def test_cloud_image_base_preflight_requires_image_core_only(self) -> None:
        report = asset_registry.build_asset_preflight_report(
            mode="cloud",
            ocr_backend="llm_vision",
            file_kind="image",
        )
        items = {item["asset_id"]: item for item in report["assets"]}

        self.assertTrue(items["rxn_ckpt"]["blocking"])
        self.assertTrue(items["molnextr_ckpt"]["blocking"])
        self.assertFalse(items["ner_ckpt"]["required_for_current_run"])
        self.assertFalse(items["chemrxnextractor_models"]["required_for_current_run"])
        self.assertFalse(items["visualheist_large"]["required_for_current_run"])

    def test_text_tool_assets_are_optional_even_when_selected(self) -> None:
        report = asset_registry.build_asset_preflight_report(
            mode="cloud",
            ocr_backend="llm_vision",
            file_kind="image",
            selected_tools=["text_extraction_agent"],
        )
        items = {item["asset_id"]: item for item in report["assets"]}

        self.assertTrue(items["ner_ckpt"]["required_for_current_run"])
        self.assertFalse(items["ner_ckpt"]["blocking"])
        self.assertTrue(items["chemrxnextractor_models"]["required_for_current_run"])
        self.assertFalse(items["chemrxnextractor_models"]["blocking"])

    def test_pdf_model_selection_only_requires_requested_visualheist_weights(self) -> None:
        report = asset_registry.build_asset_preflight_report(
            mode="cloud",
            ocr_backend="llm_vision",
            file_kind="pdf",
            pdf_model_size="base",
        )
        items = {item["asset_id"]: item for item in report["assets"]}

        self.assertTrue(items["visualheist_base"]["blocking"])
        self.assertFalse(items["visualheist_large"]["required_for_current_run"])

    def test_check_asset_status_prefers_bundle_then_legacy_then_cache(self) -> None:
        with tempfile.TemporaryDirectory() as asset_root_tmp, tempfile.TemporaryDirectory() as repo_root_tmp, tempfile.TemporaryDirectory() as cache_tmp:
            asset_root = Path(asset_root_tmp)
            repo_root = Path(repo_root_tmp)
            cache_path = Path(cache_tmp) / "molnextr.pth"
            cache_path.write_text("cache", encoding="utf-8")

            spec = asset_registry.get_asset_spec("molnextr_ckpt")
            expected = asset_root / spec.relative_path
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_text("bundle", encoding="utf-8")

            with mock.patch.object(asset_registry, "REPO_ROOT", repo_root):
                status = asset_registry.check_asset_status("molnextr_ckpt", asset_root=asset_root)
                self.assertEqual(status.resolved_from, "asset_root")

                expected.unlink()
                legacy = repo_root / "molnextr.pth"
                legacy.write_text("legacy", encoding="utf-8")
                status = asset_registry.check_asset_status("molnextr_ckpt", asset_root=asset_root)
                self.assertEqual(status.resolved_from, "legacy_path")

                legacy.unlink()
                with mock.patch.object(asset_registry, "_resolve_from_hf_cache", return_value=cache_path):
                    status = asset_registry.check_asset_status("molnextr_ckpt", asset_root=asset_root)
                    self.assertEqual(status.resolved_from, "huggingface_cache")

    def test_missing_text_assets_raise_warnings_not_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as asset_root_tmp:
            report = asset_registry.build_asset_preflight_report(
                mode="cloud",
                ocr_backend="llm_vision",
                file_kind="image",
                selected_tools=["text_extraction_agent"],
                asset_root=Path(asset_root_tmp),
            )

        items = {item["asset_id"]: item for item in report["assets"]}
        self.assertFalse(items["ner_ckpt"]["blocking"])
        self.assertFalse(items["chemrxnextractor_models"]["blocking"])
        warning_text = "\n".join(report["warnings"])
        self.assertIn("chemrxnextractor_models", warning_text)
        self.assertTrue(all("chemrxnextractor_models" not in issue for issue in report["blocking_errors"]))

    def test_install_assets_materializes_legacy_file_into_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as asset_root_tmp, tempfile.TemporaryDirectory() as repo_root_tmp:
            asset_root = Path(asset_root_tmp)
            repo_root = Path(repo_root_tmp)
            legacy = repo_root / "rxn.ckpt"
            legacy.write_text("legacy", encoding="utf-8")

            with mock.patch.object(asset_registry, "REPO_ROOT", repo_root):
                results = asset_registry.install_assets(["rxn_ckpt"], asset_root=asset_root, dry_run=False)

            self.assertEqual(results[0]["action"], "materialized")
            target = asset_root / "models" / "rxn.ckpt"
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "legacy")


if __name__ == "__main__":
    unittest.main()
