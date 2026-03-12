from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.pop(module_name, None)
    spec.loader.exec_module(module)
    return module


class HfRuntimeTests(unittest.TestCase):
    def test_hf_runtime_sets_transformers_env_guards(self) -> None:
        path = Path(__file__).resolve().parents[1] / "hf_runtime.py"
        old_env = {key: os.environ.get(key) for key in ("USE_TF", "USE_TORCH", "USE_FLAX", "TRANSFORMERS_NO_TF")}
        try:
            for key in old_env:
                os.environ.pop(key, None)
            _load_module("hf_runtime_under_test", path)
            self.assertEqual(os.environ.get("USE_TF"), "0")
            self.assertEqual(os.environ.get("USE_TORCH"), "1")
            self.assertEqual(os.environ.get("USE_FLAX"), "0")
            self.assertEqual(os.environ.get("TRANSFORMERS_NO_TF"), "1")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_transformers_import_sites_configure_runtime_before_import(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        files = [
            repo_root / "main.py",
            repo_root / "pdfmodel" / "methods.py",
            repo_root / "chemiener" / "model.py",
            repo_root / "chemiener" / "dataset.py",
            repo_root / "chemiener" / "main.py",
            repo_root / "rxnim" / "pix2seq" / "transformer.py",
            repo_root / "rxnim" / "pix2seq" / "pix2seq.py",
        ]

        for path in files:
            source = path.read_text(encoding="utf-8")
            self.assertIn("from hf_runtime import configure_transformers_runtime", source, msg=str(path))
            self.assertIn("configure_transformers_runtime()", source, msg=str(path))
            if "from transformers" in source or "import transformers" in source:
                transformer_import_index = (
                    source.index("from transformers")
                    if "from transformers" in source
                    else source.index("import transformers")
                )
                self.assertLess(
                    source.index("configure_transformers_runtime()"),
                    transformer_import_index,
                    msg=str(path),
                )

    def test_sitecustomize_configures_huggingface_runtime(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "sitecustomize.py").read_text(encoding="utf-8")
        self.assertIn("from hf_runtime import configure_transformers_runtime", source)
        self.assertIn("configure_transformers_runtime()", source)


if __name__ == "__main__":
    unittest.main()
