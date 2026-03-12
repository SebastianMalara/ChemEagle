from __future__ import annotations

import builtins
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_runtime_device_without_torch():
    path = Path(__file__).resolve().parents[1] / "runtime_device.py"
    spec = importlib.util.spec_from_file_location("runtime_device_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "torch":
            raise AssertionError("runtime_device imported torch at module scope")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded_import):
        spec.loader.exec_module(module)
    return module


class RuntimeDeviceImportTests(unittest.TestCase):
    def test_module_import_does_not_require_torch(self) -> None:
        sys.modules.pop("runtime_device_under_test", None)
        module = _load_runtime_device_without_torch()

        self.assertEqual(module.resolve_ocr_backend("auto", "cloud"), "llm_vision")
        self.assertEqual(module.resolve_ocr_backend("auto", "local_os"), "easyocr")
        self.assertEqual(module.resolve_ocr_backend("vision", "local_os"), "llm_vision")

    def test_easyocr_acceleration_checks_device_type_only(self) -> None:
        sys.modules.pop("runtime_device_under_test", None)
        module = _load_runtime_device_without_torch()

        self.assertTrue(module.easyocr_uses_acceleration(types.SimpleNamespace(type="cuda")))
        self.assertTrue(module.easyocr_uses_acceleration(types.SimpleNamespace(type="mps")))
        self.assertFalse(module.easyocr_uses_acceleration(types.SimpleNamespace(type="cpu")))
        self.assertFalse(module.easyocr_uses_acceleration(object()))


if __name__ == "__main__":
    unittest.main()
