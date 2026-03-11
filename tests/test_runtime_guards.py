from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from runtime_guards import RuntimeStageError, first_dict_item, first_tool_call, message_content


def _load_test_modules():
    repo_root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("molnextr")
    package.__path__ = [str(repo_root / "molnextr")]
    sys.modules["molnextr"] = package

    smilespe_package = types.ModuleType("SmilesPE")
    smilespe_pretokenizer = types.ModuleType("SmilesPE.pretokenizer")
    smilespe_pretokenizer.atomwise_tokenizer = lambda value: list(value) if isinstance(value, str) else value
    sys.modules["SmilesPE"] = smilespe_package
    sys.modules["SmilesPE.pretokenizer"] = smilespe_pretokenizer

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("molnextr.constants", repo_root / "molnextr" / "constants.py")
    _load("molnextr.utils", repo_root / "molnextr" / "utils.py")
    chemistry = _load("molnextr.chemistry", repo_root / "molnextr" / "chemistry.py")

    chemietoolkit_stub = types.ModuleType("chemietoolkit")
    chemietoolkit_stub.ChemIEToolkit = object
    chemietoolkit_stub.utils = types.SimpleNamespace()
    sys.modules["chemietoolkit"] = chemietoolkit_stub

    rxnim_stub = types.ModuleType("rxnim")
    rxnim_stub.RxnIM = object
    sys.modules["rxnim"] = rxnim_stub

    molecular = _load("get_molecular_agent", repo_root / "get_molecular_agent.py")
    return chemistry, molecular


class RuntimeGuardsTests(unittest.TestCase):
    def test_message_content_raises_on_empty_choices(self) -> None:
        response = SimpleNamespace(choices=[])
        with self.assertRaises(RuntimeStageError) as ctx:
            message_content(
                response,
                context="planner",
                required=True,
                retry_trigger="auto_no_agents_retry",
            )
        self.assertEqual(ctx.exception.retry_trigger, "auto_no_agents_retry")
        self.assertIn("planner", str(ctx.exception))

    def test_first_tool_call_raises_on_empty_tool_calls(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))]
        )
        with self.assertRaises(RuntimeStageError) as ctx:
            first_tool_call(response, context="observer", retry_trigger="auto_no_agents_retry")
        self.assertEqual(ctx.exception.retry_trigger, "auto_no_agents_retry")
        self.assertIn("observer", str(ctx.exception))

    def test_first_dict_item_raises_on_empty_list(self) -> None:
        with self.assertRaises(RuntimeStageError) as ctx:
            first_dict_item([], context="reaction_results", retry_trigger="auto_recovery_retry")
        self.assertEqual(ctx.exception.retry_trigger, "auto_recovery_retry")
        self.assertIn("reaction_results", str(ctx.exception))

    def test_convert_graph_to_smiles_raises_structured_error_on_non_square_edges(self) -> None:
        chemistry, _ = _load_test_modules()
        with self.assertRaises(RuntimeStageError) as ctx:
            chemistry._convert_graph_to_smiles([[0, 0], [1, 1]], ["C", "O"], [[0, 1]])
        self.assertEqual(ctx.exception.retry_trigger, "auto_recovery_retry")
        self.assertIn("chemistry.convert_graph_to_smiles", str(ctx.exception))
        self.assertIn("edge matrix", str(ctx.exception))

    def test_apply_smiles_conversion_to_bboxes_skips_invalid_bbox_and_keeps_valid_one(self) -> None:
        _, molecular = _load_test_modules()

        def fake_conversion(coords, symbols, edges):
            if len(edges) != len(symbols):
                raise RuntimeStageError(
                    "graph mismatch",
                    context="molecular_agent_correctmultiR.graph_conversion",
                    retry_trigger="auto_recovery_retry",
                )
            return "CCO", "molblock", None

        updated, summary = molecular.apply_smiles_conversion_to_bboxes(
            [
                {
                    "bboxes": [
                        {"coords": [[0, 0]], "symbols": ["C"], "edges": [[0]], "bbox": [0, 0, 10, 10]},
                        {"coords": [[0, 0], [1, 1]], "symbols": ["C", "O"], "edges": [[0]], "bbox": [10, 10, 20, 20]},
                    ]
                }
            ],
            fake_conversion,
            context="molecular_agent_correctmultiR.graph_conversion",
        )
        self.assertEqual(summary["converted_bbox_count"], 1)
        self.assertEqual(summary["skipped_bbox_count"], 1)
        self.assertEqual(updated[0]["bboxes"][0]["smiles"], "CCO")
        self.assertNotIn("smiles", updated[0]["bboxes"][1])

    def test_apply_smiles_conversion_to_bboxes_raises_when_all_bboxes_fail(self) -> None:
        _, molecular = _load_test_modules()

        def failing_conversion(coords, symbols, edges):
            del coords, symbols, edges
            raise RuntimeStageError(
                "graph mismatch",
                context="molecular_agent_correctmultiR.graph_conversion",
                retry_trigger="auto_recovery_retry",
            )

        with self.assertRaises(RuntimeStageError) as ctx:
            molecular.apply_smiles_conversion_to_bboxes(
                [{"bboxes": [{"coords": [[0, 0]], "symbols": ["C"], "edges": [[0]], "bbox": [0, 0, 10, 10]}]}],
                failing_conversion,
                context="molecular_agent_correctmultiR.graph_conversion",
            )
        self.assertEqual(ctx.exception.retry_trigger, "auto_recovery_retry")
        self.assertTrue(hasattr(ctx.exception, "_partial_result"))
        self.assertTrue(hasattr(ctx.exception, "_conversion_summary"))


if __name__ == "__main__":
    unittest.main()
