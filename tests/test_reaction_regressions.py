from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _DummyLLMWrapper:
    @classmethod
    def from_env(cls, default_model=None):  # noqa: ANN206
        del default_model
        return SimpleNamespace(as_chat_completion_client=lambda: None)


def _base_stub_modules() -> dict[str, types.ModuleType]:
    fake_torch = _module("torch", device=lambda name: SimpleNamespace(type=name))
    fake_cv2 = _module("cv2")
    fake_openai = _module("openai", OpenAI=object)
    fake_asset_registry = _module(
        "asset_registry",
        ensure_asset_available=lambda asset_id: REPO_ROOT / "tests" / asset_id,
    )
    fake_runtime_device = _module(
        "runtime_device",
        resolve_torch_device=lambda: SimpleNamespace(type="cpu"),
    )
    fake_llm_wrapper = _module("llm_wrapper", LLMWrapper=_DummyLLMWrapper)
    fake_rxnim = _module("rxnim", RxnIM=object, MolDetect=object)
    fake_molnextr = _module("molnextr")
    fake_molnextr.__path__ = [str(REPO_ROOT / "molnextr")]
    fake_molnextr_chemistry = _module(
        "molnextr.chemistry",
        _convert_graph_to_smiles=lambda *args, **kwargs: ("", "", None),
    )
    fake_chemietoolkit = _module(
        "chemietoolkit",
        ChemIEToolkit=object,
        utils=SimpleNamespace(backout_without_coref=lambda *args, **kwargs: []),
    )
    fake_get_molecular_agent = _module(
        "get_molecular_agent",
        process_reaction_image_with_multiple_products_and_text_correctR=lambda image_path: [],
        process_reaction_image_with_multiple_products_and_text_correctmultiR=lambda image_path: [],
        process_reaction_image_with_multiple_products_and_text_correctmultiR_OS=lambda image_path: [],
    )
    fake_get_reaction_agent = _module(
        "get_reaction_agent",
        get_reaction_withatoms_correctR=lambda image_path: [],
        get_reaction_withatoms_correctR_OS=lambda image_path: [],
    )
    return {
        "torch": fake_torch,
        "cv2": fake_cv2,
        "openai": fake_openai,
        "asset_registry": fake_asset_registry,
        "runtime_device": fake_runtime_device,
        "llm_wrapper": fake_llm_wrapper,
        "rxnim": fake_rxnim,
        "molnextr": fake_molnextr,
        "molnextr.chemistry": fake_molnextr_chemistry,
        "chemietoolkit": fake_chemietoolkit,
        "get_molecular_agent": fake_get_molecular_agent,
        "get_reaction_agent": fake_get_reaction_agent,
    }


def _load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    with mock.patch.dict(sys.modules, _base_stub_modules(), clear=False):
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _response(message) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"call-{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeChatCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class ReactionRegressionTests(unittest.TestCase):
    def test_get_reaction_returns_empty_sections_when_rxnim_returns_empty_list(self) -> None:
        module = _load_module("_test_get_reaction_agent", "get_reaction_agent.py")

        rxnim = SimpleNamespace(predict_image_file=mock.Mock(return_value=[]))

        with mock.patch.object(module.Image, "open", return_value=object()), mock.patch.object(
            module, "_get_rxnim", return_value=rxnim
        ), mock.patch("builtins.print") as print_mock:
            result = module.get_reaction("fake.png")

        self.assertEqual(
            result,
            {"reactants": [], "conditions": [], "products": []},
        )
        print_mock.assert_called_once_with("warning: reaction_agent.predict_image_file: empty result list")
        rxnim.predict_image_file.assert_called_once_with("fake.png", molnextr=True, ocr=True)

    def test_process_product_variant_handles_normalized_reaction_wrapper(self) -> None:
        module = _load_module("_test_get_r_group_sub_agent", "get_R_group_sub_agent.py")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=_FakeChatCompletions(
                    [
                        _response(
                            SimpleNamespace(
                                role="assistant",
                                content="",
                                tool_calls=[
                                    _tool_call(
                                        "get_reaction",
                                        '{"image_path": "fake.png"}',
                                    )
                                ],
                            )
                        ),
                        _response(
                            SimpleNamespace(
                                role="assistant",
                                content='{"P1": ["product template"]}',
                            )
                        ),
                    ]
                )
            )
        )

        coref_results = [
            {
                "bboxes": [
                    {
                        "smiles": "P1",
                        "bbox": [0, 0, 1, 1],
                        "category": "product",
                        "category_id": 1,
                        "score": 0.99,
                        "molfile": "molfile",
                        "atoms": [],
                        "bonds": [],
                        "symbols": [],
                        "coords": [],
                        "edges": [],
                    }
                ]
            }
        ]

        normalize_output = mock.Mock(side_effect=lambda data: data)
        toolkit = SimpleNamespace(molnextr=object())

        def fake_open(path, mode="r", *args, **kwargs):
            del args, kwargs
            if path == "fake.png" and "b" in mode:
                return io.BytesIO(b"fake-image")
            if path == "./prompt/prompt_Str_R.txt" and "r" in mode:
                return io.StringIO("prompt")
            raise AssertionError(f"unexpected open call: {path!r} ({mode!r})")

        with mock.patch("builtins.open", side_effect=fake_open), mock.patch.object(
            module, "_get_azure_client", return_value=fake_client
        ), mock.patch.object(
            module.Image,
            "open",
            return_value=SimpleNamespace(convert=lambda mode: np.zeros((1, 1, 3), dtype=np.uint8)),
        ), mock.patch.object(
            module.Image,
            "fromarray",
            return_value=SimpleNamespace(save=lambda buffer, format: buffer.write(b"png")),
        ), mock.patch.object(
            module, "get_reaction", return_value={"reactants": [], "conditions": [], "products": []}
        ), mock.patch.object(
            module, "get_cached_multi_molecular", return_value=coref_results
        ), mock.patch.object(
            module, "draw_mol_bboxes", return_value=np.zeros((1, 1, 3), dtype=np.uint8)
        ), mock.patch.object(
            module,
            "get_cached_raw_results",
            return_value=[
                {
                    "reactants": [{"smiles": "R1"}],
                    "conditions": [],
                    "products": [{"smiles": "P1"}],
                }
            ],
        ), mock.patch.object(
            module, "_get_runtime_models", return_value=(toolkit, object())
        ), mock.patch.object(
            module.utils,
            "backout_without_coref",
            return_value=[(["R1"], ["P1"], 1)],
        ), mock.patch.object(
            module, "parse_coref_data_with_fallback", return_value={"parsed": True}
        ), mock.patch.object(
            module, "normalize_product_variant_output", normalize_output
        ):
            result = module.process_reaction_image_with_product_variant_R_group("fake.png")

        normalize_output.assert_called_once()
        payload = normalize_output.call_args.args[0]
        self.assertEqual(
            payload["reaction_template"],
            {"reactants": ["R1"], "products": ["P1"]},
        )
        self.assertEqual(payload["reactions"], {1: {"reactants": ["R1"], "products": ["P1"]}})
        self.assertEqual(payload["original_molecule_list"], {"P1": ["product template"]})
        self.assertEqual(result, payload)


if __name__ == "__main__":
    unittest.main()
