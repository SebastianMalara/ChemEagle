from __future__ import annotations

import copy
import importlib
import os
import sys
import types
import unittest
from unittest import mock

from rdkit import Chem


def _valid_molfile(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToMolBlock(mol)


def _load_interface_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: types.SimpleNamespace(type=name)

    fake_layoutparser = types.ModuleType("layoutparser")
    fake_layoutparser.AutoLayoutModel = object

    fake_pdf2image = types.ModuleType("pdf2image")
    fake_pdf2image.convert_from_path = lambda *args, **kwargs: []

    fake_molnextr = types.ModuleType("molnextr")

    class DummyMolNexTR:
        def __init__(self, *args, **kwargs):
            pass

    fake_molnextr.MolNexTR = DummyMolNexTR

    fake_rxnim = types.ModuleType("rxnim")

    class DummyRxnIM:
        def __init__(self, *args, **kwargs):
            pass

    class DummyMolDetect:
        def __init__(self, *args, **kwargs):
            pass

    fake_rxnim.RxnIM = DummyRxnIM
    fake_rxnim.MolDetect = DummyMolDetect

    fake_chemiener = types.ModuleType("chemiener")

    class DummyChemNER:
        def __init__(self, *args, **kwargs):
            pass

    fake_chemiener.ChemNER = DummyChemNER

    fake_asset_registry = types.ModuleType("asset_registry")
    fake_asset_registry.ensure_asset_available = lambda asset_id: f"/tmp/{asset_id}"

    fake_runtime_device = types.ModuleType("runtime_device")
    fake_runtime_device.resolve_torch_device = lambda: types.SimpleNamespace(type="cpu")
    fake_runtime_device.warn_once = lambda message: None

    fake_utils = types.ModuleType("chemietoolkit.utils")
    fake_utils.convert_to_pil = lambda figure: figure
    fake_utils.convert_to_cv2 = lambda figure: figure
    fake_utils.clean_bbox_output = lambda figures, bboxes: ([], [], [])

    fake_chemrxnextractor = types.ModuleType("chemietoolkit.chemrxnextractor")

    class DummyChemRxnExtractor:
        def __init__(self, *args, **kwargs):
            pass

    fake_chemrxnextractor.ChemRxnExtractor = DummyChemRxnExtractor

    fake_tableextractor = types.ModuleType("chemietoolkit.tableextractor")

    class DummyTableExtractor:
        def __init__(self, *args, **kwargs):
            pass

    fake_tableextractor.TableExtractor = DummyTableExtractor

    with mock.patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "layoutparser": fake_layoutparser,
            "pdf2image": fake_pdf2image,
            "molnextr": fake_molnextr,
            "rxnim": fake_rxnim,
            "chemiener": fake_chemiener,
            "asset_registry": fake_asset_registry,
            "runtime_device": fake_runtime_device,
            "chemietoolkit.utils": fake_utils,
            "chemietoolkit.chemrxnextractor": fake_chemrxnextractor,
            "chemietoolkit.tableextractor": fake_tableextractor,
        },
        clear=False,
    ):
        for name in (
            "chemietoolkit.interface",
            "chemietoolkit",
        ):
            sys.modules.pop(name, None)
        return importlib.import_module("chemietoolkit.interface")


class RecordingMolNexTR:
    def __init__(self, outputs):
        self.outputs = [copy.deepcopy(item) for item in outputs]
        self.calls = []

    def predict_images(self, images, **kwargs):
        self.calls.append({"images": list(images), **kwargs})
        return [copy.deepcopy(item) for item in self.outputs]


class DecimerSmilesRescueTests(unittest.TestCase):
    def _run_extract(
        self,
        *,
        molnextr_output,
        rescue_mode: str,
        rescue_smiles: str = "CCN",
    ):
        module = _load_interface_module()
        toolkit = module.ChemIEToolkit(device="cpu")
        toolkit.extract_molecule_bboxes_from_figures = mock.Mock(return_value=[{"bbox": (0, 0, 1, 1)}])

        molecule_ref = {
            "bbox": (0, 0, 1, 1),
            "score": 0.99,
            "image": "crop-1",
        }
        results = [{"image": "figure-1", "molecules": [molecule_ref]}]
        refs = [molecule_ref]
        module.convert_to_cv2 = lambda figure: figure
        module.clean_bbox_output = mock.Mock(return_value=(results, ["crop-1"], refs))

        toolkit._molnextr = RecordingMolNexTR([molnextr_output])

        decimer_module = types.ModuleType("decimer")
        decimer_module.predict_SMILES = mock.Mock(return_value=rescue_smiles)

        with mock.patch.dict(
            os.environ,
            {
                "MOLECULE_SMILES_RESCUE": rescue_mode,
                "MOLECULE_SMILES_RESCUE_CONFIDENCE": "0.85",
            },
            clear=False,
        ), mock.patch.dict(sys.modules, {"decimer": decimer_module}, clear=False):
            output = toolkit.extract_molecules_from_figures(["figure-1"], batch_size=4)

        return output, toolkit._molnextr.calls, decimer_module.predict_SMILES

    def test_valid_molnextr_output_skips_rescue(self) -> None:
        output, calls, decimer_predict = self._run_extract(
            molnextr_output={
                "smiles": "CCO",
                "molfile": _valid_molfile("CCO"),
                "symbols": ["C"],
                "coords": [(0.0, 0.0)],
                "edges": [[0]],
                "confidence": 0.97,
            },
            rescue_mode="decimer",
        )

        molecule = output[0]["molecules"][0]
        self.assertEqual(molecule["smiles_backend"], "molnextr")
        self.assertNotIn("rescued_smiles", molecule)
        self.assertTrue(calls)
        self.assertTrue(calls[0].get("return_confidence"))
        decimer_predict.assert_not_called()

    def test_empty_smiles_triggers_rescue(self) -> None:
        output, _, decimer_predict = self._run_extract(
            molnextr_output={
                "smiles": "",
                "molfile": "",
                "symbols": ["C"],
                "coords": [(0.0, 0.0)],
                "edges": [[0]],
                "confidence": 0.99,
            },
            rescue_mode="decimer",
            rescue_smiles="CCN",
        )

        molecule = output[0]["molecules"][0]
        self.assertEqual(molecule["smiles_backend"], "decimer")
        self.assertEqual(molecule["rescued_smiles"], "CCN")
        self.assertEqual(molecule["rescued_molfile"], _valid_molfile("CCN"))
        decimer_predict.assert_called_once_with("crop-1")

    def test_invalid_molfile_triggers_rescue(self) -> None:
        output, _, decimer_predict = self._run_extract(
            molnextr_output={
                "smiles": "CCO",
                "molfile": "not-a-molfile",
                "symbols": ["C"],
                "coords": [(0.0, 0.0)],
                "edges": [[0]],
                "confidence": 0.99,
            },
            rescue_mode="decimer",
            rescue_smiles="CCC",
        )

        molecule = output[0]["molecules"][0]
        self.assertEqual(molecule["smiles_backend"], "decimer")
        self.assertEqual(molecule["rescued_smiles"], "CCC")
        decimer_predict.assert_called_once_with("crop-1")

    def test_low_confidence_triggers_rescue(self) -> None:
        output, _, decimer_predict = self._run_extract(
            molnextr_output={
                "smiles": "CCO",
                "molfile": _valid_molfile("CCO"),
                "symbols": ["C"],
                "coords": [(0.0, 0.0)],
                "edges": [[0]],
                "confidence": 0.2,
            },
            rescue_mode="decimer",
            rescue_smiles="CCCl",
        )

        molecule = output[0]["molecules"][0]
        self.assertEqual(molecule["smiles_backend"], "decimer")
        self.assertEqual(molecule["rescued_smiles"], "CCCl")
        decimer_predict.assert_called_once_with("crop-1")

    def test_invalid_decimer_preserves_original_result(self) -> None:
        output, _, decimer_predict = self._run_extract(
            molnextr_output={
                "smiles": "CCO",
                "molfile": "broken",
                "symbols": ["C"],
                "coords": [(0.0, 0.0)],
                "edges": [[0]],
                "confidence": 0.1,
            },
            rescue_mode="decimer",
            rescue_smiles="C1",
        )

        molecule = output[0]["molecules"][0]
        self.assertEqual(molecule["smiles_backend"], "molnextr")
        self.assertNotIn("rescued_smiles", molecule)
        self.assertEqual(molecule["smiles"], "CCO")
        decimer_predict.assert_called_once_with("crop-1")

    def test_rescue_disabled_skips_fallback(self) -> None:
        output, _, decimer_predict = self._run_extract(
            molnextr_output={
                "smiles": "",
                "molfile": "",
                "symbols": ["C"],
                "coords": [(0.0, 0.0)],
                "edges": [[0]],
                "confidence": 0.0,
            },
            rescue_mode="off",
        )

        molecule = output[0]["molecules"][0]
        self.assertEqual(molecule["smiles_backend"], "molnextr")
        self.assertNotIn("rescued_smiles", molecule)
        decimer_predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
