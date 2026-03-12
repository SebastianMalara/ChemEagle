from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem

from runtime_device import warn_once

from .utils import convert_to_pil

DEFAULT_RESCUE_MODE = "off"
DEFAULT_RESCUE_CONFIDENCE = 0.85
VALID_RESCUE_MODES = {"off", "decimer"}


def resolve_smiles_rescue_mode() -> str:
    mode = (os.getenv("MOLECULE_SMILES_RESCUE") or DEFAULT_RESCUE_MODE).strip().lower()
    if mode in VALID_RESCUE_MODES:
        return mode
    warn_once(f"Unsupported MOLECULE_SMILES_RESCUE value {mode!r}. Falling back to 'off'.")
    return DEFAULT_RESCUE_MODE


def resolve_smiles_rescue_confidence() -> float:
    raw = (os.getenv("MOLECULE_SMILES_RESCUE_CONFIDENCE") or str(DEFAULT_RESCUE_CONFIDENCE)).strip()
    try:
        threshold = float(raw)
    except ValueError:
        warn_once(
            f"Invalid MOLECULE_SMILES_RESCUE_CONFIDENCE value {raw!r}. "
            f"Falling back to {DEFAULT_RESCUE_CONFIDENCE}."
        )
        return DEFAULT_RESCUE_CONFIDENCE

    if 0.0 <= threshold <= 1.0:
        return threshold

    warn_once(
        f"Out-of-range MOLECULE_SMILES_RESCUE_CONFIDENCE value {raw!r}. "
        f"Falling back to {DEFAULT_RESCUE_CONFIDENCE}."
    )
    return DEFAULT_RESCUE_CONFIDENCE


def _validated_smiles_payload(smiles: str) -> dict[str, str] | None:
    cleaned = (smiles or "").strip()
    if not cleaned:
        return None

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return {
        "smiles": Chem.MolToSmiles(mol),
        "molfile": Chem.MolToMolBlock(mol),
    }


def _looks_like_molblock(molfile: str) -> bool:
    normalized = molfile.lstrip()
    return "M  END" in normalized and ("V2000" in normalized or "V3000" in normalized)


def _prediction_is_usable(prediction: dict[str, Any]) -> bool:
    smiles_payload = _validated_smiles_payload(str(prediction.get("smiles") or ""))
    if smiles_payload is None:
        return False

    molfile = str(prediction.get("molfile") or "").strip()
    if not molfile:
        return False

    mol = Chem.MolFromMolBlock(
        molfile.lstrip(),
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        return _looks_like_molblock(molfile)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return _looks_like_molblock(molfile)
    return True


def _prediction_needs_rescue(prediction: dict[str, Any], threshold: float) -> bool:
    if not _prediction_is_usable(prediction):
        return True

    confidence = prediction.get("confidence")
    if confidence is None:
        return False

    try:
        return float(confidence) < threshold
    except (TypeError, ValueError):
        return True


def _load_decimer_predictor():
    try:
        from DECIMER import predict_SMILES as predictor  # type: ignore[import-not-found]

        return predictor
    except ImportError:
        try:
            from decimer import predict_SMILES as predictor  # type: ignore[import-not-found]

            return predictor
        except ImportError:
            warn_once(
                "DECIMER SMILES rescue was requested but DECIMER is unavailable. "
                "Install it with `pip install decimer` or disable MOLECULE_SMILES_RESCUE."
            )
            return None


def _rescue_smiles_with_decimer(image: Any) -> dict[str, str] | None:
    predictor = _load_decimer_predictor()
    if predictor is None:
        return None

    if isinstance(image, (str, os.PathLike)):
        image_path = Path(image)
        try:
            smiles = predictor(str(image_path))
        except Exception as exc:
            warn_once(f"DECIMER SMILES rescue failed: {exc}")
            return None
        if isinstance(smiles, (list, tuple)):
            smiles = smiles[0] if smiles else ""
        return _validated_smiles_payload(str(smiles or ""))

    pil_image = convert_to_pil(image)
    with tempfile.TemporaryDirectory(prefix="chemeagle_decimer_") as tmpdir:
        image_path = Path(tmpdir) / "molecule.png"
        pil_image.save(image_path)
        try:
            smiles = predictor(str(image_path))
        except Exception as exc:
            warn_once(f"DECIMER SMILES rescue failed: {exc}")
            return None

    if isinstance(smiles, (list, tuple)):
        smiles = smiles[0] if smiles else ""
    return _validated_smiles_payload(str(smiles or ""))


def apply_smiles_rescue(
    predictions: Sequence[dict[str, Any]],
    cropped_images: Sequence[Any],
) -> list[dict[str, Any]]:
    mode = resolve_smiles_rescue_mode()
    threshold = resolve_smiles_rescue_confidence()
    outputs: list[dict[str, Any]] = []

    for prediction, image in zip(predictions, cropped_images):
        updated = dict(prediction)
        updated["smiles_backend"] = "molnextr"
        if mode != "decimer" or not _prediction_needs_rescue(updated, threshold):
            outputs.append(updated)
            continue

        rescued = _rescue_smiles_with_decimer(image)
        if rescued is not None:
            updated["rescued_smiles"] = rescued["smiles"]
            updated["rescued_molfile"] = rescued["molfile"]
            updated["smiles_backend"] = "decimer"

        outputs.append(updated)

    return outputs
