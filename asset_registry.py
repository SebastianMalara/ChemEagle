from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets"
ASSET_ENV_VAR = "CHEMEAGLE_ASSET_ROOT"
ASSET_MANIFEST_NAME = "asset-manifest.json"

MODEL_REPO = "CYF200127/ChemEAGLEModel"
CRE_REPO = "amberwang/chemrxnextractor-training-modules"
VISUALHEIST_BASE_REPO = "shixuanleong/visualheist-base"
VISUALHEIST_LARGE_REPO = "shixuanleong/visualheist-large"

AssetSourceType = Literal["hf_file", "hf_snapshot"]
ResolvedFrom = Literal["asset_root", "legacy_path", "huggingface_cache", "missing"]


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    relative_path: str
    source_type: AssetSourceType
    repo_id: str
    filename: str | None = None
    allow_patterns: tuple[str, ...] = ()
    feature_tags: tuple[str, ...] = ()
    runtime_tags: tuple[str, ...] = ()
    legacy_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssetStatus:
    asset_id: str
    expected_path: Path
    present: bool
    resolved_path: Path | None
    resolved_from: ResolvedFrom
    diagnostic: str
    in_bundle: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "expected_path": str(self.expected_path),
            "present": self.present,
            "resolved_path": str(self.resolved_path) if self.resolved_path else "",
            "resolved_from": self.resolved_from,
            "diagnostic": self.diagnostic,
            "in_bundle": self.in_bundle,
        }


@dataclass(frozen=True)
class RequirementInfo:
    asset_id: str
    required_for_current_run: bool
    blocking: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "required_for_current_run": self.required_for_current_run,
            "blocking": self.blocking,
            "reason": self.reason,
        }


class AssetNotAvailableError(RuntimeError):
    def __init__(self, asset_id: str, status: AssetStatus):
        self.asset_id = asset_id
        self.status = status
        super().__init__(f"{asset_id} is unavailable: {status.diagnostic}")


ASSET_SPECS: dict[str, AssetSpec] = {
    "rxn_ckpt": AssetSpec(
        asset_id="rxn_ckpt",
        relative_path="models/rxn.ckpt",
        source_type="hf_file",
        repo_id=MODEL_REPO,
        filename="rxn.ckpt",
        feature_tags=("image_core", "cloud_runtime", "os_runtime"),
        legacy_paths=("rxn.ckpt", "models/rxn.ckpt"),
    ),
    "ner_ckpt": AssetSpec(
        asset_id="ner_ckpt",
        relative_path="models/ner.ckpt",
        source_type="hf_file",
        repo_id=MODEL_REPO,
        filename="ner.ckpt",
        feature_tags=("chemner", "text_rxn"),
        legacy_paths=("ner.ckpt", "models/ner.ckpt"),
    ),
    "molnextr_ckpt": AssetSpec(
        asset_id="molnextr_ckpt",
        relative_path="models/molnextr.pth",
        source_type="hf_file",
        repo_id=MODEL_REPO,
        filename="molnextr.pth",
        feature_tags=("image_core", "cloud_runtime", "os_runtime"),
        legacy_paths=("molnextr.pth", "models/molnextr.pth"),
    ),
    "moldet_ckpt": AssetSpec(
        asset_id="moldet_ckpt",
        relative_path="models/moldet.ckpt",
        source_type="hf_file",
        repo_id=MODEL_REPO,
        filename="moldet.ckpt",
        feature_tags=("image_core", "cloud_runtime", "os_runtime"),
        legacy_paths=("moldet.ckpt", "models/moldet.ckpt"),
    ),
    "corefdet_ckpt": AssetSpec(
        asset_id="corefdet_ckpt",
        relative_path="models/corefdet.ckpt",
        source_type="hf_file",
        repo_id=MODEL_REPO,
        filename="corefdet.ckpt",
        feature_tags=("image_core", "cloud_runtime", "os_runtime"),
        legacy_paths=("corefdet.ckpt", "models/corefdet.ckpt"),
    ),
    "chemrxnextractor_models": AssetSpec(
        asset_id="chemrxnextractor_models",
        relative_path="cre_models_v0.1",
        source_type="hf_snapshot",
        repo_id=CRE_REPO,
        allow_patterns=("cre_models_v0.1/*", "cre_models_v0.1/**"),
        feature_tags=("text_rxn",),
        legacy_paths=("cre_models_v0.1",),
    ),
    "visualheist_base": AssetSpec(
        asset_id="visualheist_base",
        relative_path="safetensors/base_model.safetensors",
        source_type="hf_file",
        repo_id=VISUALHEIST_BASE_REPO,
        filename="model.safetensors",
        feature_tags=("pdf",),
        legacy_paths=("safetensors/base_model.safetensors",),
    ),
    "visualheist_large": AssetSpec(
        asset_id="visualheist_large",
        relative_path="safetensors/large_model.safetensors",
        source_type="hf_file",
        repo_id=VISUALHEIST_LARGE_REPO,
        filename="model.safetensors",
        feature_tags=("pdf",),
        legacy_paths=("safetensors/large_model.safetensors",),
    ),
}

IMAGE_CORE_ASSET_IDS = ("rxn_ckpt", "molnextr_ckpt", "moldet_ckpt", "corefdet_ckpt")
TEXT_TOOL_ASSET_IDS = ("ner_ckpt", "chemrxnextractor_models")
ALL_ASSET_IDS = tuple(ASSET_SPECS.keys())


def get_asset_root() -> Path:
    raw_root = os.getenv(ASSET_ENV_VAR, "").strip()
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return DEFAULT_ASSET_ROOT.resolve()


def get_asset_spec(asset_id: str) -> AssetSpec:
    try:
        return ASSET_SPECS[asset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown asset id: {asset_id}") from exc


def iter_asset_specs(asset_ids: Iterable[str] | None = None) -> Iterable[AssetSpec]:
    ids = asset_ids or ALL_ASSET_IDS
    for asset_id in ids:
        yield get_asset_spec(asset_id)


def resolve_asset_path(asset_id: str, *, asset_root: Path | None = None) -> Path:
    spec = get_asset_spec(asset_id)
    root = asset_root or get_asset_root()
    return (root / spec.relative_path).resolve()


def _legacy_candidate_paths(spec: AssetSpec) -> list[Path]:
    return [(REPO_ROOT / rel).resolve() for rel in spec.legacy_paths]


def _resolve_from_hf_cache(spec: AssetSpec) -> Path | None:
    if spec.source_type == "hf_file":
        if not spec.filename:
            return None
        cached = try_to_load_from_cache(spec.repo_id, spec.filename)
        if isinstance(cached, str):
            cached_path = Path(cached)
            if cached_path.exists():
                return cached_path.resolve()
        return None

    try:
        cached_root = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                local_files_only=True,
                allow_patterns=list(spec.allow_patterns) or None,
            )
        )
    except Exception:
        return None

    candidate = cached_root / Path(spec.relative_path).name
    if candidate.exists():
        return candidate.resolve()
    return None


def check_asset_status(asset_id: str, *, asset_root: Path | None = None) -> AssetStatus:
    spec = get_asset_spec(asset_id)
    expected_path = resolve_asset_path(asset_id, asset_root=asset_root)

    if expected_path.exists():
        return AssetStatus(
            asset_id=asset_id,
            expected_path=expected_path,
            present=True,
            resolved_path=expected_path,
            resolved_from="asset_root",
            diagnostic="Asset is present in the canonical bundle.",
            in_bundle=True,
        )

    for candidate in _legacy_candidate_paths(spec):
        if candidate.exists():
            return AssetStatus(
                asset_id=asset_id,
                expected_path=expected_path,
                present=True,
                resolved_path=candidate,
                resolved_from="legacy_path",
                diagnostic=f"Asset resolved from legacy path: {candidate}",
                in_bundle=False,
            )

    cached = _resolve_from_hf_cache(spec)
    if cached is not None:
        return AssetStatus(
            asset_id=asset_id,
            expected_path=expected_path,
            present=True,
            resolved_path=cached,
            resolved_from="huggingface_cache",
            diagnostic="Asset resolved from Hugging Face cache.",
            in_bundle=False,
        )

    return AssetStatus(
        asset_id=asset_id,
        expected_path=expected_path,
        present=False,
        resolved_path=None,
        resolved_from="missing",
        diagnostic=f"Missing asset. Expected bundle path: {expected_path}",
        in_bundle=False,
    )


def resolve_available_asset(asset_id: str, *, asset_root: Path | None = None) -> Path | None:
    status = check_asset_status(asset_id, asset_root=asset_root)
    return status.resolved_path


def ensure_asset_available(asset_id: str, *, asset_root: Path | None = None) -> Path:
    status = check_asset_status(asset_id, asset_root=asset_root)
    if status.present and status.resolved_path is not None:
        return status.resolved_path

    spec = get_asset_spec(asset_id)
    expected_path = status.expected_path
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if spec.source_type == "hf_file":
            downloaded = Path(hf_hub_download(repo_id=spec.repo_id, filename=spec.filename))
            if downloaded.resolve() != expected_path:
                shutil.copy2(downloaded, expected_path)
            return expected_path

        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(expected_path.parent),
            allow_patterns=list(spec.allow_patterns) or None,
        )
        if expected_path.exists():
            return expected_path
    except Exception as exc:
        missing_status = check_asset_status(asset_id, asset_root=asset_root)
        raise AssetNotAvailableError(
            asset_id,
            AssetStatus(
                asset_id=missing_status.asset_id,
                expected_path=missing_status.expected_path,
                present=False,
                resolved_path=None,
                resolved_from="missing",
                diagnostic=f"{missing_status.diagnostic} Download failed: {exc}",
                in_bundle=False,
            ),
        ) from exc

    raise AssetNotAvailableError(asset_id, check_asset_status(asset_id, asset_root=asset_root))


def install_assets(
    asset_ids: Sequence[str] | None = None,
    *,
    asset_root: Path | None = None,
    dry_run: bool = False,
) -> list[Dict[str, Any]]:
    root = asset_root or get_asset_root()
    results: list[Dict[str, Any]] = []
    for spec in iter_asset_specs(asset_ids):
        status = check_asset_status(spec.asset_id, asset_root=root)
        target_path = resolve_asset_path(spec.asset_id, asset_root=root)
        result: Dict[str, Any] = {
            "asset_id": spec.asset_id,
            "target_path": str(target_path),
            "already_present_in_bundle": status.in_bundle,
            "resolved_from": status.resolved_from,
        }
        if status.in_bundle:
            result["action"] = "skipped"
            results.append(result)
            continue
        if status.present and status.resolved_path is not None:
            if dry_run:
                result["action"] = "would_materialize"
                result["source_path"] = str(status.resolved_path)
                results.append(result)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if spec.source_type == "hf_snapshot":
                shutil.copytree(status.resolved_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(status.resolved_path, target_path)
            result["action"] = "materialized"
            result["source_path"] = str(status.resolved_path)
            result["installed_path"] = str(target_path)
            results.append(result)
            continue
        if dry_run:
            result["action"] = "would_download"
            results.append(result)
            continue
        installed_path = ensure_asset_available(spec.asset_id, asset_root=root)
        result["action"] = "downloaded"
        result["installed_path"] = str(installed_path)
        results.append(result)
    return results


def write_asset_manifest(
    install_results: Sequence[Dict[str, Any]],
    *,
    asset_root: Path | None = None,
    dry_run: bool = False,
) -> Path:
    root = asset_root or get_asset_root()
    manifest_path = (root / ASSET_MANIFEST_NAME).resolve()
    payload = {
        "asset_root": str(root.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": [],
    }
    for spec in iter_asset_specs():
        status = check_asset_status(spec.asset_id, asset_root=root)
        payload["assets"].append(
            {
                "asset_id": spec.asset_id,
                "repo_id": spec.repo_id,
                "relative_path": spec.relative_path,
                "source_type": spec.source_type,
                "filename": spec.filename or "",
                "present_in_bundle": status.in_bundle,
                "resolved_from": status.resolved_from,
                "resolved_path": str(status.resolved_path) if status.resolved_path else "",
            }
        )
    payload["install_results"] = list(install_results)
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def asset_bundle_report(*, asset_root: Path | None = None) -> Dict[str, Any]:
    root = asset_root or get_asset_root()
    items = [check_asset_status(asset_id, asset_root=root).to_dict() for asset_id in ALL_ASSET_IDS]
    missing_in_bundle = [item["asset_id"] for item in items if not Path(item["expected_path"]).exists()]
    return {
        "asset_root": str(root.resolve()),
        "all_assets_present_in_bundle": not missing_in_bundle,
        "missing_in_bundle": missing_in_bundle,
        "assets": items,
    }


def classify_asset_requirements(
    *,
    mode: str,
    ocr_backend: str,
    file_kind: str,
    pdf_model_size: str = "large",
    selected_tools: Sequence[str] | None = None,
) -> Dict[str, RequirementInfo]:
    del mode
    del ocr_backend

    kind = (file_kind or "image").strip().lower()
    pdf_size = (pdf_model_size or "large").strip().lower()
    selected = {tool.strip() for tool in (selected_tools or []) if tool and tool.strip()}

    requirements: Dict[str, RequirementInfo] = {}
    for asset_id in ALL_ASSET_IDS:
        if asset_id in IMAGE_CORE_ASSET_IDS:
            required = kind in {"image", "pdf"}
            requirements[asset_id] = RequirementInfo(
                asset_id=asset_id,
                required_for_current_run=required,
                blocking=required,
                reason="Required by the core image/reaction extraction pipeline." if required else "Unused for the current run.",
            )
            continue

        if asset_id in TEXT_TOOL_ASSET_IDS:
            selected_text = "text_extraction_agent" in selected
            requirements[asset_id] = RequirementInfo(
                asset_id=asset_id,
                required_for_current_run=selected_text,
                blocking=False,
                reason=(
                    "Required only if the planner selects text_extraction_agent."
                    if selected_text
                    else "Optional text-extraction asset; not required for base preflight."
                ),
            )
            continue

        if asset_id == "visualheist_base":
            required = kind == "pdf" and pdf_size == "base"
            requirements[asset_id] = RequirementInfo(
                asset_id=asset_id,
                required_for_current_run=required,
                blocking=required,
                reason="Required for PDF figure extraction with VisualHeist base." if required else "Unused unless processing PDFs with the base VisualHeist model.",
            )
            continue

        if asset_id == "visualheist_large":
            required = kind == "pdf" and pdf_size == "large"
            requirements[asset_id] = RequirementInfo(
                asset_id=asset_id,
                required_for_current_run=required,
                blocking=required,
                reason="Required for PDF figure extraction with VisualHeist large." if required else "Unused unless processing PDFs with the large VisualHeist model.",
            )
            continue

    return requirements


def build_asset_preflight_report(
    *,
    mode: str,
    ocr_backend: str,
    file_kind: str,
    pdf_model_size: str = "large",
    selected_tools: Sequence[str] | None = None,
    asset_root: Path | None = None,
) -> Dict[str, Any]:
    root = asset_root or get_asset_root()
    requirements = classify_asset_requirements(
        mode=mode,
        ocr_backend=ocr_backend,
        file_kind=file_kind,
        pdf_model_size=pdf_model_size,
        selected_tools=selected_tools,
    )

    blocking_errors: list[str] = []
    warnings: list[str] = []
    items: list[Dict[str, Any]] = []

    for asset_id in ALL_ASSET_IDS:
        requirement = requirements[asset_id]
        status = check_asset_status(asset_id, asset_root=root)
        item = {
            **status.to_dict(),
            **requirement.to_dict(),
        }
        items.append(item)
        if requirement.blocking and not status.present:
            blocking_errors.append(f"{asset_id}: {status.diagnostic}")
        elif requirement.required_for_current_run and not status.present:
            warnings.append(f"{asset_id}: {status.diagnostic}")

    return {
        "asset_root": str(root.resolve()),
        "mode": mode,
        "ocr_backend": ocr_backend,
        "file_kind": file_kind,
        "pdf_model_size": pdf_model_size,
        "selected_tools": list(selected_tools or []),
        "assets": items,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
    }
