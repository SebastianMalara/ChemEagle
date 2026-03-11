#!/usr/bin/env python3
import gc
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gradio as gr
except ImportError as e:
    raise ImportError("gradio is required for gui_app.py. Install dependencies with: pip install -r requirements.txt") from e

from asset_registry import ASSET_ENV_VAR, build_asset_preflight_report
from llm_preflight import collect_runtime_provider_preflight
from llm_profiles import MANUAL_MODEL_LIST_PROVIDERS, list_available_models, resolve_llm_profile
from review_artifacts import create_artifact_store_from_config
from review_service import get_review_service
from runtime_device import resolve_ocr_backend

import pandas as pd


ENV_FILE_DEFAULT = Path(".env.chemeagle")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LLM_PROVIDER_CHOICES = ["azure", "openai", "openai_compatible", "lmstudio", "local_openai", "anthropic"]
OCR_BACKEND_CHOICES = ["auto", "llm_vision", "easyocr", "tesseract"]
PDF_MODEL_CHOICES = ["base", "large"]
ENV_KEYS = [
    "CHEMEAGLE_RUN_MODE",
    "CHEMEAGLE_DEVICE",
    "CHEMEAGLE_ASSET_ROOT",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OCR_BACKEND",
    "OCR_LLM_INHERIT_MAIN",
    "OCR_LLM_PROVIDER",
    "OCR_LLM_MODEL",
    "OCR_API_KEY",
    "OCR_AZURE_ENDPOINT",
    "OCR_API_VERSION",
    "OCR_OPENAI_API_KEY",
    "OCR_OPENAI_BASE_URL",
    "OCR_ANTHROPIC_API_KEY",
    "OCR_VLLM_BASE_URL",
    "OCR_VLLM_API_KEY",
    "OCR_LANG",
    "OCR_CONFIG",
    "TESSERACT_CMD",
    "PDF_MODEL_SIZE",
    "PDF_PERSIST_IMAGES",
    "PDF_PERSIST_DIR",
    "API_KEY",
    "AZURE_ENDPOINT",
    "API_VERSION",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "ARTIFACT_BACKEND",
    "ARTIFACT_FILESYSTEM_ROOT",
    "ARTIFACT_S3_ENDPOINT_URL",
    "ARTIFACT_S3_ACCESS_KEY_ID",
    "ARTIFACT_S3_SECRET_ACCESS_KEY",
    "ARTIFACT_S3_BUCKET",
    "ARTIFACT_S3_REGION",
    "ARTIFACT_S3_USE_SSL",
    "ARTIFACT_S3_KEY_PREFIX",
    "REVIEW_DB_PATH",
]
DEFAULT_REVIEW_DB_FILENAME = "review_dataset.sqlite3"
DEFAULT_REVIEW_DB_PATH = str(Path(f"./data/{DEFAULT_REVIEW_DB_FILENAME}").resolve())
DEFAULT_ARTIFACT_FILESYSTEM_ROOT = str(Path("./data/artifacts").resolve())
DEFAULT_EXPORT_DIR = str(Path("./data/exports").resolve())
DEFAULT_MINIO_ENDPOINT = "http://127.0.0.1:9000"
DEFAULT_MINIO_BUCKET = "chemeagle-review"
ALL_EXPERIMENTS = "__all_experiments__"


def parse_env_file(env_path: Path) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if not env_path.exists():
        return vals
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        vals[k.strip()] = v.strip()
    return vals


def merged_env_values(env_path: Path) -> Dict[str, str]:
    from_file = parse_env_file(env_path)
    out: Dict[str, str] = {}
    for key in ENV_KEYS:
        out[key] = os.getenv(key, from_file.get(key, ""))
    return out


def save_env_file(env_path: Path, values: Dict[str, str]) -> str:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    managed_keys = set(ENV_KEYS)
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    rendered_lines: List[str] = []
    seen_keys: set[str] = set()

    for line in existing_lines:
        raw = line.strip()
        if raw and not raw.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in managed_keys:
                seen_keys.add(key)
                value = values.get(key, "")
                if value:
                    rendered_lines.append(f"{key}={value}")
                continue
        rendered_lines.append(line)

    for key in ENV_KEYS:
        if key in seen_keys:
            continue
        value = values.get(key, "")
        if value:
            rendered_lines.append(f"{key}={value}")

    payload = "\n".join(rendered_lines).rstrip()
    env_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    saved_count = sum(1 for key in ENV_KEYS if values.get(key, ""))
    return f"Saved {env_path} with {saved_count} keys."


def apply_runtime_env(values: Dict[str, str]) -> None:
    for key, value in values.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

    pref = (values.get("CHEMEAGLE_DEVICE") or "").strip().lower()
    if pref in {"auto", "cuda"} and not os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def build_runtime_values(
    mode: str,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    chemeagle_device: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
) -> Dict[str, str]:
    return {
        "CHEMEAGLE_RUN_MODE": mode,
        "CHEMEAGLE_ASSET_ROOT": os.getenv(ASSET_ENV_VAR, ""),
        "LLM_PROVIDER": llm_provider,
        "LLM_MODEL": llm_model,
        "API_KEY": api_key,
        "AZURE_ENDPOINT": azure_endpoint,
        "API_VERSION": api_version,
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_BASE_URL": openai_base_url,
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "VLLM_BASE_URL": vllm_base_url,
        "VLLM_API_KEY": vllm_api_key,
        "CHEMEAGLE_DEVICE": chemeagle_device,
        "OCR_BACKEND": ocr_backend,
        "OCR_LLM_INHERIT_MAIN": "1" if ocr_llm_inherit_main else "0",
        "OCR_LLM_PROVIDER": ocr_llm_provider,
        "OCR_LLM_MODEL": ocr_llm_model,
        "OCR_API_KEY": ocr_api_key,
        "OCR_AZURE_ENDPOINT": ocr_azure_endpoint,
        "OCR_API_VERSION": ocr_api_version,
        "OCR_OPENAI_API_KEY": ocr_openai_api_key,
        "OCR_OPENAI_BASE_URL": ocr_openai_base_url,
        "OCR_ANTHROPIC_API_KEY": ocr_anthropic_api_key,
        "OCR_VLLM_BASE_URL": ocr_vllm_base_url,
        "OCR_VLLM_API_KEY": ocr_vllm_api_key,
        "OCR_LANG": ocr_lang,
        "OCR_CONFIG": ocr_config,
        "TESSERACT_CMD": tesseract_cmd,
        "PDF_MODEL_SIZE": pdf_model_size,
        "PDF_PERSIST_IMAGES": "1" if pdf_persist_images else "0",
        "PDF_PERSIST_DIR": pdf_persist_dir,
    }


def _resolve_upload_path(upload) -> str:
    if upload is None:
        return ""
    if isinstance(upload, str):
        return upload
    return getattr(upload, "name", "") or ""


def _resolve_upload_paths(upload) -> List[str]:
    if upload is None:
        return []
    if isinstance(upload, list):
        paths: List[str] = []
        for item in upload:
            paths.extend(_resolve_upload_paths(item))
        return [path for path in paths if path]
    return [path for path in [_resolve_upload_path(upload)] if path]


def _scan_supported_files(folder_path: str) -> List[str]:
    raw = (folder_path or "").strip()
    if not raw:
        return []
    root = Path(raw).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    suffixes = IMAGE_SUFFIXES | {".pdf"}
    files = [str(path.resolve()) for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes]
    return files


def _env_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_dataset_runtime_values(
    mode: str,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    chemeagle_device: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
    artifact_backend: str,
    artifact_filesystem_root: str,
    artifact_s3_endpoint_url: str,
    artifact_s3_access_key_id: str,
    artifact_s3_secret_access_key: str,
    artifact_s3_bucket: str,
    artifact_s3_region: str,
    artifact_s3_use_ssl: bool,
    artifact_s3_key_prefix: str,
    review_db_path: str,
) -> Dict[str, str]:
    values = build_runtime_values(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )
    values.update(
        {
            "ARTIFACT_BACKEND": artifact_backend,
            "ARTIFACT_FILESYSTEM_ROOT": artifact_filesystem_root,
            "ARTIFACT_S3_ENDPOINT_URL": artifact_s3_endpoint_url,
            "ARTIFACT_S3_ACCESS_KEY_ID": artifact_s3_access_key_id,
            "ARTIFACT_S3_SECRET_ACCESS_KEY": artifact_s3_secret_access_key,
            "ARTIFACT_S3_BUCKET": artifact_s3_bucket,
            "ARTIFACT_S3_REGION": artifact_s3_region,
            "ARTIFACT_S3_USE_SSL": "1" if artifact_s3_use_ssl else "0",
            "ARTIFACT_S3_KEY_PREFIX": artifact_s3_key_prefix,
            "REVIEW_DB_PATH": review_db_path,
        }
    )
    return values


def _parse_profile_configs(base_values: Dict[str, str], profiles_json: str) -> List[Dict[str, str]]:
    raw = (profiles_json or "").strip()
    if not raw:
        return [{**base_values, "profile_label": "baseline"}]
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed = [parsed]
    configs: List[Dict[str, str]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        merged = dict(base_values)
        for key, value in item.items():
            if value is None:
                continue
            merged[str(key)] = str(value) if not isinstance(value, bool) else ("1" if value else "0")
            if str(key).upper() in merged:
                merged[str(key).upper()] = merged[str(key)]
        merged["profile_label"] = str(item.get("profile_label") or f"profile-{index + 1}")
        configs.append(merged)
    return configs or [{**base_values, "profile_label": "baseline"}]


def _collect_batch_runtime_diagnostics(
    *,
    base_values: Dict[str, str],
    profile_configs: List[Dict[str, str]],
    source_paths: List[str],
    mode: str,
) -> Dict[str, Any]:
    has_pdf = any(Path(path).suffix.lower() == ".pdf" for path in source_paths)
    sample_file = source_paths[0] if source_paths else ""
    static_preflight = collect_preflight_diagnostics(sample_file, mode, base_values, include_pdf_section=has_pdf)
    runtime_provider_preflight = collect_runtime_provider_preflight(profile_configs=profile_configs, mode=mode)
    blocking_errors = list(static_preflight.get("blocking_errors", []))
    blocking_errors.extend(f"runtime_provider_preflight: {item}" for item in runtime_provider_preflight.get("blocking_errors", []))
    warnings = list(static_preflight.get("warnings", []))
    warnings.extend(f"runtime_provider_preflight: {item}" for item in runtime_provider_preflight.get("warnings", []))
    return {
        "source_paths": source_paths,
        "profile_labels": [config.get("profile_label", "") for config in profile_configs],
        "static_preflight": static_preflight,
        "runtime_provider_preflight": runtime_provider_preflight,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
    }


def _parse_newline_paths(raw: str) -> List[str]:
    return [line.strip() for line in (raw or "").splitlines() if line.strip()]


def _picked_path(upload: Any) -> str:
    paths = _resolve_upload_paths(upload)
    return paths[0] if paths else ""


def _picked_directory(upload: Any) -> str:
    picked = _picked_path(upload)
    if not picked:
        return ""
    candidate = Path(picked).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    return str(candidate.resolve())


def _replace_path_from_picker(upload: Any, current: str) -> str:
    picked = _picked_path(upload)
    return picked or current or ""


def _replace_directory_from_picker(upload: Any, current: str) -> str:
    picked = _picked_directory(upload)
    return picked or current or ""


def _replace_db_path_from_directory(upload: Any, current: str) -> str:
    picked = _picked_directory(upload)
    if not picked:
        return current or DEFAULT_REVIEW_DB_PATH
    filename = Path(current).name if current else DEFAULT_REVIEW_DB_FILENAME
    return str((Path(picked).expanduser() / filename).resolve())


def _append_path_from_picker(existing: str, upload: Any) -> str:
    picked = _picked_directory(upload)
    if not picked:
        return existing or ""
    values = _parse_newline_paths(existing)
    if picked not in values:
        values.append(picked)
    return "\n".join(values)


def _runs_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["experiment_name", "run_id", "profile_label", "ingest_mode", "status", "estimated_cost_usd", "total_reactions"])
    return pd.DataFrame(
        [
            {
                "experiment_name": row.get("experiment_name", ""),
                "run_id": row.get("run_id", ""),
                "profile_label": row.get("profile_label", ""),
                "ingest_mode": row.get("ingest_mode", ""),
                "status": row.get("status", ""),
                "estimated_cost_usd": row.get("estimated_cost_usd"),
                "total_reactions": row.get("total_reactions", 0),
                "total_failures": row.get("total_failures", 0),
                "total_redo": row.get("total_redo", 0),
                "total_tokens": row.get("total_tokens", 0),
            }
            for row in records
        ]
    )


def _review_run_choices(records: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = []
    for row in records:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        label = (
            f"{row.get('experiment_name', '')} | {run_id} | "
            f"{row.get('status', '')} | reactions={row.get('total_reactions', 0)}"
        )
        choices.append((label, run_id))
    return choices


def _run_sources_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "run_source_id",
                "input_order",
                "source",
                "source_type",
                "status",
                "current_phase",
                "completed_derived_images",
                "expected_derived_images",
                "reaction_count",
                "failed_derived_images",
                "error_summary",
            ]
        )
    return pd.DataFrame(
        [
            {
                "run_source_id": row.get("run_source_id", ""),
                "input_order": row.get("input_order", 0),
                "source": row.get("original_filename", ""),
                "source_type": row.get("source_type", ""),
                "status": row.get("status", ""),
                "current_phase": row.get("current_phase", ""),
                "completed_derived_images": row.get("completed_derived_images", 0),
                "expected_derived_images": row.get("expected_derived_images", 0),
                "reaction_count": row.get("reaction_count", 0),
                "failed_derived_images": row.get("failed_derived_images", 0),
                "error_summary": row.get("error_summary", ""),
            }
            for row in records
        ]
    )


def _derived_images_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "derived_image_id",
                "image_index",
                "page_hint",
                "status",
                "outcome_class",
                "reaction_count",
                "accepted_reaction_count",
                "rejected_reaction_count",
                "attempt_count",
                "normalization_status",
                "last_retry_reason",
                "error_text",
            ]
        )
    return pd.DataFrame(
        [
            {
                "derived_image_id": row.get("derived_image_id", ""),
                "image_index": row.get("image_index", 0),
                "page_hint": row.get("page_hint", ""),
                "status": row.get("status", ""),
                "outcome_class": row.get("outcome_class", ""),
                "reaction_count": row.get("reaction_count", 0),
                "accepted_reaction_count": row.get("accepted_reaction_count", row.get("reaction_count", 0)),
                "rejected_reaction_count": row.get("rejected_reaction_count", 0),
                "attempt_count": row.get("attempt_count", 0),
                "normalization_status": row.get("normalization_status", ""),
                "last_retry_reason": row.get("last_retry_reason", ""),
                "error_text": row.get("error_text", ""),
            }
            for row in records
        ]
    )


def _attempts_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "attempt_id",
                "attempt_no",
                "trigger",
                "execution_mode",
                "status",
                "failure_kind",
                "error_summary",
                "raw_artifact_key",
            ]
        )
    return pd.DataFrame(
        [
            {
                "attempt_id": row.get("attempt_id", ""),
                "attempt_no": row.get("attempt_no", 0),
                "trigger": row.get("trigger", ""),
                "execution_mode": row.get("execution_mode", ""),
                "status": row.get("status", ""),
                "failure_kind": row.get("failure_kind", ""),
                "error_summary": row.get("error_summary", ""),
                "raw_artifact_key": row.get("raw_artifact_key", ""),
            }
            for row in records
        ]
    )


def _retry_candidates_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "derived_image_id",
                "source",
                "image_index",
                "status",
                "outcome_class",
                "accepted_reaction_count",
                "rejected_reaction_count",
                "normalization_status",
                "error_text",
            ]
        )
    return pd.DataFrame(
        [
            {
                "derived_image_id": row.get("derived_image_id", ""),
                "source": row.get("original_filename", ""),
                "image_index": row.get("image_index", 0),
                "status": row.get("status", ""),
                "outcome_class": row.get("outcome_class", ""),
                "accepted_reaction_count": row.get("accepted_reaction_count", row.get("reaction_count", 0)),
                "rejected_reaction_count": row.get("rejected_reaction_count", 0),
                "normalization_status": row.get("normalization_status", ""),
                "error_text": row.get("error_text", ""),
            }
            for row in records
        ]
    )


def _reactions_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["reaction_uid", "run_id", "reaction_id", "review_status", "outcome_class", "structure_quality", "source"])
    return pd.DataFrame(
        [
            {
                "reaction_uid": row.get("reaction_uid", ""),
                "run_id": row.get("run_id", ""),
                "reaction_id": row.get("reaction_id", ""),
                "review_status": row.get("review_status", ""),
                "outcome_class": row.get("outcome_class", ""),
                "structure_quality": row.get("structure_quality", ""),
                "profile_label": row.get("profile_label", ""),
                "source": row.get("original_filename", ""),
                "estimated_cost_usd": row.get("estimated_cost_usd"),
            }
            for row in records
        ]
    )


def _artifact_preview_path(values: Dict[str, str], backend: str, key: str, *, suffix: str) -> Optional[str]:
    if not key:
        return None
    store = create_artifact_store_from_config({**values, "ARTIFACT_BACKEND": backend or values.get("ARTIFACT_BACKEND", "filesystem")})
    if backend == "filesystem":
        ref = store.get_download_ref(key)
        return ref if Path(ref).exists() else None
    data = store.get_bytes(key)
    temp = tempfile.NamedTemporaryFile(prefix="review_preview_", suffix=suffix, delete=False)
    temp.write(data)
    temp.flush()
    temp.close()
    return temp.name


def _artifact_download_ref(values: Dict[str, str], backend: str, key: str) -> str:
    if not key:
        return ""
    store = create_artifact_store_from_config({**values, "ARTIFACT_BACKEND": backend or values.get("ARTIFACT_BACKEND", "filesystem")})
    return store.get_download_ref(key)


def _model_picker_help(provider: str) -> str:
    provider_norm = (provider or "").strip().lower()
    if provider_norm in MANUAL_MODEL_LIST_PROVIDERS:
        return "Azure model listing stays manual in this pass. Enter the deployment/model name directly."
    if provider_norm in {"openai_compatible", "lmstudio", "local_openai"}:
        return "Manual entry is enabled. Refresh will try GET /models on the configured compatible endpoint."
    return "Manual entry is enabled. Click Refresh Models to fetch the current provider catalog."


def _ocr_profile_summary(inherit_main: bool, main_provider: str, main_model: str, ocr_provider: str, ocr_model: str) -> str:
    if inherit_main:
        return (
            "LLM vision OCR inherits the main profile: "
            f"provider={main_provider or 'unset'}, model={main_model or '(provider default)'}."
        )
    return (
        "LLM vision OCR uses its own profile: "
        f"provider={ocr_provider or main_provider or 'unset'}, model={ocr_model or '(provider default)'}."
    )


def _model_choices(current_value: str, catalog_ids: List[str]) -> List[str]:
    ordered: List[str] = []
    if current_value:
        ordered.append(current_value)
    for model_id in catalog_ids:
        if model_id and model_id not in ordered:
            ordered.append(model_id)
    return ordered


def _model_catalog_guard(profile) -> str:
    provider = profile.provider
    if provider in MANUAL_MODEL_LIST_PROVIDERS:
        return "Azure model listing stays manual in this pass. Enter the deployment/model name directly."
    if provider == "anthropic" and not profile.api_key:
        return "Set ANTHROPIC_API_KEY before refreshing Anthropic models."
    if provider == "openai" and not profile.api_key and not profile.base_url:
        return "Set OPENAI_API_KEY (or a compatible base URL) before refreshing OpenAI models."
    if provider in {"openai_compatible", "lmstudio", "local_openai"} and not profile.base_url:
        return f"Set a base URL before refreshing models for {provider}."
    return ""


def _main_model_picker_updates(llm_provider: str):
    provider_norm = (llm_provider or "").strip().lower()
    return gr.update(interactive=provider_norm not in MANUAL_MODEL_LIST_PROVIDERS), _model_picker_help(provider_norm)


def _ocr_model_picker_updates(
    ocr_llm_inherit_main: bool,
    llm_provider: str,
    llm_model: str,
    ocr_llm_provider: str,
    ocr_llm_model: str,
):
    inherit_main = bool(ocr_llm_inherit_main)
    provider_norm = ((ocr_llm_provider if not inherit_main else llm_provider) or "").strip().lower()
    interactive = not inherit_main
    refresh_interactive = interactive and provider_norm not in MANUAL_MODEL_LIST_PROVIDERS
    if inherit_main:
        status = "LLM vision OCR is inheriting the main profile. Disable inherit to pick a separate provider/model."
    else:
        status = _model_picker_help(provider_norm)
    summary = _ocr_profile_summary(inherit_main, llm_provider, llm_model, ocr_llm_provider, ocr_llm_model)
    field_updates = [gr.update(interactive=interactive)] * 10
    return (
        *field_updates,
        gr.update(interactive=refresh_interactive),
        status,
        summary,
    )


def _build_values_from_form(
    mode: str,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    chemeagle_device: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
) -> Dict[str, str]:
    return build_runtime_values(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )


def _resolve_pdf_persist_dir(pdf_path: str, persist_dir: str) -> Path:
    pdf_stem = Path(pdf_path).stem or "pdf"
    if persist_dir.strip():
        base_dir = Path(persist_dir).expanduser()
        target_dir = base_dir / pdf_stem
        counter = 1
        while target_dir.exists():
            target_dir = base_dir / f"{pdf_stem}-{counter}"
            counter += 1
        return target_dir

    default_base = Path("debug") / "pdf_images"
    default_base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{pdf_stem}-", dir=str(default_base)))


def _trim_text(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _html_escape(value: Any) -> str:
    return html.escape(str(value or ""))


def _render_progress_monitor(monitor: Dict[str, Any]) -> str:
    progress = monitor.get("progress", {}) or {}
    run = monitor.get("run", {}) or {}
    progress_fraction = max(0.0, min(float(progress.get("progress_fraction", 0.0) or 0.0), 1.0))
    width_percent = round(progress_fraction * 100, 1)
    active = bool(progress.get("is_active"))
    spinner = '<span class="ce-monitor-spinner" aria-hidden="true"></span>' if active else ""
    tone = str(run.get("status") or "queued").strip().lower()
    border_color = {
        "completed": "#16a34a",
        "failed": "#dc2626",
        "interrupted": "#d97706",
    }.get(tone, "#f97316")
    source_label = progress.get("current_source_label") or "Waiting for next source"
    phase_label = progress.get("current_phase_label") or "Queued"
    summary = progress.get("status_summary") or ""
    run_id = run.get("run_id") or ""
    experiment_id = run.get("experiment_id") or ""
    progress_label = progress.get("progress_label") or "0/0 sources finished"
    return f"""
<style>
.ce-monitor {{
  border: 1px solid #e5e7eb;
  border-left: 4px solid {border_color};
  border-radius: 10px;
  padding: 14px 16px;
  background: #fafafa;
}}
.ce-monitor-head {{
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  color: #111827;
}}
.ce-monitor-spinner {{
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid rgba(249, 115, 22, 0.25);
  border-top-color: #f97316;
  display: inline-block;
  animation: ce-monitor-spin 0.8s linear infinite;
}}
.ce-monitor-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px 16px;
  margin-top: 10px;
  color: #374151;
  font-size: 14px;
}}
.ce-monitor-bar {{
  margin-top: 12px;
  width: 100%;
  height: 10px;
  background: #e5e7eb;
  border-radius: 999px;
  overflow: hidden;
}}
.ce-monitor-bar-fill {{
  height: 100%;
  width: {width_percent}%;
  background: linear-gradient(90deg, #fb923c 0%, #f97316 100%);
}}
.ce-monitor-summary {{
  margin-top: 10px;
  color: #1f2937;
  font-size: 14px;
}}
@keyframes ce-monitor-spin {{
  to {{ transform: rotate(360deg); }}
}}
</style>
<div class="ce-monitor">
  <div class="ce-monitor-head">{spinner}<span>{_html_escape(phase_label)}</span></div>
  <div class="ce-monitor-grid">
    <div><strong>Experiment</strong><br>{_html_escape(experiment_id)}</div>
    <div><strong>Run</strong><br>{_html_escape(run_id)}</div>
    <div><strong>Source</strong><br>{_html_escape(source_label)}</div>
    <div><strong>Progress</strong><br>{_html_escape(progress_label)}</div>
  </div>
  <div class="ce-monitor-bar"><div class="ce-monitor-bar-fill"></div></div>
  <div class="ce-monitor-summary">{_html_escape(summary)}</div>
</div>
""".strip()


def _probe_python_code(code: str, env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": _trim_text(proc.stdout),
        "stderr": _trim_text(proc.stderr),
    }


def _resolve_tesseract_cmd(values: Dict[str, str]) -> str:
    explicit = values.get("TESSERACT_CMD") or os.getenv("CHEMEAGLE_TESSERACT_CMD") or ""
    if explicit:
        normalized = os.path.normpath(explicit)
        if os.path.exists(normalized):
            return normalized
    return shutil.which("tesseract") or ""


def _profile_preflight(scope: str, *, required: bool, values: Dict[str, str]) -> Dict[str, Any]:
    blocking_errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {
        "scope": scope,
        "required": required,
    }

    try:
        profile = resolve_llm_profile(scope=scope, values=values, default_model="gpt-5-mini")
    except Exception as exc:
        if required:
            blocking_errors.append(f"Failed to resolve the {scope} LLM profile: {exc}")
        else:
            warnings.append(f"The optional {scope} LLM profile could not be resolved: {exc}")
        checks["resolution_error"] = str(exc)
        return {
            "blocking_errors": blocking_errors,
            "warnings": warnings,
            "checks": checks,
        }

    checks.update(
        {
            "provider": profile.provider,
            "model": profile.model,
            "inherit_main": getattr(profile, "inherit_main", False),
            "base_url": profile.base_url,
            "azure_endpoint_present": bool(profile.azure_endpoint),
            "api_key_present": bool(profile.api_key),
            "api_version": profile.api_version,
        }
    )

    if not required:
        checks["status"] = "not_required_for_current_run"
        return {
            "blocking_errors": blocking_errors,
            "warnings": warnings,
            "checks": checks,
        }

    provider = profile.provider
    if provider == "azure":
        if not profile.api_key:
            blocking_errors.append(f"{scope} Azure profile requires API_KEY or AZURE_OPENAI_API_KEY.")
        if not profile.azure_endpoint:
            blocking_errors.append(f"{scope} Azure profile requires AZURE_ENDPOINT or AZURE_OPENAI_ENDPOINT.")
    elif provider in {"openai", "openai_compatible", "lmstudio", "local_openai"}:
        if provider in {"openai_compatible", "lmstudio", "local_openai"} and not profile.base_url:
            blocking_errors.append(f"{scope} {provider} profile requires OPENAI_BASE_URL or VLLM_BASE_URL.")
        if provider == "openai" and not profile.api_key and not profile.base_url:
            blocking_errors.append(f"{scope} OpenAI profile requires OPENAI_API_KEY or an explicit compatible base URL.")
    elif provider == "anthropic":
        if not profile.api_key:
            blocking_errors.append(f"{scope} Anthropic profile requires ANTHROPIC_API_KEY.")
    else:
        blocking_errors.append(f"Unsupported LLM provider for {scope}: {provider}")

    if not profile.model:
        warnings.append(f"{scope} model is empty; runtime will fall back to provider defaults when possible.")

    return {
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "checks": checks,
    }


def _ocr_preflight(mode: str, values: Dict[str, str]) -> Dict[str, Any]:
    requested_backend = values.get("OCR_BACKEND") or "auto"
    resolved_backend = resolve_ocr_backend(requested_backend, mode)
    checks: Dict[str, Any] = {
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "ocr_lang": values.get("OCR_LANG") or "eng",
    }
    blocking_errors: List[str] = []
    warnings: List[str] = []
    env = dict(os.environ)

    allowed_backends = {"llm_vision", "easyocr", "tesseract"}
    if resolved_backend not in allowed_backends:
        blocking_errors.append(f"Unsupported OCR backend: {resolved_backend}")
        return {
            "blocking_errors": blocking_errors,
            "warnings": warnings,
            "checks": checks,
        }

    probe_lines: List[str] = []
    if resolved_backend == "easyocr":
        probe_lines.append("import easyocr")
    elif resolved_backend == "tesseract":
        probe_lines.append("import pytesseract")
    if probe_lines:
        checks["python_import_probe"] = _probe_python_code("\n".join(probe_lines), env)
        if not checks["python_import_probe"]["ok"]:
            blocking_errors.append("OCR backend dependencies failed to import. See diagnostics.ocr_preflight.checks.python_import_probe.")

    if resolved_backend == "tesseract":
        tesseract_cmd = _resolve_tesseract_cmd(values)
        checks["tesseract_cmd"] = tesseract_cmd
        if not tesseract_cmd:
            blocking_errors.append("Tesseract backend selected, but no Tesseract executable was found.")
    elif resolved_backend == "easyocr":
        checks["device_pref"] = values.get("CHEMEAGLE_DEVICE") or "auto"
    elif resolved_backend == "llm_vision":
        try:
            ocr_profile = resolve_llm_profile(scope="ocr", values=values, default_model="gpt-5-mini")
            checks["ocr_llm_provider"] = ocr_profile.provider
            checks["ocr_llm_model"] = ocr_profile.model
            checks["ocr_llm_inherit_main"] = getattr(ocr_profile, "inherit_main", False)
        except Exception as exc:
            warnings.append(f"Could not resolve the OCR LLM profile during OCR preflight: {exc}")
        warnings.append("LLM vision OCR assumes the selected model supports image inputs.")

    if requested_backend == "auto":
        checks["auto_resolution_rule"] = f"{mode} -> {resolved_backend}"

    return {
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "checks": checks,
    }


def _model_catalog_preflight(mode: str, resolved_ocr_backend: str, values: Dict[str, str]) -> Dict[str, Any]:
    blocking_errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {}

    catalog_scopes = [
        ("main", mode == "cloud"),
        ("ocr", resolved_ocr_backend == "llm_vision"),
    ]
    for scope, required in catalog_scopes:
        try:
            profile = resolve_llm_profile(scope=scope, values=values, default_model="gpt-5-mini")
        except Exception as exc:
            checks[scope] = {"resolution_error": str(exc), "required": required}
            continue

        entry = {
            "provider": profile.provider,
            "required": required,
            "inherit_main": getattr(profile, "inherit_main", False),
            "supports_refresh": profile.provider not in MANUAL_MODEL_LIST_PROVIDERS,
            "manual_only": profile.provider in MANUAL_MODEL_LIST_PROVIDERS,
        }
        if profile.provider in MANUAL_MODEL_LIST_PROVIDERS:
            entry["status"] = "Azure stays manual-only for model selection in this pass."
        elif profile.provider in {"openai_compatible", "lmstudio", "local_openai"}:
            entry["status"] = "Refresh will try GET /models on the configured compatible endpoint."
            if not profile.base_url and required:
                warnings.append(f"{scope} model refresh needs a base URL before GET /models can work.")
        elif profile.provider == "anthropic":
            entry["status"] = "Refresh will call Anthropic's models.list() endpoint."
        else:
            entry["status"] = "Refresh will call the provider model catalog endpoint."
        checks[scope] = entry

    return {
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "checks": checks,
    }


def _visualheist_cache_state(model_size: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    local_weights = repo_root / "safetensors" / f"{model_size}_model.safetensors"
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    repo_cache = hf_home / "hub" / f"models--shixuanleong--visualheist-{model_size}"
    snapshot_root = repo_cache / "snapshots"
    snapshot_dirs = sorted(str(path) for path in snapshot_root.glob("*")) if snapshot_root.exists() else []
    return {
        "local_weights": str(local_weights),
        "local_weights_present": local_weights.exists(),
        "huggingface_repo_cache": str(repo_cache),
        "cached_snapshots": snapshot_dirs,
        "cached": local_weights.exists() or bool(snapshot_dirs),
    }


def _pdf_preflight(file_path: str, values: Dict[str, str]) -> Dict[str, Any]:
    model_size = (values.get("PDF_MODEL_SIZE") or "large").strip().lower()
    checks: Dict[str, Any] = {
        "model_size": model_size,
        "pdftoppm": shutil.which("pdftoppm") or "",
    }
    blocking_errors: List[str] = []
    warnings: List[str] = []

    if model_size not in {"base", "large"}:
        blocking_errors.append(f"Unsupported PDF_MODEL_SIZE: {model_size}")

    if not checks["pdftoppm"]:
        blocking_errors.append("pdf2image requires pdftoppm, but it was not found on PATH.")

    if file_path:
        candidate = Path(file_path)
        checks["file"] = str(candidate)
        checks["file_exists"] = candidate.exists()
        if not candidate.exists():
            blocking_errors.append(f"Uploaded PDF does not exist: {candidate}")
        elif candidate.suffix.lower() != ".pdf":
            blocking_errors.append(f"Expected a PDF upload, got {candidate.suffix or 'no extension'}.")
    else:
        warnings.append("No PDF uploaded; file-specific PDF checks were skipped.")

    try:
        with tempfile.TemporaryDirectory(prefix="chemeagle_preflight_") as tmpdir:
            probe_file = Path(tmpdir) / "write_test.txt"
            probe_file.write_text("ok", encoding="utf-8")
        checks["temp_dir_writable"] = True
    except Exception as exc:
        checks["temp_dir_writable"] = False
        checks["temp_dir_error"] = str(exc)
        blocking_errors.append(f"Temporary output directory is not writable: {exc}")

    checks["visualheist_cache"] = _visualheist_cache_state(model_size)
    if not checks["visualheist_cache"]["cached"]:
        warnings.append(f"VisualHeist {model_size} weights are not in the legacy locations or Hugging Face cache.")

    env = dict(os.environ)
    checks["python_import_probe"] = _probe_python_code(
        "from pdf_extraction import run_pdf\nfrom pdfmodel.methods import _pdf_to_figures_and_tables",
        env,
    )
    if not checks["python_import_probe"]["ok"]:
        blocking_errors.append("PDF extraction dependencies failed to import. See diagnostics.pdf_preflight.checks.python_import_probe.")

    return {
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "checks": checks,
    }


def _asset_preflight(
    file_path: str,
    mode: str,
    values: Dict[str, str],
    *,
    selected_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    suffix = Path(file_path).suffix.lower() if file_path else ""
    file_kind = "pdf" if suffix == ".pdf" else "image"
    return build_asset_preflight_report(
        mode=mode,
        ocr_backend=resolve_ocr_backend(values.get("OCR_BACKEND"), mode),
        file_kind=file_kind,
        pdf_model_size=values.get("PDF_MODEL_SIZE") or "large",
        selected_tools=selected_tools,
    )


def collect_preflight_diagnostics(file_path: str, mode: str, values: Dict[str, str], *, include_pdf_section: bool) -> Dict[str, Any]:
    resolved_ocr_backend = resolve_ocr_backend(values.get("OCR_BACKEND"), mode)
    main_required = mode == "cloud"
    ocr_llm_required = resolved_ocr_backend == "llm_vision"
    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "device": values.get("CHEMEAGLE_DEVICE") or "auto",
        "resolved_ocr_backend": resolved_ocr_backend,
        "main_llm_preflight": _profile_preflight("main", required=main_required, values=values),
        "ocr_llm_preflight": _profile_preflight("ocr", required=ocr_llm_required, values=values),
        "model_catalog_preflight": _model_catalog_preflight(mode, resolved_ocr_backend, values),
        "ocr_preflight": _ocr_preflight(mode, values),
        "asset_preflight": _asset_preflight(file_path, mode, values),
        "runtime_provider_preflight": collect_runtime_provider_preflight(profile_configs=[values], mode=mode),
    }

    if include_pdf_section:
        diagnostics["pdf_preflight"] = _pdf_preflight(file_path, values)

    blocking_errors: List[str] = []
    warnings: List[str] = []
    for section_name in (
        "main_llm_preflight",
        "ocr_llm_preflight",
        "model_catalog_preflight",
        "ocr_preflight",
        "asset_preflight",
        "pdf_preflight",
        "runtime_provider_preflight",
    ):
        section = diagnostics.get(section_name)
        if not section:
            continue
        blocking_errors.extend(f"{section_name}: {item}" for item in section.get("blocking_errors", []))
        warnings.extend(f"{section_name}: {item}" for item in section.get("warnings", []))

    diagnostics["blocking_errors"] = blocking_errors
    diagnostics["warnings"] = warnings
    return diagnostics


def _append_preflight_status(status_bits: List[str], diagnostics: Dict[str, Any]) -> None:
    status_bits.append(f"Resolved OCR backend: {diagnostics.get('resolved_ocr_backend')}")
    if diagnostics.get("blocking_errors"):
        status_bits.append(f"Precheck found {len(diagnostics['blocking_errors'])} blocking issue(s).")
        status_bits.extend(diagnostics["blocking_errors"])
    else:
        status_bits.append("Precheck passed with no blocking issues.")
    if diagnostics.get("warnings"):
        status_bits.append(f"Warnings: {len(diagnostics['warnings'])}")
        status_bits.extend(diagnostics["warnings"])


def _run_on_image(image_path: str, mode: str) -> dict:
    from main import ChemEagle, ChemEagle_OS

    if mode == "cloud":
        return ChemEagle(image_path)
    return ChemEagle_OS(image_path)


def _run_on_image_cpu_subprocess(image_path: str, mode: str) -> dict:
    script = (
        "import json\n"
        "import traceback\n"
        "from main import ChemEagle, ChemEagle_OS\n"
        "fn = ChemEagle if " + repr(mode == "cloud") + " else ChemEagle_OS\n"
        "try:\n"
        "    result = fn(" + repr(image_path) + ")\n"
        "    print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': str(e), 'traceback': traceback.format_exc()}, ensure_ascii=False))\n"
        "    raise\n"
    )
    env = dict(os.environ)
    env["CHEMEAGLE_DEVICE"] = "cpu"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parent),
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"CPU retry produced no output. stderr: {proc.stderr.strip()}")

    payload = json.loads(lines[-1])
    if not isinstance(payload, dict) or not payload.get("ok"):
        error = payload.get("error") if isinstance(payload, dict) else "Unknown CPU retry failure"
        tb = payload.get("traceback", "") if isinstance(payload, dict) else ""
        stderr = proc.stderr.strip()
        raise RuntimeError(f"CPU retry failed: {error}\n{tb}\nSubprocess stderr:\n{stderr}")
    return payload["result"]


def _release_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


def _run_on_pdf(
    pdf_path: str,
    mode: str,
    pdf_model_size: str,
    *,
    persist_images: bool = False,
    persist_dir: str = "",
) -> Tuple[List[dict], str]:
    from pdf_extraction import run_pdf

    persisted_dir = ""
    with tempfile.TemporaryDirectory(prefix="chemeagle_pdf_") as tmpdir:
        run_pdf(pdf_dir=pdf_path, image_dir=tmpdir, model_size=pdf_model_size)
        if persist_images:
            target_dir = _resolve_pdf_persist_dir(pdf_path, persist_dir)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmpdir, target_dir, dirs_exist_ok=True)
            persisted_dir = str(target_dir.resolve())
        results: List[dict] = []
        for fname in sorted(os.listdir(tmpdir)):
            if not fname.lower().endswith(".png"):
                continue
            img_path = os.path.join(tmpdir, fname)
            try:
                result = _run_on_image(img_path, mode)
                result["image_name"] = fname
                results.append(result)
            except Exception as exc:
                results.append({"image_name": fname, "error": str(exc)})
        return results, persisted_dir


def _refresh_model_catalog(scope: str, current_model: str, values: Dict[str, str]):
    try:
        profile = resolve_llm_profile(scope=scope, values=values, default_model="gpt-5-mini")
    except Exception as exc:
        return gr.update(choices=_model_choices(current_model, []), value=current_model), f"Could not resolve the {scope} profile: {exc}"

    if scope == "ocr" and getattr(profile, "inherit_main", False):
        return (
            gr.update(choices=_model_choices(current_model, []), value=current_model),
            "LLM vision OCR is inheriting the main profile. Disable inherit to fetch a separate catalog.",
        )

    guard = _model_catalog_guard(profile)
    if guard:
        return gr.update(choices=_model_choices(current_model, []), value=current_model), guard

    try:
        catalog = list_available_models(profile)
    except Exception as exc:
        return gr.update(choices=_model_choices(current_model, []), value=current_model), f"Model refresh failed: {exc}"

    fetched_ids = [item.id for item in catalog.models]
    selected_value = current_model or (fetched_ids[0] if fetched_ids else "")
    status = catalog.status
    if current_model and current_model not in fetched_ids:
        status += " Current value was kept even though it was not present in the fetched catalog."
    return gr.update(choices=_model_choices(selected_value, fetched_ids), value=selected_value), status


def refresh_main_models(
    env_path_str: str,
    mode: str,
    chemeagle_device: str,
    save_env: bool,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
):
    values = _build_values_from_form(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )
    return _refresh_model_catalog("main", llm_model, values)


def refresh_ocr_models(
    env_path_str: str,
    mode: str,
    chemeagle_device: str,
    save_env: bool,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
):
    values = _build_values_from_form(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )
    return _refresh_model_catalog("ocr", ocr_llm_model, values)


def run_preflight(
    env_path_str: str,
    mode: str,
    chemeagle_device: str,
    upload,
    save_env: bool,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
) -> Tuple[str, str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
    values = _build_values_from_form(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )
    status_bits = [f"Runtime env applied from form. mode={mode}"]
    apply_runtime_env(values)

    if save_env:
        status_bits.append(save_env_file(env_path, values))

    file_path = _resolve_upload_path(upload)
    include_pdf_section = not file_path or Path(file_path).suffix.lower() == ".pdf"
    diagnostics = collect_preflight_diagnostics(file_path, mode, values, include_pdf_section=include_pdf_section)
    _append_preflight_status(status_bits, diagnostics)
    if file_path:
        status_bits.append(f"Prechecked file: {Path(file_path).name}")

    return "\n".join(status_bits), "{}", _json_text(diagnostics)


def run_pipeline(
    env_path_str: str,
    mode: str,
    chemeagle_device: str,
    upload,
    save_env: bool,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
) -> Tuple[str, str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
    values = _build_values_from_form(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )
    status_bits = [f"Runtime env applied from form. mode={mode}"]

    provider_norm = (llm_provider or "").strip().lower()
    if provider_norm in {"openai", "openai_compatible", "lmstudio", "local_openai"}:
        if not values.get("OPENAI_API_KEY") and values.get("API_KEY"):
            values["OPENAI_API_KEY"] = values["API_KEY"]
            status_bits.append("OPENAI_API_KEY was empty; using API_KEY as fallback for OpenAI-compatible provider.")
        if not values.get("OPENAI_API_KEY") and values.get("VLLM_API_KEY"):
            values["OPENAI_API_KEY"] = values["VLLM_API_KEY"]
            status_bits.append("OPENAI_API_KEY was empty; using VLLM_API_KEY as fallback.")

    apply_runtime_env(values)

    if save_env:
        status_bits.append(save_env_file(env_path, values))

    file_path = _resolve_upload_path(upload)
    if not file_path:
        return "\n".join(status_bits + ["No file uploaded."]), "{}", "{}"

    suffix = Path(file_path).suffix.lower()
    include_pdf_section = suffix == ".pdf"
    diagnostics = collect_preflight_diagnostics(file_path, mode, values, include_pdf_section=include_pdf_section)
    _append_preflight_status(status_bits, diagnostics)

    if diagnostics.get("blocking_errors"):
        status_bits.append("Aborting before execution because precheck failed.")
        return "\n".join(status_bits), "{}", _json_text(diagnostics)

    try:
        if suffix == ".pdf":
            result, persisted_pdf_dir = _run_on_pdf(
                file_path,
                mode,
                values.get("PDF_MODEL_SIZE") or "large",
                persist_images=_env_truthy(values.get("PDF_PERSIST_IMAGES")),
                persist_dir=values.get("PDF_PERSIST_DIR", ""),
            )
            if persisted_pdf_dir:
                status_bits.append(f"Persisted extracted PDF images to: {persisted_pdf_dir}")
        elif suffix in IMAGE_SUFFIXES:
            result = _run_on_image(file_path, mode)
        else:
            status_bits.append(f"Unsupported file type: {suffix}")
            return "\n".join(status_bits), "{}", _json_text(diagnostics)
    except Exception as exc:
        error_text = str(exc).lower()
        is_device_oom = any(
            msg in error_text
            for msg in ["cuda out of memory", "mps out of memory", "out of memory"]
        )
        if chemeagle_device in {"auto", "cuda", "metal", "mps"} and is_device_oom and suffix != ".pdf":
            status_bits.append("Device OOM detected; retrying once on CPU.")
            try:
                _release_gpu_memory()
                result = _run_on_image_cpu_subprocess(file_path, mode)
                status_bits.append(f"Completed for file: {Path(file_path).name}")
                return "\n".join(status_bits), _json_text(result), _json_text(diagnostics)
            except Exception as cpu_exc:
                exc = cpu_exc
        tb = traceback.format_exc()
        error_payload = {
            "error": str(exc),
            "traceback": tb,
            "mode": mode,
            "file": Path(file_path).name,
        }
        status_bits.extend([f"Pipeline failed: {exc}", "", "Traceback:", tb])
        return "\n".join(status_bits), _json_text(error_payload), _json_text(diagnostics)
    finally:
        _release_gpu_memory()

    status_bits.append(f"Completed for file: {Path(file_path).name}")
    return "\n".join(status_bits), _json_text(result), _json_text(diagnostics)


def submit_live_dataset(
    env_path_str: str,
    mode: str,
    chemeagle_device: str,
    save_env: bool,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
    artifact_backend: str,
    artifact_filesystem_root: str,
    artifact_s3_endpoint_url: str,
    artifact_s3_access_key_id: str,
    artifact_s3_secret_access_key: str,
    artifact_s3_bucket: str,
    artifact_s3_region: str,
    artifact_s3_use_ssl: bool,
    artifact_s3_key_prefix: str,
    review_db_path: str,
    experiment_name: str,
    experiment_notes: str,
    batch_folder_path: str,
    batch_upload,
    comparison_profiles_json: str,
) -> Tuple[str, str, str, str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
    values = _build_dataset_runtime_values(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
        artifact_backend,
        artifact_filesystem_root,
        artifact_s3_endpoint_url,
        artifact_s3_access_key_id,
        artifact_s3_secret_access_key,
        artifact_s3_bucket,
        artifact_s3_region,
        artifact_s3_use_ssl,
        artifact_s3_key_prefix,
        review_db_path,
    )
    apply_runtime_env(values)
    if save_env:
        save_env_file(env_path, values)
    source_paths = _resolve_upload_paths(batch_upload)
    source_paths.extend(path for path in _scan_supported_files(batch_folder_path) if path not in source_paths)
    if not source_paths:
        return "No PDF/image sources found for batch ingest.", "[]", "{}", "", ""
    try:
        profile_configs = _parse_profile_configs(values, comparison_profiles_json)
    except json.JSONDecodeError as exc:
        return f"Invalid comparison profile JSON: {exc}", "[]", "{}", "", ""
    diagnostics = _collect_batch_runtime_diagnostics(
        base_values=values,
        profile_configs=profile_configs,
        source_paths=source_paths,
        mode=mode,
    )
    if diagnostics["blocking_errors"]:
        return (
            f"Batch preflight failed with {len(diagnostics['blocking_errors'])} blocking issue(s).",
            "[]",
            _json_text(diagnostics),
            "",
            "",
        )
    preflight_summary = f"Runtime provider preflight {diagnostics['runtime_provider_preflight']['status']}."
    for config in profile_configs:
        config["preflight_status"] = diagnostics["runtime_provider_preflight"]["status"]
        config["preflight_summary"] = preflight_summary
    service = get_review_service(review_db_path)
    result = service.submit_live_experiment(
        experiment_name=experiment_name or "Live Experiment",
        notes=experiment_notes,
        source_paths=source_paths,
        profile_configs=profile_configs,
    )
    diagnostics["artifact_backend"] = artifact_backend
    diagnostics["review_db_path"] = review_db_path
    status_text = f"Queued {len(result['run_ids'])} run(s) in experiment {result['experiment_id']}."
    first_run_id = result["run_ids"][0] if result.get("run_ids") else ""
    return status_text, _json_text(result), _json_text(diagnostics), result["experiment_id"], first_run_id


def submit_sideload_dataset(
    env_path_str: str,
    mode: str,
    chemeagle_device: str,
    save_env: bool,
    llm_provider: str,
    llm_model: str,
    api_key: str,
    azure_endpoint: str,
    api_version: str,
    openai_api_key: str,
    openai_base_url: str,
    anthropic_api_key: str,
    vllm_base_url: str,
    vllm_api_key: str,
    ocr_backend: str,
    ocr_llm_inherit_main: bool,
    ocr_llm_provider: str,
    ocr_llm_model: str,
    ocr_api_key: str,
    ocr_azure_endpoint: str,
    ocr_api_version: str,
    ocr_openai_api_key: str,
    ocr_openai_base_url: str,
    ocr_anthropic_api_key: str,
    ocr_vllm_base_url: str,
    ocr_vllm_api_key: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
    artifact_backend: str,
    artifact_filesystem_root: str,
    artifact_s3_endpoint_url: str,
    artifact_s3_access_key_id: str,
    artifact_s3_secret_access_key: str,
    artifact_s3_bucket: str,
    artifact_s3_region: str,
    artifact_s3_use_ssl: bool,
    artifact_s3_key_prefix: str,
    review_db_path: str,
    experiment_name: str,
    experiment_notes: str,
    sideload_upload,
    recovery_roots_text: str,
) -> Tuple[str, str, str, str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
    values = _build_dataset_runtime_values(
        mode,
        llm_provider,
        llm_model,
        api_key,
        azure_endpoint,
        api_version,
        openai_api_key,
        openai_base_url,
        anthropic_api_key,
        vllm_base_url,
        vllm_api_key,
        chemeagle_device,
        ocr_backend,
        ocr_llm_inherit_main,
        ocr_llm_provider,
        ocr_llm_model,
        ocr_api_key,
        ocr_azure_endpoint,
        ocr_api_version,
        ocr_openai_api_key,
        ocr_openai_base_url,
        ocr_anthropic_api_key,
        ocr_vllm_base_url,
        ocr_vllm_api_key,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
        artifact_backend,
        artifact_filesystem_root,
        artifact_s3_endpoint_url,
        artifact_s3_access_key_id,
        artifact_s3_secret_access_key,
        artifact_s3_bucket,
        artifact_s3_region,
        artifact_s3_use_ssl,
        artifact_s3_key_prefix,
        review_db_path,
    )
    apply_runtime_env(values)
    if save_env:
        save_env_file(env_path, values)
    json_paths = [path for path in _resolve_upload_paths(sideload_upload) if path.lower().endswith(".json")]
    if not json_paths:
        return "No sideload JSON files selected.", "[]", "{}", "", ""
    recovery_roots = _parse_newline_paths(recovery_roots_text)
    service = get_review_service(review_db_path)
    result = service.submit_sideload_experiment(
        experiment_name=experiment_name or "Sideload Experiment",
        notes=experiment_notes,
        json_paths=json_paths,
        recovery_roots=recovery_roots,
        config_snapshot=values,
    )
    diagnostics = {
        "json_paths": json_paths,
        "recovery_roots": recovery_roots,
        "artifact_backend": artifact_backend,
        "review_db_path": review_db_path,
    }
    status_text = f"Queued {len(result['run_ids'])} sideload run(s) in experiment {result['experiment_id']}."
    first_run_id = result["run_ids"][0] if result.get("run_ids") else ""
    return status_text, _json_text(result), _json_text(diagnostics), result["experiment_id"], first_run_id


def refresh_ingest_monitor(
    experiment_id: str,
    run_id_hint: str,
    review_db_path: str,
    tail_lines: int,
    min_level: str,
    raw_logs: bool,
    show_suppressed: bool,
) -> Tuple[str, str, str, str, str, str]:
    selected_experiment_id = (experiment_id or "").strip()
    if not selected_experiment_id:
        return _render_progress_monitor({"progress": {}, "run": {}}), "No live ingest experiment selected.", "{}", "", "", ""
    service = get_review_service(review_db_path)
    runs = service.list_runs(experiment_id=selected_experiment_id)
    if not runs:
        return _render_progress_monitor({"progress": {}, "run": {}}), f"No runs found for experiment {selected_experiment_id}.", "{}", "", "", ""

    run_choice = next((row for row in runs if row["status"] == "running"), None)
    if run_choice is None:
        run_choice = next((row for row in runs if row["status"] == "queued"), None)
    if run_choice is None and run_id_hint:
        run_choice = next((row for row in runs if row["run_id"] == run_id_hint), None)
    if run_choice is None:
        run_choice = runs[0]

    monitor = service.get_run_monitor(
        run_choice["run_id"],
        tail_lines=int(tail_lines or 150),
        min_level=min_level or "INFO",
        raw=bool(raw_logs),
        include_suppressed=bool(show_suppressed),
    )
    run_row = monitor["run"]
    progress = monitor.get("progress", {}) or {}
    log_tail = monitor["log_tail"]
    log_tail_text = log_tail["raw"] if raw_logs else log_tail["formatted"]
    log_download_ref = service.get_log_download_ref(run_choice["run_id"])
    status_text = str(progress.get("status_summary") or f"Run {run_row.get('run_id', '')} is {run_row.get('status', '')}.")
    monitor_json = _json_text(
        {
            "run": run_row,
            "progress": progress,
            "sources": monitor.get("sources", []),
            "log_tail_events": log_tail.get("events", []),
        }
    )
    return (
        _render_progress_monitor(monitor),
        status_text,
        monitor_json,
        log_tail_text,
        log_download_ref,
        str(run_row.get("run_id") or ""),
    )


def refresh_runs_view(
    selected_experiment_id: str,
    selected_run_id: str,
    selected_run_source_id: str,
    selected_derived_image_id: str,
    review_db_path: str,
    tail_lines: int,
    min_level: str,
    raw_logs: bool,
    show_suppressed: bool,
) -> Tuple[str, Any, pd.DataFrame, Any, str, str, str, str, pd.DataFrame, Any, pd.DataFrame, pd.DataFrame, Any, pd.DataFrame, str, str]:
    service = get_review_service(review_db_path)
    experiments = service.list_experiments()
    experiment_choices = [ALL_EXPERIMENTS] + [row["experiment_id"] for row in experiments]
    experiment_dropdown_choices = [("All experiments", ALL_EXPERIMENTS)] + [
        (f"{row['name']} ({row['experiment_id']})", row["experiment_id"]) for row in experiments
    ]
    experiment_value = selected_experiment_id if selected_experiment_id in experiment_choices else ALL_EXPERIMENTS
    runs = service.list_runs(experiment_id="" if experiment_value == ALL_EXPERIMENTS else (experiment_value or ""))
    run_choices = [row["run_id"] for row in runs]
    run_value = selected_run_id if selected_run_id in run_choices else (run_choices[0] if run_choices else None)

    monitor_status = "No run selected."
    monitor_progress_html = _render_progress_monitor({"progress": {}, "run": {}})
    monitor_json = "{}"
    source_rows: List[Dict[str, Any]] = []
    source_choices: List[str] = []
    source_value: Optional[str] = None
    derived_rows: List[Dict[str, Any]] = []
    retry_candidates: List[Dict[str, Any]] = []
    derived_choices: List[str] = []
    derived_value: Optional[str] = None
    attempt_rows: List[Dict[str, Any]] = []
    log_tail_text = ""
    log_download_ref = ""

    if run_value:
        monitor = service.get_run_monitor(
            run_value,
            tail_lines=int(tail_lines or 200),
            min_level=min_level or "INFO",
            raw=bool(raw_logs),
            include_suppressed=bool(show_suppressed),
        )
        run_row = monitor["run"]
        progress = monitor.get("progress", {}) or {}
        source_rows = monitor["sources"]
        source_choices = [row["run_source_id"] for row in source_rows]
        preferred_source = selected_run_source_id or str(run_row.get("current_run_source_id") or "")
        source_value = preferred_source if preferred_source in source_choices else (source_choices[0] if source_choices else None)
        if source_value:
            source_detail = service.get_run_source_monitor(source_value)
            derived_rows = source_detail.get("derived_images", [])
            derived_choices = [row["derived_image_id"] for row in derived_rows]
            derived_value = (
                selected_derived_image_id
                if selected_derived_image_id in derived_choices
                else (derived_choices[0] if derived_choices else None)
            )
            if derived_value:
                selected_derived = next((row for row in derived_rows if row.get("derived_image_id") == derived_value), None)
                if selected_derived is not None:
                    attempt_rows = list(selected_derived.get("attempts", []))
        retry_candidates = service.list_retry_candidates(run_value)
        log_tail = monitor["log_tail"]
        log_tail_text = log_tail["raw"] if raw_logs else log_tail["formatted"]
        log_download_ref = service.get_log_download_ref(run_value)
        monitor_progress_html = _render_progress_monitor(monitor)
        monitor_status = str(progress.get("status_summary") or f"Run {run_value} is {run_row.get('status', '')}.")
        monitor_json = _json_text(
            {
                "run": run_row,
                "progress": progress,
                "sources": source_rows,
                "log_tail_events": log_tail.get("events", []),
            }
        )

    status_text = f"Loaded {len(experiments)} experiment(s) and {len(runs)} run(s)."
    return (
        status_text,
        gr.update(choices=experiment_dropdown_choices, value=experiment_value),
        _runs_dataframe(runs),
        gr.update(choices=run_choices, value=run_value),
        _json_text(experiments),
        monitor_progress_html,
        monitor_status,
        monitor_json,
        _run_sources_dataframe(source_rows),
        gr.update(choices=source_choices, value=source_value),
        _derived_images_dataframe(derived_rows),
        _retry_candidates_dataframe(retry_candidates),
        gr.update(choices=derived_choices, value=derived_value),
        _attempts_dataframe(attempt_rows),
        log_tail_text,
        log_download_ref,
    )


def refresh_runs_overview(selected_experiment_id: str, review_db_path: str) -> Tuple[str, Any, pd.DataFrame, Any, Any, str]:
    service = get_review_service(review_db_path)
    experiments = service.list_experiments()
    experiment_choices = [ALL_EXPERIMENTS] + [row["experiment_id"] for row in experiments]
    experiment_dropdown_choices = [("All experiments", ALL_EXPERIMENTS)] + [
        (f"{row['name']} ({row['experiment_id']})", row["experiment_id"]) for row in experiments
    ]
    experiment_value = selected_experiment_id if selected_experiment_id in experiment_choices else ALL_EXPERIMENTS
    runs = service.list_runs(experiment_id="" if experiment_value == ALL_EXPERIMENTS else (experiment_value or ""))
    run_choices = [row["run_id"] for row in runs]
    run_value = run_choices[0] if run_choices else None
    status_text = f"Loaded {len(experiments)} experiment(s) and {len(runs)} run(s)."
    return (
        status_text,
        gr.update(choices=experiment_dropdown_choices, value=experiment_value),
        _runs_dataframe(runs),
        gr.update(choices=run_choices, value=run_value),
        gr.update(choices=run_choices, value=run_value),
        _json_text(experiments),
    )


def retry_run_candidates_view(run_id: str, review_db_path: str, include_failed: bool, include_redo: bool) -> str:
    if not run_id:
        return "No run selected."
    retried = get_review_service(review_db_path).retry_failed_derived_images(
        run_id,
        include_needs_redo=include_redo,
        include_failed=include_failed,
    )
    return f"Retried {len(retried)} derived image(s) for run {run_id}."


def retry_failed_only_view(run_id: str, review_db_path: str) -> str:
    return retry_run_candidates_view(run_id, review_db_path, True, False)


def retry_redo_only_view(run_id: str, review_db_path: str) -> str:
    return retry_run_candidates_view(run_id, review_db_path, False, True)


def reprocess_run_view(run_id: str, review_db_path: str) -> str:
    if not run_id:
        return "No run selected."
    summary = get_review_service(review_db_path).reprocess_normalization_for_run(run_id, only_invalid_reactions=False)
    return (
        f"Reprocessed normalization for run {run_id}: "
        f"{summary.get('derived_images', 0)} derived image(s), "
        f"{summary.get('accepted_reactions', 0)} accepted reaction(s)."
    )


def retry_selected_derived_image_view(derived_image_id: str, retry_mode: str, review_db_path: str) -> str:
    if not derived_image_id:
        return "No derived image selected."
    execution_mode = retry_mode or "normal"
    trigger = "manual_no_agents_retry" if execution_mode == "no_agents" else "manual_retry"
    return get_review_service(review_db_path).retry_derived_image(
        derived_image_id,
        trigger=trigger,
        execution_mode=execution_mode,
    )


def reprocess_selected_derived_image_view(derived_image_id: str, review_db_path: str) -> str:
    if not derived_image_id:
        return "No derived image selected."
    summary = get_review_service(review_db_path).reprocess_normalization_for_derived_images([derived_image_id], purge_existing=True)
    return (
        f"Reprocessed normalization for {derived_image_id}: "
        f"{summary.get('accepted_reactions', 0)} accepted / {summary.get('rejected_reactions', 0)} rejected."
    )


def load_runs_for_experiment(selected_experiment_id: str, review_db_path: str) -> Tuple[str, pd.DataFrame, Any, Any]:
    experiment_filter = "" if selected_experiment_id in {"", ALL_EXPERIMENTS} else selected_experiment_id
    runs = get_review_service(review_db_path).list_runs(experiment_id=experiment_filter)
    run_choices = [row["run_id"] for row in runs]
    status_text = f"Loaded {len(runs)} run(s) for experiment {selected_experiment_id or 'all'}."
    run_update = gr.update(choices=run_choices, value=(run_choices[0] if run_choices else None))
    return status_text, _runs_dataframe(runs), run_update, run_update


def export_selected_run(run_id: str, export_dir: str, review_db_path: str) -> Tuple[str, str]:
    if not run_id:
        return "No run selected for export.", "{}"
    target_dir = export_dir or str((Path("./data/exports")).resolve())
    exported = get_review_service(review_db_path).export_run_to_parquet(run_id, target_dir)
    return f"Exported parquet files for run {run_id}.", _json_text(exported)


def refresh_review_list(selected_run_id: str, review_status_filter: str, outcome_class_filter: str, review_db_path: str) -> Tuple[str, pd.DataFrame, Any]:
    service = get_review_service(review_db_path)
    reactions = service.list_reactions(
        run_id=selected_run_id or "",
        review_status="" if review_status_filter == "all" else review_status_filter,
        outcome_class="" if outcome_class_filter == "all" else outcome_class_filter,
    )
    reaction_choices = [row["reaction_uid"] for row in reactions]
    status_text = f"Loaded {len(reactions)} reaction row(s)."
    if not reactions and selected_run_id:
        run_row = next((row for row in service.list_runs() if row.get("run_id") == selected_run_id), None)
        if run_row is not None:
            status_text = (
                f"Loaded 0 reaction row(s). "
                f"Run {selected_run_id} has {run_row.get('total_derived_images', 0)} derived image(s), "
                f"{run_row.get('total_redo', 0)} redo item(s), and {run_row.get('total_failures', 0)} failure(s). "
                f"This means the pipeline produced extraction candidates, but none passed canonical normalization yet. "
                f"Use Runs > Retry or Reprocess."
            )
    return status_text, _reactions_dataframe(reactions), gr.update(choices=reaction_choices, value=(reaction_choices[0] if reaction_choices else None))


def refresh_review_view(
    selected_run_id: str,
    review_status_filter: str,
    outcome_class_filter: str,
    review_db_path: str,
) -> Tuple[str, Any, pd.DataFrame, Any]:
    service = get_review_service(review_db_path)
    runs = service.list_runs()
    run_choices = _review_run_choices(runs)
    valid_run_ids = {value for _, value in run_choices}
    run_value = selected_run_id if selected_run_id in valid_run_ids else (run_choices[0][1] if run_choices else None)
    status_text, table, reaction_update = refresh_review_list(
        run_value or "",
        review_status_filter,
        outcome_class_filter,
        review_db_path,
    )
    return status_text, gr.update(choices=run_choices, value=run_value), table, reaction_update


def load_reaction_detail_view(
    reaction_uid: str,
    artifact_backend: str,
    artifact_filesystem_root: str,
    artifact_s3_endpoint_url: str,
    artifact_s3_access_key_id: str,
    artifact_s3_secret_access_key: str,
    artifact_s3_bucket: str,
    artifact_s3_region: str,
    artifact_s3_use_ssl: bool,
    artifact_s3_key_prefix: str,
    review_db_path: str,
) -> Tuple[str, str, Optional[str], Optional[str], str, str, str, str]:
    if not reaction_uid:
        return "No reaction selected.", "", None, None, "{}", "{}", "unchecked", ""
    values = {
        "ARTIFACT_BACKEND": artifact_backend,
        "ARTIFACT_FILESYSTEM_ROOT": artifact_filesystem_root,
        "ARTIFACT_S3_ENDPOINT_URL": artifact_s3_endpoint_url,
        "ARTIFACT_S3_ACCESS_KEY_ID": artifact_s3_access_key_id,
        "ARTIFACT_S3_SECRET_ACCESS_KEY": artifact_s3_secret_access_key,
        "ARTIFACT_S3_BUCKET": artifact_s3_bucket,
        "ARTIFACT_S3_REGION": artifact_s3_region,
        "ARTIFACT_S3_USE_SSL": "1" if artifact_s3_use_ssl else "0",
        "ARTIFACT_S3_KEY_PREFIX": artifact_s3_key_prefix,
        "REVIEW_DB_PATH": review_db_path,
    }
    detail = get_review_service(review_db_path).get_reaction_detail(reaction_uid)
    source_link = _artifact_download_ref(values, detail.get("source_artifact_backend", ""), detail.get("source_artifact_key", ""))
    derived_preview = _artifact_preview_path(values, detail.get("derived_backend", ""), detail.get("derived_artifact_key", ""), suffix=".png")
    render_preview = _artifact_preview_path(values, artifact_backend or detail.get("derived_backend", ""), detail.get("render_artifact_key", ""), suffix=".png")
    reaction_payload = {
        "reaction_uid": detail["reaction_uid"],
        "reaction_id": detail["reaction_id"],
        "profile_label": detail.get("profile_label", ""),
        "source": detail.get("original_filename", ""),
        "raw_reaction_json": json.loads(detail["raw_reaction_json"]),
    }
    return (
        f"Loaded reaction {reaction_uid}.",
        source_link,
        derived_preview,
        render_preview,
        _json_text(reaction_payload),
        _json_text({"conditions": detail.get("conditions", []), "additional_info": detail.get("additional_info", [])}),
        detail.get("review_status", "unchecked"),
        detail.get("review_notes", ""),
    )


def save_reaction_review_view(reaction_uid: str, review_status: str, review_notes: str, review_db_path: str) -> Tuple[str, str]:
    if not reaction_uid:
        return "No reaction selected.", "{}"
    get_review_service(review_db_path).update_reaction_review(reaction_uid, review_status=review_status, review_notes=review_notes)
    return f"Saved review state for {reaction_uid}.", _json_text({"reaction_uid": reaction_uid, "review_status": review_status, "review_notes": review_notes})


def build_app() -> gr.Blocks:
    vals = merged_env_values(ENV_FILE_DEFAULT)
    initial_main_provider = vals.get("LLM_PROVIDER", "azure") or "azure"
    initial_main_model = vals.get("LLM_MODEL", "gpt-5-mini") or "gpt-5-mini"
    initial_ocr_inherit = _env_truthy(vals.get("OCR_LLM_INHERIT_MAIN", "1") or "1")
    initial_ocr_provider = vals.get("OCR_LLM_PROVIDER", initial_main_provider) or initial_main_provider
    initial_ocr_model = vals.get("OCR_LLM_MODEL", "")
    initial_main_model_status = _model_picker_help(initial_main_provider)
    if initial_ocr_inherit:
        initial_ocr_model_status = "LLM vision OCR is inheriting the main profile. Disable inherit to pick a separate provider/model."
    else:
        initial_ocr_model_status = _model_picker_help(initial_ocr_provider)
    initial_ocr_summary = _ocr_profile_summary(
        initial_ocr_inherit,
        initial_main_provider,
        initial_main_model,
        initial_ocr_provider,
        initial_ocr_model,
    )
    override_interactive = not initial_ocr_inherit
    main_refresh_interactive = initial_main_provider.strip().lower() not in MANUAL_MODEL_LIST_PROVIDERS
    ocr_refresh_interactive = override_interactive and initial_ocr_provider.strip().lower() not in MANUAL_MODEL_LIST_PROVIDERS
    initial_artifact_backend = vals.get("ARTIFACT_BACKEND", "filesystem") or "filesystem"
    initial_artifact_fs_root = vals.get("ARTIFACT_FILESYSTEM_ROOT") or DEFAULT_ARTIFACT_FILESYSTEM_ROOT
    initial_review_db_path = vals.get("REVIEW_DB_PATH") or DEFAULT_REVIEW_DB_PATH
    initial_s3_endpoint_url = vals.get("ARTIFACT_S3_ENDPOINT_URL") or DEFAULT_MINIO_ENDPOINT
    initial_s3_bucket = vals.get("ARTIFACT_S3_BUCKET") or DEFAULT_MINIO_BUCKET

    with gr.Blocks(title="ChemEagle Self-Hosted GUI") as demo:
        gr.Markdown("# ChemEagle Self-Hosted GUI")
        gr.Markdown("Configure the run mode, provider profiles, OCR backend, PDF extraction settings, and storage before launching runs and review datasets.")

        with gr.Accordion("Setting", open=True):
            gr.Markdown("Global configuration shared by batch runs and dataset workflows.")
            with gr.Row():
                env_path = gr.Textbox(label="Env file path", value=str(ENV_FILE_DEFAULT), scale=3)
                env_path_picker = gr.FileExplorer(
                    label="Pick env file",
                    root_dir=str(Path.home()),
                    file_count="single",
                    height=160,
                    scale=2,
                )
            with gr.Row():
                mode = gr.Radio(
                    ["cloud", "local_os"],
                    value=vals.get("CHEMEAGLE_RUN_MODE", "cloud") or "cloud",
                    label="Run mode",
                )
                chemeagle_device = gr.Radio(
                    ["auto", "cpu", "cuda", "metal"],
                    value=vals.get("CHEMEAGLE_DEVICE", "auto") or "auto",
                    label="Compute device",
                )
                save_env = gr.Checkbox(label="Save form values to env file", value=True)

            with gr.Tabs():
                with gr.Tab("LLM"):
                    gr.Markdown("Main provider profile used by the cloud planner, synthesis, and any tool-calling cloud agent.")
                    with gr.Row():
                        llm_provider = gr.Dropdown(
                            choices=LLM_PROVIDER_CHOICES,
                            value=initial_main_provider,
                            allow_custom_value=True,
                            label="LLM provider",
                            scale=1,
                        )
                        llm_model = gr.Dropdown(
                            choices=_model_choices(initial_main_model, []),
                            value=initial_main_model,
                            allow_custom_value=True,
                            label="LLM_MODEL",
                            scale=2,
                        )
                        refresh_main_btn = gr.Button("Refresh Models", interactive=main_refresh_interactive, scale=1)
                    llm_model_status = gr.Textbox(
                        label="Main model catalog",
                        value=initial_main_model_status,
                        interactive=False,
                        lines=2,
                    )
                    with gr.Row():
                        api_version = gr.Textbox(label="API_VERSION", value=vals.get("API_VERSION", "2024-06-01"))
                    with gr.Row():
                        api_key = gr.Textbox(label="API_KEY / Azure key", type="password", value=vals.get("API_KEY", ""))
                        azure_endpoint = gr.Textbox(label="AZURE_ENDPOINT", value=vals.get("AZURE_ENDPOINT", ""))
                    with gr.Row():
                        openai_api_key = gr.Textbox(label="OPENAI_API_KEY", type="password", value=vals.get("OPENAI_API_KEY", ""))
                        openai_base_url = gr.Textbox(label="OPENAI_BASE_URL", value=vals.get("OPENAI_BASE_URL", ""))
                    with gr.Row():
                        vllm_api_key = gr.Textbox(label="VLLM_API_KEY", type="password", value=vals.get("VLLM_API_KEY", "EMPTY"))
                        vllm_base_url = gr.Textbox(label="VLLM_BASE_URL", value=vals.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
                    with gr.Row():
                        anthropic_api_key = gr.Textbox(label="ANTHROPIC_API_KEY", type="password", value=vals.get("ANTHROPIC_API_KEY", ""))

                with gr.Tab("OCR"):
                    gr.Markdown("`auto` resolves to `llm_vision` in cloud mode and `easyocr` in local_os mode. `tesseract` stays available as an explicit fallback.")
                    with gr.Row():
                        ocr_backend = gr.Dropdown(
                            choices=OCR_BACKEND_CHOICES,
                            value=vals.get("OCR_BACKEND", "auto") or "auto",
                            label="OCR backend",
                            scale=1,
                        )
                        ocr_llm_inherit_main = gr.Checkbox(
                            label="Use main LLM config for LLM vision",
                            value=initial_ocr_inherit,
                            scale=1,
                        )
                    ocr_profile_summary = gr.Textbox(
                        label="Effective LLM vision profile",
                        value=initial_ocr_summary,
                        interactive=False,
                        lines=2,
                    )
                    with gr.Row():
                        ocr_llm_provider = gr.Dropdown(
                            choices=LLM_PROVIDER_CHOICES,
                            value=initial_ocr_provider,
                            allow_custom_value=True,
                            label="OCR_LLM_PROVIDER",
                            interactive=override_interactive,
                            scale=1,
                        )
                        ocr_llm_model = gr.Dropdown(
                            choices=_model_choices(initial_ocr_model, []),
                            value=initial_ocr_model or None,
                            allow_custom_value=True,
                            label="OCR_LLM_MODEL",
                            interactive=override_interactive,
                            scale=2,
                        )
                        refresh_ocr_btn = gr.Button("Refresh Vision Models", interactive=ocr_refresh_interactive, scale=1)
                    ocr_model_status = gr.Textbox(
                        label="Vision model catalog",
                        value=initial_ocr_model_status,
                        interactive=False,
                        lines=2,
                    )
                    with gr.Accordion("Vision LLM override credentials", open=False):
                        with gr.Row():
                            ocr_api_version = gr.Textbox(
                                label="OCR_API_VERSION",
                                value=vals.get("OCR_API_VERSION", vals.get("API_VERSION", "2024-06-01")),
                                interactive=override_interactive,
                            )
                        with gr.Row():
                            ocr_api_key = gr.Textbox(
                                label="OCR_API_KEY / OCR Azure key",
                                type="password",
                                value=vals.get("OCR_API_KEY", ""),
                                interactive=override_interactive,
                            )
                            ocr_azure_endpoint = gr.Textbox(
                                label="OCR_AZURE_ENDPOINT",
                                value=vals.get("OCR_AZURE_ENDPOINT", ""),
                                interactive=override_interactive,
                            )
                        with gr.Row():
                            ocr_openai_api_key = gr.Textbox(
                                label="OCR_OPENAI_API_KEY",
                                type="password",
                                value=vals.get("OCR_OPENAI_API_KEY", ""),
                                interactive=override_interactive,
                            )
                            ocr_openai_base_url = gr.Textbox(
                                label="OCR_OPENAI_BASE_URL",
                                value=vals.get("OCR_OPENAI_BASE_URL", ""),
                                interactive=override_interactive,
                            )
                        with gr.Row():
                            ocr_vllm_api_key = gr.Textbox(
                                label="OCR_VLLM_API_KEY",
                                type="password",
                                value=vals.get("OCR_VLLM_API_KEY", ""),
                                interactive=override_interactive,
                            )
                            ocr_vllm_base_url = gr.Textbox(
                                label="OCR_VLLM_BASE_URL",
                                value=vals.get("OCR_VLLM_BASE_URL", ""),
                                interactive=override_interactive,
                            )
                        with gr.Row():
                            ocr_anthropic_api_key = gr.Textbox(
                                label="OCR_ANTHROPIC_API_KEY",
                                type="password",
                                value=vals.get("OCR_ANTHROPIC_API_KEY", ""),
                                interactive=override_interactive,
                            )
                    with gr.Row():
                        ocr_lang = gr.Textbox(label="OCR_LANG", value=vals.get("OCR_LANG", "eng"))
                        ocr_config = gr.Textbox(label="OCR_CONFIG", value=vals.get("OCR_CONFIG", ""))
                    tesseract_cmd = gr.Textbox(
                        label="TESSERACT_CMD",
                        value=vals.get("TESSERACT_CMD", ""),
                        placeholder="Optional absolute path to tesseract",
                    )

                with gr.Tab("PDF"):
                    gr.Markdown("PDF runs use VisualHeist to crop figures/tables into temporary PNGs before sending them through the normal image pipeline.")
                    with gr.Row():
                        pdf_model_size = gr.Dropdown(
                            choices=PDF_MODEL_CHOICES,
                            value=vals.get("PDF_MODEL_SIZE", "large") or "large",
                            label="PDF_MODEL_SIZE",
                        )
                        pdf_persist_images = gr.Checkbox(
                            label="Persist extracted PDF images",
                            value=_env_truthy(vals.get("PDF_PERSIST_IMAGES", "")),
                        )
                    with gr.Row():
                        pdf_persist_dir = gr.Textbox(
                            label="PDF_PERSIST_DIR",
                            value=vals.get("PDF_PERSIST_DIR", ""),
                            placeholder="Optional folder for debug PNGs; defaults to ./debug/pdf_images/",
                            scale=3,
                        )
                        pdf_persist_dir_picker = gr.FileExplorer(
                            label="Pick PDF debug folder",
                            root_dir=str(Path.home()),
                            file_count="single",
                            height=160,
                            scale=2,
                        )

                with gr.Tab("Storage"):
                    gr.Markdown(
                        f"Default local storage uses `filesystem`. Leave MinIO fields empty unless `ARTIFACT_BACKEND=minio`. "
                        f"Defaults: `REVIEW_DB_PATH={DEFAULT_REVIEW_DB_PATH}` and `ARTIFACT_FILESYSTEM_ROOT={DEFAULT_ARTIFACT_FILESYSTEM_ROOT}`."
                    )
                    with gr.Row():
                        artifact_backend = gr.Dropdown(
                            choices=["filesystem", "minio"],
                            value=initial_artifact_backend,
                            label="ARTIFACT_BACKEND",
                        )
                        review_db_path = gr.Textbox(
                            label="REVIEW_DB_PATH",
                            value=initial_review_db_path,
                            scale=3,
                        )
                        review_db_dir_picker = gr.FileExplorer(
                            label="Pick DB folder",
                            root_dir=str(Path.home()),
                            file_count="single",
                            height=160,
                            scale=2,
                        )
                    with gr.Row():
                        artifact_filesystem_root = gr.Textbox(
                            label="ARTIFACT_FILESYSTEM_ROOT",
                            value=initial_artifact_fs_root,
                            scale=3,
                        )
                        artifact_root_picker = gr.FileExplorer(
                            label="Pick artifact folder",
                            root_dir=str(Path.home()),
                            file_count="single",
                            height=160,
                            scale=2,
                        )
                    with gr.Row():
                        artifact_s3_endpoint_url = gr.Textbox(
                            label="ARTIFACT_S3_ENDPOINT_URL",
                            value=initial_s3_endpoint_url,
                        )
                        artifact_s3_bucket = gr.Textbox(
                            label="ARTIFACT_S3_BUCKET",
                            value=initial_s3_bucket,
                        )
                    with gr.Row():
                        artifact_s3_access_key_id = gr.Textbox(
                            label="ARTIFACT_S3_ACCESS_KEY_ID",
                            type="password",
                            value=vals.get("ARTIFACT_S3_ACCESS_KEY_ID", ""),
                        )
                        artifact_s3_secret_access_key = gr.Textbox(
                            label="ARTIFACT_S3_SECRET_ACCESS_KEY",
                            type="password",
                            value=vals.get("ARTIFACT_S3_SECRET_ACCESS_KEY", ""),
                        )
                    with gr.Row():
                        artifact_s3_region = gr.Textbox(
                            label="ARTIFACT_S3_REGION",
                            value=vals.get("ARTIFACT_S3_REGION", ""),
                        )
                        artifact_s3_key_prefix = gr.Textbox(
                            label="ARTIFACT_S3_KEY_PREFIX",
                            value=vals.get("ARTIFACT_S3_KEY_PREFIX", ""),
                        )
                        artifact_s3_use_ssl = gr.Checkbox(
                            label="ARTIFACT_S3_USE_SSL",
                            value=_env_truthy(vals.get("ARTIFACT_S3_USE_SSL", "")),
                        )

        with gr.Accordion("Batch Run and Dataset Creation", open=False):
            gr.Markdown("Create comparison experiments, sideload older JSON runs, inspect run metrics, and review extracted reactions.")
            with gr.Tabs():
                with gr.Tab("Ingest"):
                    gr.Markdown("Queue new experiments from live files or folders, including a batch of one file, or sideload raw JSON from previous unstored runs.")
                    experiment_name = gr.Textbox(label="Experiment name", value="ChemEagle Review Experiment")
                    experiment_notes = gr.Textbox(label="Experiment notes", lines=2)
                    comparison_profiles_json = gr.Code(
                        label="Comparison profiles JSON",
                        language="json",
                        value='[\n  {"profile_label": "baseline"}\n]',
                    )
                    with gr.Row():
                        batch_folder_path = gr.Textbox(
                            label="Batch folder path",
                            placeholder="Optional folder to scan recursively for PDFs/images",
                            scale=3,
                        )
                        batch_folder_picker = gr.FileExplorer(
                            label="Pick batch folder",
                            root_dir=str(Path.home()),
                            file_count="single",
                            height=160,
                            scale=2,
                        )
                    batch_upload = gr.File(
                        label="Batch PDF/image upload",
                        file_count="multiple",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"],
                    )
                    queue_live_btn = gr.Button("Queue Live Experiment", variant="primary")
                    gr.Markdown("### Sideload")
                    sideload_upload = gr.File(
                        label="Sideload JSON files",
                        file_count="multiple",
                        file_types=[".json"],
                    )
                    with gr.Row():
                        recovery_roots_text = gr.Textbox(
                            label="Recovery roots (one path per line)",
                            lines=4,
                            placeholder="/path/to/old/output\n/path/to/pdf_images",
                            scale=3,
                        )
                        recovery_root_picker = gr.FileExplorer(
                            label="Add recovery root",
                            root_dir=str(Path.home()),
                            file_count="single",
                            height=160,
                            scale=2,
                        )
                    queue_sideload_btn = gr.Button("Queue Sideload Import")
                    ingest_status = gr.Textbox(label="Ingest status", lines=8)
                    with gr.Row():
                        ingest_output = gr.Code(label="Queued payload", language="json")
                        ingest_diagnostics = gr.Code(label="Ingest diagnostics", language="json")
                    gr.Markdown("### Live ingest monitor")
                    with gr.Row():
                        ingest_experiment_id = gr.Textbox(label="Current ingest experiment", interactive=False, visible=False)
                        ingest_run_id = gr.Textbox(label="Current ingest run", interactive=False, visible=False)
                    with gr.Row():
                        ingest_auto_refresh = gr.Checkbox(label="Auto refresh", value=True)
                    ingest_progress_html = gr.HTML()
                    ingest_monitor_status = gr.Textbox(label="Current status", lines=3)
                    with gr.Accordion("Troubleshooting Logs", open=False):
                        with gr.Row():
                            ingest_log_level = gr.Dropdown(
                                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                                value="INFO",
                                label="Log level",
                            )
                            ingest_log_raw = gr.Checkbox(label="Raw JSON logs", value=False)
                            ingest_show_noise = gr.Checkbox(label="Show suppressed library warnings", value=False)
                            ingest_tail_lines = gr.Slider(label="Tail lines", minimum=25, maximum=500, step=25, value=150)
                        ingest_monitor_json = gr.Code(label="Live ingest monitor JSON", language="json")
                        ingest_log_tail = gr.Textbox(label="Live ingest log tail", lines=12)
                        ingest_log_download_ref = gr.Textbox(label="Log download / local path", lines=2)
                    ingest_timer = gr.Timer(1.0, active=True)

                with gr.Tab("Runs"):
                    runs_status = gr.Textbox(label="Runs status", lines=3)
                    with gr.Row():
                        runs_refresh_btn = gr.Button("Refresh Runs")
                        runs_experiment_id = gr.Dropdown(label="Experiment")
                        runs_run_id = gr.Dropdown(label="Run")
                    with gr.Row():
                        runs_auto_refresh = gr.Checkbox(label="Auto refresh", value=True)
                    runs_table = gr.Dataframe(label="Runs", interactive=False)
                    runs_json = gr.Code(label="Experiments JSON", language="json")
                    runs_progress_html = gr.HTML()
                    runs_monitor_status = gr.Textbox(label="Current status", lines=3)
                    with gr.Accordion("Troubleshooting Logs", open=False):
                        with gr.Row():
                            runs_log_level = gr.Dropdown(
                                choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                                value="INFO",
                                label="Log level",
                            )
                            runs_log_raw = gr.Checkbox(label="Raw JSON logs", value=False)
                            runs_show_noise = gr.Checkbox(label="Show suppressed library warnings", value=False)
                            runs_tail_lines = gr.Slider(label="Tail lines", minimum=25, maximum=500, step=25, value=200)
                        runs_monitor_json = gr.Code(label="Run monitor JSON", language="json")
                        runs_log_tail = gr.Textbox(label="Live log tail", lines=16)
                        runs_log_download_ref = gr.Textbox(label="Log download / local path", lines=2)
                    runs_source_table = gr.Dataframe(label="Sources in selected run", interactive=False)
                    runs_source_id = gr.Dropdown(label="Source")
                    runs_derived_table = gr.Dataframe(label="Derived images for selected source", interactive=False)
                    runs_retry_candidates_table = gr.Dataframe(label="Retry / reprocess candidates", interactive=False)
                    with gr.Row():
                        retry_failed_btn = gr.Button("Retry failed derived images")
                        retry_redo_btn = gr.Button("Retry redo-only derived images")
                        reprocess_run_btn = gr.Button("Reprocess normalization for run")
                    runs_recovery_status = gr.Textbox(label="Recovery status", lines=2)
                    with gr.Row():
                        runs_derived_image_id = gr.Dropdown(label="Derived image")
                        runs_retry_mode = gr.Dropdown(
                            label="Retry mode",
                            choices=["normal", "no_agents", "recovery"],
                            value="normal",
                        )
                        retry_selected_btn = gr.Button("Retry selected derived image")
                        reprocess_selected_btn = gr.Button("Reprocess selected derived image")
                    runs_attempts_table = gr.Dataframe(label="Attempt history for selected derived image", interactive=False)
                    runs_timer = gr.Timer(1.0, active=True)
                    with gr.Row():
                        export_dir = gr.Textbox(label="Parquet export directory", value=DEFAULT_EXPORT_DIR, scale=3)
                        export_dir_picker = gr.FileExplorer(
                            label="Pick export folder",
                            root_dir=str(Path.home()),
                            file_count="single",
                            height=160,
                            scale=2,
                        )
                        export_btn = gr.Button("Export Selected Run")
                    export_status = gr.Textbox(label="Export status", lines=2)
                    export_json = gr.Code(label="Exported files", language="json")

                with gr.Tab("Review"):
                    review_status_box = gr.Textbox(label="Review status", lines=3)
                    with gr.Row():
                        review_run_id = gr.Dropdown(label="Run")
                        review_status_filter = gr.Dropdown(
                            choices=["all", "unchecked", "ok", "not_ok"],
                            value="all",
                            label="Review status filter",
                        )
                        review_outcome_filter = gr.Dropdown(
                            choices=["all", "succeeded", "empty", "failed", "needs_redo", "imported_without_artifact"],
                            value="all",
                            label="Outcome filter",
                        )
                        review_refresh_btn = gr.Button("Refresh Review List")
                    review_table = gr.Dataframe(label="Reactions", interactive=False)
                    review_reaction_uid = gr.Dropdown(label="Reaction")
                    review_detail_status = gr.Textbox(label="Reaction detail status", lines=2)
                    source_link = gr.Textbox(label="Source link")
                    with gr.Row():
                        derived_preview = gr.Image(label="Derived/source preview", type="filepath")
                        render_preview = gr.Image(label="RDKit reaction depiction", type="filepath")
                    review_reaction_json = gr.Code(label="Reaction JSON", language="json")
                    review_supporting_json = gr.Code(label="Conditions / additional info", language="json")
                    with gr.Row():
                        review_status_radio = gr.Radio(
                            ["unchecked", "ok", "not_ok"],
                            value="unchecked",
                            label="Review decision",
                        )
                        review_notes = gr.Textbox(label="Review notes", lines=3)
                    review_save_btn = gr.Button("Save Review Decision")
                    review_save_status = gr.Textbox(label="Review save status", lines=2)
                    review_save_json = gr.Code(label="Saved review payload", language="json")

        settings_inputs = [
            env_path,
            mode,
            chemeagle_device,
            save_env,
            llm_provider,
            llm_model,
            api_key,
            azure_endpoint,
            api_version,
            openai_api_key,
            openai_base_url,
            anthropic_api_key,
            vllm_base_url,
            vllm_api_key,
            ocr_backend,
            ocr_llm_inherit_main,
            ocr_llm_provider,
            ocr_llm_model,
            ocr_api_key,
            ocr_azure_endpoint,
            ocr_api_version,
            ocr_openai_api_key,
            ocr_openai_base_url,
            ocr_anthropic_api_key,
            ocr_vllm_base_url,
            ocr_vllm_api_key,
            ocr_lang,
            ocr_config,
            tesseract_cmd,
            pdf_model_size,
            pdf_persist_images,
            pdf_persist_dir,
        ]
        dataset_common_inputs = [
            env_path,
            mode,
            chemeagle_device,
            save_env,
            llm_provider,
            llm_model,
            api_key,
            azure_endpoint,
            api_version,
            openai_api_key,
            openai_base_url,
            anthropic_api_key,
            vllm_base_url,
            vllm_api_key,
            ocr_backend,
            ocr_llm_inherit_main,
            ocr_llm_provider,
            ocr_llm_model,
            ocr_api_key,
            ocr_azure_endpoint,
            ocr_api_version,
            ocr_openai_api_key,
            ocr_openai_base_url,
            ocr_anthropic_api_key,
            ocr_vllm_base_url,
            ocr_vllm_api_key,
            ocr_lang,
            ocr_config,
            tesseract_cmd,
            pdf_model_size,
            pdf_persist_images,
            pdf_persist_dir,
            artifact_backend,
            artifact_filesystem_root,
            artifact_s3_endpoint_url,
            artifact_s3_access_key_id,
            artifact_s3_secret_access_key,
            artifact_s3_bucket,
            artifact_s3_region,
            artifact_s3_use_ssl,
            artifact_s3_key_prefix,
            review_db_path,
        ]
        ocr_control_inputs = [ocr_llm_inherit_main, llm_provider, llm_model, ocr_llm_provider, ocr_llm_model]
        ocr_control_outputs = [
            ocr_llm_provider,
            ocr_llm_model,
            ocr_api_key,
            ocr_azure_endpoint,
            ocr_api_version,
            ocr_openai_api_key,
            ocr_openai_base_url,
            ocr_anthropic_api_key,
            ocr_vllm_base_url,
            ocr_vllm_api_key,
            refresh_ocr_btn,
            ocr_model_status,
            ocr_profile_summary,
        ]
        runs_view_inputs = [
            runs_experiment_id,
            runs_run_id,
            runs_source_id,
            runs_derived_image_id,
            review_db_path,
            runs_tail_lines,
            runs_log_level,
            runs_log_raw,
            runs_show_noise,
        ]
        runs_view_outputs = [
            runs_status,
            runs_experiment_id,
            runs_table,
            runs_run_id,
            runs_json,
            runs_progress_html,
            runs_monitor_status,
            runs_monitor_json,
            runs_source_table,
            runs_source_id,
            runs_derived_table,
            runs_retry_candidates_table,
            runs_derived_image_id,
            runs_attempts_table,
            runs_log_tail,
            runs_log_download_ref,
        ]
        ingest_monitor_inputs = [
            ingest_experiment_id,
            ingest_run_id,
            review_db_path,
            ingest_tail_lines,
            ingest_log_level,
            ingest_log_raw,
            ingest_show_noise,
        ]
        ingest_monitor_outputs = [
            ingest_progress_html,
            ingest_monitor_status,
            ingest_monitor_json,
            ingest_log_tail,
            ingest_log_download_ref,
            ingest_run_id,
        ]

        queue_live_event = queue_live_btn.click(
            fn=submit_live_dataset,
            inputs=dataset_common_inputs + [experiment_name, experiment_notes, batch_folder_path, batch_upload, comparison_profiles_json],
            outputs=[ingest_status, ingest_output, ingest_diagnostics, ingest_experiment_id, ingest_run_id],
        )
        queue_live_event.then(
            fn=refresh_ingest_monitor,
            inputs=ingest_monitor_inputs,
            outputs=ingest_monitor_outputs,
        )
        queue_sideload_event = queue_sideload_btn.click(
            fn=submit_sideload_dataset,
            inputs=dataset_common_inputs + [experiment_name, experiment_notes, sideload_upload, recovery_roots_text],
            outputs=[ingest_status, ingest_output, ingest_diagnostics, ingest_experiment_id, ingest_run_id],
        )
        queue_sideload_event.then(
            fn=refresh_ingest_monitor,
            inputs=ingest_monitor_inputs,
            outputs=ingest_monitor_outputs,
        )
        env_path_picker.change(
            fn=_replace_path_from_picker,
            inputs=[env_path_picker, env_path],
            outputs=[env_path],
        )
        pdf_persist_dir_picker.change(
            fn=_replace_directory_from_picker,
            inputs=[pdf_persist_dir_picker, pdf_persist_dir],
            outputs=[pdf_persist_dir],
        )
        review_db_dir_picker.change(
            fn=_replace_db_path_from_directory,
            inputs=[review_db_dir_picker, review_db_path],
            outputs=[review_db_path],
        )
        artifact_root_picker.change(
            fn=_replace_directory_from_picker,
            inputs=[artifact_root_picker, artifact_filesystem_root],
            outputs=[artifact_filesystem_root],
        )
        batch_folder_picker.change(
            fn=_replace_directory_from_picker,
            inputs=[batch_folder_picker, batch_folder_path],
            outputs=[batch_folder_path],
        )
        recovery_root_picker.change(
            fn=_append_path_from_picker,
            inputs=[recovery_roots_text, recovery_root_picker],
            outputs=[recovery_roots_text],
        )
        export_dir_picker.change(
            fn=_replace_directory_from_picker,
            inputs=[export_dir_picker, export_dir],
            outputs=[export_dir],
        )
        refresh_main_btn.click(
            fn=refresh_main_models,
            inputs=settings_inputs,
            outputs=[llm_model, llm_model_status],
        )
        refresh_ocr_btn.click(
            fn=refresh_ocr_models,
            inputs=settings_inputs,
            outputs=[ocr_llm_model, ocr_model_status],
        )
        llm_provider.change(
            fn=_main_model_picker_updates,
            inputs=[llm_provider],
            outputs=[refresh_main_btn, llm_model_status],
        )
        for component in [ocr_llm_inherit_main, llm_provider, llm_model, ocr_llm_provider, ocr_llm_model]:
            component.change(
                fn=_ocr_model_picker_updates,
                inputs=ocr_control_inputs,
                outputs=ocr_control_outputs,
            )
        runs_refresh_btn.click(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        runs_experiment_id.change(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        runs_run_id.change(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        runs_source_id.change(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        runs_derived_image_id.change(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        for component in [runs_log_level, runs_log_raw, runs_tail_lines, runs_show_noise]:
            component.change(
                fn=refresh_runs_view,
                inputs=runs_view_inputs,
                outputs=runs_view_outputs,
            )
        for component in [ingest_log_level, ingest_log_raw, ingest_tail_lines, ingest_show_noise]:
            component.change(
                fn=refresh_ingest_monitor,
                inputs=ingest_monitor_inputs,
                outputs=ingest_monitor_outputs,
            )
        runs_auto_refresh.change(
            fn=lambda enabled: gr.update(active=enabled),
            inputs=[runs_auto_refresh],
            outputs=[runs_timer],
        )
        ingest_auto_refresh.change(
            fn=lambda enabled: gr.update(active=enabled),
            inputs=[ingest_auto_refresh],
            outputs=[ingest_timer],
        )
        runs_timer.tick(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        ingest_timer.tick(
            fn=refresh_ingest_monitor,
            inputs=ingest_monitor_inputs,
            outputs=ingest_monitor_outputs,
        )
        export_btn.click(
            fn=export_selected_run,
            inputs=[runs_run_id, export_dir, review_db_path],
            outputs=[export_status, export_json],
        )
        retry_failed_btn.click(
            fn=retry_failed_only_view,
            inputs=[runs_run_id, review_db_path],
            outputs=[runs_recovery_status],
        ).then(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        retry_redo_btn.click(
            fn=retry_redo_only_view,
            inputs=[runs_run_id, review_db_path],
            outputs=[runs_recovery_status],
        ).then(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        reprocess_run_btn.click(
            fn=reprocess_run_view,
            inputs=[runs_run_id, review_db_path],
            outputs=[runs_recovery_status],
        ).then(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        retry_selected_btn.click(
            fn=retry_selected_derived_image_view,
            inputs=[runs_derived_image_id, runs_retry_mode, review_db_path],
            outputs=[runs_recovery_status],
        ).then(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        reprocess_selected_btn.click(
            fn=reprocess_selected_derived_image_view,
            inputs=[runs_derived_image_id, review_db_path],
            outputs=[runs_recovery_status],
        ).then(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        review_refresh_btn.click(
            fn=refresh_review_view,
            inputs=[review_run_id, review_status_filter, review_outcome_filter, review_db_path],
            outputs=[review_status_box, review_run_id, review_table, review_reaction_uid],
        )
        review_run_id.change(
            fn=refresh_review_list,
            inputs=[review_run_id, review_status_filter, review_outcome_filter, review_db_path],
            outputs=[review_status_box, review_table, review_reaction_uid],
        )
        review_reaction_uid.change(
            fn=load_reaction_detail_view,
            inputs=[
                review_reaction_uid,
                artifact_backend,
                artifact_filesystem_root,
                artifact_s3_endpoint_url,
                artifact_s3_access_key_id,
                artifact_s3_secret_access_key,
                artifact_s3_bucket,
                artifact_s3_region,
                artifact_s3_use_ssl,
                artifact_s3_key_prefix,
                review_db_path,
            ],
            outputs=[
                review_detail_status,
                source_link,
                derived_preview,
                render_preview,
                review_reaction_json,
                review_supporting_json,
                review_status_radio,
                review_notes,
            ],
        )
        review_save_btn.click(
            fn=save_reaction_review_view,
            inputs=[review_reaction_uid, review_status_radio, review_notes, review_db_path],
            outputs=[review_save_status, review_save_json],
        )
        demo.load(
            fn=refresh_runs_view,
            inputs=runs_view_inputs,
            outputs=runs_view_outputs,
        )
        demo.load(
            fn=refresh_ingest_monitor,
            inputs=ingest_monitor_inputs,
            outputs=ingest_monitor_outputs,
        )
        demo.load(
            fn=refresh_review_view,
            inputs=[review_run_id, review_status_filter, review_outcome_filter, review_db_path],
            outputs=[review_status_box, review_run_id, review_table, review_reaction_uid],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "true").strip().lower() in {"1", "true", "yes", "on"},
    )
