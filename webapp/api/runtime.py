from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from asset_registry import ASSET_ENV_VAR, build_asset_preflight_report
from hf_runtime import configure_transformers_runtime
from llm_preflight import collect_runtime_provider_preflight
from llm_profiles import MANUAL_MODEL_LIST_PROVIDERS, list_available_models, resolve_llm_profile
from runtime_device import resolve_ocr_backend


ENV_FILE_DEFAULT = Path(".env.chemeagle")
UPLOAD_ROOT = Path("./data/webapp_uploads").resolve()
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LLM_PROVIDER_CHOICES = ["azure", "openai", "openai_compatible", "lmstudio", "local_openai", "anthropic"]
OCR_BACKEND_CHOICES = ["auto", "llm_vision", "easyocr", "tesseract"]
MOLECULE_SMILES_RESCUE_CHOICES = ["off", "decimer"]
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
    "MOLECULE_SMILES_RESCUE",
    "MOLECULE_SMILES_RESCUE_CONFIDENCE",
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


def parse_env_file(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def merged_env_values(env_path: Path) -> Dict[str, str]:
    from_file = parse_env_file(env_path)
    merged: Dict[str, str] = {}
    for key in ENV_KEYS:
        merged[key] = os.getenv(key, from_file.get(key, ""))
    return merged


def default_runtime_values(env_path: Path | None = None) -> Dict[str, str]:
    env_path = env_path or ENV_FILE_DEFAULT
    values = merged_env_values(env_path)
    if not values.get("CHEMEAGLE_RUN_MODE"):
        values["CHEMEAGLE_RUN_MODE"] = "cloud"
    if not values.get("CHEMEAGLE_DEVICE"):
        values["CHEMEAGLE_DEVICE"] = "auto"
    if not values.get("LLM_PROVIDER"):
        values["LLM_PROVIDER"] = "azure"
    if not values.get("LLM_MODEL"):
        values["LLM_MODEL"] = "gpt-5-mini"
    if not values.get("OCR_BACKEND"):
        values["OCR_BACKEND"] = "auto"
    if not values.get("OCR_LANG"):
        values["OCR_LANG"] = "eng"
    if not values.get("MOLECULE_SMILES_RESCUE"):
        values["MOLECULE_SMILES_RESCUE"] = "off"
    if not values.get("MOLECULE_SMILES_RESCUE_CONFIDENCE"):
        values["MOLECULE_SMILES_RESCUE_CONFIDENCE"] = "0.85"
    if not values.get("PDF_MODEL_SIZE"):
        values["PDF_MODEL_SIZE"] = "large"
    if not values.get("API_VERSION"):
        values["API_VERSION"] = "2024-06-01"
    if not values.get("OCR_API_VERSION"):
        values["OCR_API_VERSION"] = values["API_VERSION"]
    if not values.get("OCR_LLM_INHERIT_MAIN"):
        values["OCR_LLM_INHERIT_MAIN"] = "1"
    if not values.get("ARTIFACT_BACKEND"):
        values["ARTIFACT_BACKEND"] = "filesystem"
    if not values.get("ARTIFACT_FILESYSTEM_ROOT"):
        values["ARTIFACT_FILESYSTEM_ROOT"] = DEFAULT_ARTIFACT_FILESYSTEM_ROOT
    if not values.get("ARTIFACT_S3_ENDPOINT_URL"):
        values["ARTIFACT_S3_ENDPOINT_URL"] = DEFAULT_MINIO_ENDPOINT
    if not values.get("ARTIFACT_S3_BUCKET"):
        values["ARTIFACT_S3_BUCKET"] = DEFAULT_MINIO_BUCKET
    if not values.get("REVIEW_DB_PATH"):
        values["REVIEW_DB_PATH"] = DEFAULT_REVIEW_DB_PATH
    return values


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
    configure_transformers_runtime()
    device_preference = (values.get("CHEMEAGLE_DEVICE") or "").strip().lower()
    if device_preference in {"auto", "cuda"} and not os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def env_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_runtime_values(values: Dict[str, Any]) -> Dict[str, str]:
    normalized = default_runtime_values()
    for key, value in (values or {}).items():
        if key not in ENV_KEYS:
            continue
        if isinstance(value, bool):
            normalized[key] = "1" if value else "0"
        elif value is None:
            normalized[key] = ""
        else:
            normalized[key] = str(value)

    provider = (normalized.get("LLM_PROVIDER") or "").strip().lower()
    if provider in {"openai", "openai_compatible", "lmstudio", "local_openai"}:
        if not normalized.get("OPENAI_API_KEY") and normalized.get("API_KEY"):
            normalized["OPENAI_API_KEY"] = normalized["API_KEY"]
        if not normalized.get("OPENAI_API_KEY") and normalized.get("VLLM_API_KEY"):
            normalized["OPENAI_API_KEY"] = normalized["VLLM_API_KEY"]
    return normalized


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
    molecule_smiles_rescue: str,
    molecule_smiles_rescue_confidence: str,
    pdf_model_size: str,
    pdf_persist_images: bool,
    pdf_persist_dir: str,
) -> Dict[str, str]:
    return normalize_runtime_values(
        {
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
            "OCR_LLM_INHERIT_MAIN": ocr_llm_inherit_main,
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
            "MOLECULE_SMILES_RESCUE": molecule_smiles_rescue,
            "MOLECULE_SMILES_RESCUE_CONFIDENCE": molecule_smiles_rescue_confidence,
            "TESSERACT_CMD": tesseract_cmd,
            "PDF_MODEL_SIZE": pdf_model_size,
            "PDF_PERSIST_IMAGES": pdf_persist_images,
            "PDF_PERSIST_DIR": pdf_persist_dir,
        }
    )


def build_dataset_runtime_values(
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
    molecule_smiles_rescue: str,
    molecule_smiles_rescue_confidence: str,
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
        molecule_smiles_rescue,
        molecule_smiles_rescue_confidence,
        pdf_model_size,
        pdf_persist_images,
        pdf_persist_dir,
    )
    values.update(
        normalize_runtime_values(
            {
                "ARTIFACT_BACKEND": artifact_backend,
                "ARTIFACT_FILESYSTEM_ROOT": artifact_filesystem_root,
                "ARTIFACT_S3_ENDPOINT_URL": artifact_s3_endpoint_url,
                "ARTIFACT_S3_ACCESS_KEY_ID": artifact_s3_access_key_id,
                "ARTIFACT_S3_SECRET_ACCESS_KEY": artifact_s3_secret_access_key,
                "ARTIFACT_S3_BUCKET": artifact_s3_bucket,
                "ARTIFACT_S3_REGION": artifact_s3_region,
                "ARTIFACT_S3_USE_SSL": artifact_s3_use_ssl,
                "ARTIFACT_S3_KEY_PREFIX": artifact_s3_key_prefix,
                "REVIEW_DB_PATH": review_db_path,
            }
        )
    )
    return values


def resolve_upload_path(upload: Any) -> str:
    if upload is None:
        return ""
    if isinstance(upload, str):
        return upload
    return getattr(upload, "name", "") or ""


def resolve_upload_paths(upload: Any) -> List[str]:
    if upload is None:
        return []
    if isinstance(upload, list):
        paths: List[str] = []
        for item in upload:
            paths.extend(resolve_upload_paths(item))
        return [path for path in paths if path]
    return [path for path in [resolve_upload_path(upload)] if path]


def scan_supported_files(folder_path: str) -> List[str]:
    raw = (folder_path or "").strip()
    if not raw:
        return []
    root = Path(raw).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    suffixes = IMAGE_SUFFIXES | {".pdf"}
    return [str(path.resolve()) for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes]


def parse_profile_configs(base_values: Dict[str, str], profiles: str | Sequence[Dict[str, Any]] | Dict[str, Any] | None) -> List[Dict[str, str]]:
    if profiles is None or profiles == "":
        return [{**base_values, "profile_label": "baseline"}]

    parsed: Any = profiles
    if isinstance(profiles, str):
        parsed = json.loads(profiles)
    if isinstance(parsed, dict):
        parsed = [parsed]

    configs: List[Dict[str, str]] = []
    for index, item in enumerate(parsed or []):
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


def collect_batch_runtime_diagnostics(
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
    blocking_errors.extend(
        f"runtime_provider_preflight: {item}" for item in runtime_provider_preflight.get("blocking_errors", [])
    )
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


def model_picker_help(provider: str) -> str:
    provider_norm = (provider or "").strip().lower()
    if provider_norm in MANUAL_MODEL_LIST_PROVIDERS:
        return "Azure model listing stays manual in this pass. Enter the deployment/model name directly."
    if provider_norm in {"openai_compatible", "lmstudio", "local_openai"}:
        return "Manual entry is enabled. Refresh will try GET /models on the configured compatible endpoint."
    return "Manual entry is enabled. Click Refresh Models to fetch the current provider catalog."


def ocr_profile_summary(
    inherit_main: bool,
    main_provider: str,
    main_model: str,
    ocr_provider: str,
    ocr_model: str,
) -> str:
    if inherit_main:
        return (
            "LLM vision OCR inherits the main profile: "
            f"provider={main_provider or 'unset'}, model={main_model or '(provider default)'}."
        )
    return (
        "LLM vision OCR uses its own profile: "
        f"provider={ocr_provider or main_provider or 'unset'}, model={ocr_model or '(provider default)'}."
    )


def model_choices(current_value: str, catalog_ids: List[str]) -> List[str]:
    ordered: List[str] = []
    if current_value:
        ordered.append(current_value)
    for model_id in catalog_ids:
        if model_id and model_id not in ordered:
            ordered.append(model_id)
    return ordered


def model_catalog_guard(profile: Any) -> str:
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


def trim_text(text: str, limit: int = 1200) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def probe_python_code(code: str, env: Dict[str, str]) -> Dict[str, Any]:
    timeout_seconds = max(1.0, float(os.getenv("CHEMEAGLE_PREFLIGHT_PROBE_TIMEOUT_SECONDS", "15")))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        timeout_message = f"Probe timed out after {timeout_seconds:.1f}s."
        stderr = f"{stderr}\n{timeout_message}".strip() if stderr else timeout_message
        return {
            "ok": False,
            "returncode": None,
            "stdout": trim_text(stdout),
            "stderr": trim_text(stderr),
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": trim_text(proc.stdout),
        "stderr": trim_text(proc.stderr),
    }


def resolve_tesseract_cmd(values: Dict[str, str]) -> str:
    explicit = values.get("TESSERACT_CMD") or os.getenv("CHEMEAGLE_TESSERACT_CMD") or ""
    if explicit:
        normalized = os.path.normpath(explicit)
        if os.path.exists(normalized):
            return normalized
    return shutil.which("tesseract") or ""


def profile_preflight(scope: str, *, required: bool, values: Dict[str, str]) -> Dict[str, Any]:
    blocking_errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {"scope": scope, "required": required}
    try:
        profile = resolve_llm_profile(scope=scope, values=values, default_model="gpt-5-mini")
    except Exception as exc:
        if required:
            blocking_errors.append(f"Failed to resolve the {scope} LLM profile: {exc}")
        else:
            warnings.append(f"The optional {scope} LLM profile could not be resolved: {exc}")
        checks["resolution_error"] = str(exc)
        return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}

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
        return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}

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

    return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}


def ocr_preflight(mode: str, values: Dict[str, str]) -> Dict[str, Any]:
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
        return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}

    probe_lines: List[str] = []
    if resolved_backend == "easyocr":
        probe_lines.append("import easyocr")
    elif resolved_backend == "tesseract":
        probe_lines.append("import pytesseract")
    if probe_lines:
        checks["python_import_probe"] = probe_python_code("\n".join(probe_lines), env)
        if not checks["python_import_probe"]["ok"]:
            blocking_errors.append("OCR backend dependencies failed to import. See diagnostics.ocr_preflight.checks.python_import_probe.")

    if resolved_backend == "tesseract":
        tesseract_cmd = resolve_tesseract_cmd(values)
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
    return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}


def torch_runtime_preflight(
    mode: str,
    resolved_ocr_backend: str,
    *,
    include_pdf_section: bool,
    values: Dict[str, str],
) -> Dict[str, Any]:
    requires_torch = mode == "local_os" or resolved_ocr_backend == "easyocr" or include_pdf_section
    checks: Dict[str, Any] = {
        "device_preference": values.get("CHEMEAGLE_DEVICE") or "auto",
        "required_for_current_run": requires_torch,
        "mode": mode,
        "resolved_ocr_backend": resolved_ocr_backend,
        "include_pdf_section": include_pdf_section,
    }
    blocking_errors: List[str] = []
    warnings: List[str] = []
    checks["python_import_probe"] = probe_python_code(
        "import torch\nprint(getattr(torch, '__version__', 'unknown'))",
        dict(os.environ),
    )
    if not checks["python_import_probe"]["ok"]:
        message = (
            "PyTorch failed to import. Local model execution will fail. "
            "See diagnostics.torch_runtime_preflight.checks.python_import_probe."
        )
        if requires_torch:
            blocking_errors.append(message)
        else:
            warnings.append(message)
    return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}


def molecule_smiles_rescue_preflight(values: Dict[str, str]) -> Dict[str, Any]:
    requested_rescue = (values.get("MOLECULE_SMILES_RESCUE") or "off").strip().lower()
    raw_confidence = (values.get("MOLECULE_SMILES_RESCUE_CONFIDENCE") or "0.85").strip()
    checks: Dict[str, Any] = {
        "requested_rescue": requested_rescue,
        "raw_confidence_threshold": raw_confidence,
        "python_executable": sys.executable,
    }
    blocking_errors: List[str] = []
    warnings: List[str] = [
        "DECIMER rescue in this branch applies only to direct molecule extraction, not RxnIM reaction/coreference flows."
    ]

    if requested_rescue not in {"off", "decimer"}:
        blocking_errors.append("MOLECULE_SMILES_RESCUE must be one of: off, decimer.")
        return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}

    try:
        confidence_threshold = float(raw_confidence)
    except ValueError:
        blocking_errors.append("MOLECULE_SMILES_RESCUE_CONFIDENCE must be a float between 0 and 1.")
        return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}

    checks["confidence_threshold"] = confidence_threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        blocking_errors.append("MOLECULE_SMILES_RESCUE_CONFIDENCE must be a float between 0 and 1.")

    if requested_rescue == "decimer":
        probe_lines = [
            "import importlib.util, json",
            "decimer_spec = importlib.util.find_spec('DECIMER')",
            "shim_spec = importlib.util.find_spec('decimer')",
            "if decimer_spec is None and shim_spec is None:",
            "    raise ModuleNotFoundError(\"No module named 'DECIMER' or 'decimer'\")",
            "print(json.dumps({'DECIMER': getattr(decimer_spec, 'origin', None), 'decimer': getattr(shim_spec, 'origin', None)}))",
        ]
        checks["python_import_probe"] = probe_python_code("\n".join(probe_lines), dict(os.environ))
        if checks["python_import_probe"]["stdout"]:
            checks["module_probe"] = checks["python_import_probe"]["stdout"]
        if not checks["python_import_probe"]["ok"]:
            blocking_errors.append(
                f"DECIMER failed to import in {sys.executable}. Install `decimer` into that interpreter or set MOLECULE_SMILES_RESCUE=off."
            )
    return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}


def model_catalog_preflight(mode: str, resolved_ocr_backend: str, values: Dict[str, str]) -> Dict[str, Any]:
    blocking_errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {}

    for scope, required in [("main", mode == "cloud"), ("ocr", resolved_ocr_backend == "llm_vision")]:
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

    return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}


def visualheist_cache_state(model_size: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
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


def pdf_preflight(file_path: str, values: Dict[str, str]) -> Dict[str, Any]:
    model_size = (values.get("PDF_MODEL_SIZE") or "large").strip().lower()
    checks: Dict[str, Any] = {"model_size": model_size, "pdftoppm": shutil.which("pdftoppm") or ""}
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
            (Path(tmpdir) / "write_test.txt").write_text("ok", encoding="utf-8")
        checks["temp_dir_writable"] = True
    except Exception as exc:
        checks["temp_dir_writable"] = False
        checks["temp_dir_error"] = str(exc)
        blocking_errors.append(f"Temporary output directory is not writable: {exc}")

    checks["visualheist_cache"] = visualheist_cache_state(model_size)
    if not checks["visualheist_cache"]["cached"]:
        warnings.append(f"VisualHeist {model_size} weights are not in the legacy locations or Hugging Face cache.")

    checks["python_import_probe"] = probe_python_code(
        "from pdf_extraction import run_pdf\nfrom pdfmodel.methods import _pdf_to_figures_and_tables",
        dict(os.environ),
    )
    if not checks["python_import_probe"]["ok"]:
        blocking_errors.append("PDF extraction dependencies failed to import. See diagnostics.pdf_preflight.checks.python_import_probe.")
    return {"blocking_errors": blocking_errors, "warnings": warnings, "checks": checks}


def asset_preflight(
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
    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "device": values.get("CHEMEAGLE_DEVICE") or "auto",
        "resolved_ocr_backend": resolved_ocr_backend,
        "main_llm_preflight": profile_preflight("main", required=(mode == "cloud"), values=values),
        "ocr_llm_preflight": profile_preflight("ocr", required=(resolved_ocr_backend == "llm_vision"), values=values),
        "model_catalog_preflight": model_catalog_preflight(mode, resolved_ocr_backend, values),
        "torch_runtime_preflight": torch_runtime_preflight(
            mode,
            resolved_ocr_backend,
            include_pdf_section=include_pdf_section,
            values=values,
        ),
        "ocr_preflight": ocr_preflight(mode, values),
        "molecule_smiles_rescue_preflight": molecule_smiles_rescue_preflight(values),
        "asset_preflight": asset_preflight(file_path, mode, values),
        "runtime_provider_preflight": collect_runtime_provider_preflight(profile_configs=[values], mode=mode),
    }
    if include_pdf_section:
        diagnostics["pdf_preflight"] = pdf_preflight(file_path, values)

    blocking_errors: List[str] = []
    warnings: List[str] = []
    for section_name in (
        "main_llm_preflight",
        "ocr_llm_preflight",
        "model_catalog_preflight",
        "torch_runtime_preflight",
        "ocr_preflight",
        "molecule_smiles_rescue_preflight",
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


def refresh_model_catalog(scope: str, current_model: str, values: Dict[str, str]) -> Dict[str, Any]:
    try:
        profile = resolve_llm_profile(scope=scope, values=values, default_model="gpt-5-mini")
    except Exception as exc:
        return {
            "scope": scope,
            "models": model_choices(current_model, []),
            "selected_model": current_model,
            "status": f"Could not resolve the {scope} profile: {exc}",
        }

    if scope == "ocr" and getattr(profile, "inherit_main", False):
        return {
            "scope": scope,
            "models": model_choices(current_model, []),
            "selected_model": current_model,
            "status": "LLM vision OCR is inheriting the main profile. Disable inherit to fetch a separate catalog.",
        }

    guard = model_catalog_guard(profile)
    if guard:
        return {
            "scope": scope,
            "models": model_choices(current_model, []),
            "selected_model": current_model,
            "status": guard,
        }

    try:
        catalog = list_available_models(profile)
    except Exception as exc:
        return {
            "scope": scope,
            "models": model_choices(current_model, []),
            "selected_model": current_model,
            "status": f"Model refresh failed: {exc}",
        }

    fetched_ids = [item.id for item in catalog.models]
    selected_value = current_model or (fetched_ids[0] if fetched_ids else "")
    status = catalog.status
    if current_model and current_model not in fetched_ids:
        status += " Current value was kept even though it was not present in the fetched catalog."
    return {
        "scope": scope,
        "models": model_choices(selected_value, fetched_ids),
        "selected_model": selected_value,
        "status": status,
    }


def config_metadata(values: Dict[str, str]) -> Dict[str, Any]:
    main_provider = values.get("LLM_PROVIDER", "azure") or "azure"
    main_model = values.get("LLM_MODEL", "gpt-5-mini") or "gpt-5-mini"
    ocr_inherit = env_truthy(values.get("OCR_LLM_INHERIT_MAIN", "1") or "1")
    ocr_provider = values.get("OCR_LLM_PROVIDER", main_provider) or main_provider
    ocr_model = values.get("OCR_LLM_MODEL", "")
    return {
        "provider_choices": LLM_PROVIDER_CHOICES,
        "ocr_backend_choices": OCR_BACKEND_CHOICES,
        "molecule_smiles_rescue_choices": MOLECULE_SMILES_RESCUE_CHOICES,
        "pdf_model_choices": PDF_MODEL_CHOICES,
        "defaults": {
            "env_path": str(ENV_FILE_DEFAULT),
            "review_db_path": DEFAULT_REVIEW_DB_PATH,
            "artifact_filesystem_root": DEFAULT_ARTIFACT_FILESYSTEM_ROOT,
            "export_dir": DEFAULT_EXPORT_DIR,
            "artifact_s3_endpoint_url": DEFAULT_MINIO_ENDPOINT,
            "artifact_s3_bucket": DEFAULT_MINIO_BUCKET,
        },
        "main_model_status": model_picker_help(main_provider),
        "ocr_model_status": (
            "LLM vision OCR is inheriting the main profile. Disable inherit to pick a separate provider/model."
            if ocr_inherit
            else model_picker_help(ocr_provider)
        ),
        "ocr_profile_summary": ocr_profile_summary(ocr_inherit, main_provider, main_model, ocr_provider, ocr_model),
    }


def persist_uploaded_files(files: Sequence[Any], *, prefix: str = "upload") -> List[str]:
    if not files:
        return []
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(UPLOAD_ROOT)))
    stored_paths: List[str] = []
    for index, upload in enumerate(files):
        filename = Path(getattr(upload, "filename", "") or f"{prefix}_{index}").name
        if not filename:
            filename = f"{prefix}_{index}"
        target = session_dir / filename
        with target.open("wb") as handle:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        stored_paths.append(str(target.resolve()))
    return stored_paths
