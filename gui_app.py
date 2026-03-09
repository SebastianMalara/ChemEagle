#!/usr/bin/env python3
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import gradio as gr
except ImportError as e:
    raise ImportError("gradio is required for gui_app.py. Install dependencies with: pip install -r requirements.txt") from e

from runtime_device import resolve_ocr_backend


ENV_FILE_DEFAULT = Path(".env.chemeagle")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LLM_PROVIDER_CHOICES = ["azure", "openai", "openai_compatible", "lmstudio", "local_openai", "anthropic"]
OCR_BACKEND_CHOICES = ["auto", "llm_vision", "easyocr", "tesseract"]
PDF_MODEL_CHOICES = ["base", "large"]
ENV_KEYS = [
    "CHEMEAGLE_RUN_MODE",
    "CHEMEAGLE_DEVICE",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OCR_BACKEND",
    "OCR_LLM_MODEL",
    "OCR_LANG",
    "OCR_CONFIG",
    "TESSERACT_CMD",
    "PDF_MODEL_SIZE",
    "API_KEY",
    "AZURE_ENDPOINT",
    "API_VERSION",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
]


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
    lines = [f"{key}={values.get(key, '')}" for key in ENV_KEYS if values.get(key, "")]
    env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return f"Saved {env_path} with {len(lines)} keys."


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
    ocr_llm_model: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
) -> Dict[str, str]:
    return {
        "CHEMEAGLE_RUN_MODE": mode,
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
        "OCR_LLM_MODEL": ocr_llm_model,
        "OCR_LANG": ocr_lang,
        "OCR_CONFIG": ocr_config,
        "TESSERACT_CMD": tesseract_cmd,
        "PDF_MODEL_SIZE": pdf_model_size,
    }


def _resolve_upload_path(upload) -> str:
    if upload is None:
        return ""
    if isinstance(upload, str):
        return upload
    return getattr(upload, "name", "") or ""


def _trim_text(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


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


def _llm_preflight(mode: str, resolved_ocr_backend: str, values: Dict[str, str]) -> Dict[str, Any]:
    provider = (values.get("LLM_PROVIDER") or "azure").strip().lower()
    checks: Dict[str, Any] = {
        "provider": provider,
        "model": values.get("LLM_MODEL") or "gpt-5-mini",
    }
    blocking_errors: List[str] = []
    warnings: List[str] = []
    requires_provider_config = mode == "cloud" or resolved_ocr_backend == "llm_vision"

    if mode == "local_os" and not requires_provider_config:
        checks["base_url"] = values.get("VLLM_BASE_URL") or "http://localhost:8000/v1"
        checks["api_key_present"] = bool(values.get("VLLM_API_KEY") or "EMPTY")
        if not values.get("VLLM_BASE_URL"):
            warnings.append("VLLM_BASE_URL is empty; local_os will fall back to http://localhost:8000/v1.")
        return {
            "blocking_errors": blocking_errors,
            "warnings": warnings,
            "checks": checks,
        }

    if provider == "azure":
        checks["api_key_present"] = bool(values.get("API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
        checks["endpoint_present"] = bool(values.get("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT"))
        if not checks["api_key_present"]:
            blocking_errors.append("Azure mode requires API_KEY or AZURE_OPENAI_API_KEY.")
        if not checks["endpoint_present"]:
            blocking_errors.append("Azure mode requires AZURE_ENDPOINT or AZURE_OPENAI_ENDPOINT.")
    elif provider in {"openai", "openai_compatible", "lmstudio", "local_openai"}:
        base_url = values.get("OPENAI_BASE_URL") or values.get("VLLM_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
        api_key = values.get("OPENAI_API_KEY") or values.get("VLLM_API_KEY") or values.get("API_KEY") or ""
        checks["base_url"] = base_url
        checks["api_key_present"] = bool(api_key)
        if provider in {"openai_compatible", "lmstudio", "local_openai"} and not base_url:
            blocking_errors.append(f"{provider} requires OPENAI_BASE_URL or VLLM_BASE_URL.")
        if provider == "openai" and not api_key and not base_url:
            blocking_errors.append("OpenAI mode requires OPENAI_API_KEY or an explicit compatible base URL.")
    elif provider == "anthropic":
        checks["api_key_present"] = bool(values.get("ANTHROPIC_API_KEY"))
        if not checks["api_key_present"]:
            blocking_errors.append("Anthropic mode requires ANTHROPIC_API_KEY.")
    else:
        blocking_errors.append(f"Unsupported LLM provider: {provider}")

    if not checks.get("model"):
        warnings.append("LLM_MODEL is empty; runtime will fall back to provider defaults when possible.")

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

    probe_lines = [
        "from chemrxnextractor import RxnExtractor",
        "from chemiener import ChemNER",
    ]
    if resolved_backend == "easyocr":
        probe_lines.append("import easyocr")
    elif resolved_backend == "tesseract":
        probe_lines.append("import pytesseract")
    checks["python_import_probe"] = _probe_python_code("\n".join(probe_lines), env)
    if not checks["python_import_probe"]["ok"]:
        blocking_errors.append("OCR/text extraction dependencies failed to import. See diagnostics.ocr_preflight.checks.python_import_probe.")

    if resolved_backend == "tesseract":
        tesseract_cmd = _resolve_tesseract_cmd(values)
        checks["tesseract_cmd"] = tesseract_cmd
        if not tesseract_cmd:
            blocking_errors.append("Tesseract backend selected, but no Tesseract executable was found.")
    elif resolved_backend == "easyocr":
        checks["device_pref"] = values.get("CHEMEAGLE_DEVICE") or "auto"
    elif resolved_backend == "llm_vision":
        checks["ocr_llm_model"] = values.get("OCR_LLM_MODEL") or values.get("LLM_MODEL") or "gpt-5-mini"
        warnings.append("LLM vision OCR assumes the selected model supports image inputs.")

    if requested_backend == "auto":
        checks["auto_resolution_rule"] = f"{mode} -> {resolved_backend}"

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
        warnings.append(f"VisualHeist {model_size} weights are not cached locally. The first PDF run may need to download them.")

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


def collect_preflight_diagnostics(file_path: str, mode: str, values: Dict[str, str], *, include_pdf_section: bool) -> Dict[str, Any]:
    resolved_ocr_backend = resolve_ocr_backend(values.get("OCR_BACKEND"), mode)
    diagnostics: Dict[str, Any] = {
        "mode": mode,
        "device": values.get("CHEMEAGLE_DEVICE") or "auto",
        "resolved_ocr_backend": resolved_ocr_backend,
        "llm_preflight": _llm_preflight(mode, resolved_ocr_backend, values),
        "ocr_preflight": _ocr_preflight(mode, values),
    }

    if include_pdf_section:
        diagnostics["pdf_preflight"] = _pdf_preflight(file_path, values)

    blocking_errors: List[str] = []
    warnings: List[str] = []
    for section_name in ("llm_preflight", "ocr_preflight", "pdf_preflight"):
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


def _run_on_pdf(pdf_path: str, mode: str, pdf_model_size: str) -> List[dict]:
    from pdf_extraction import run_pdf

    with tempfile.TemporaryDirectory(prefix="chemeagle_pdf_") as tmpdir:
        run_pdf(pdf_dir=pdf_path, image_dir=tmpdir, model_size=pdf_model_size)
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
        return results


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
    ocr_llm_model: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
) -> Tuple[str, str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
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
        ocr_llm_model,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
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
    ocr_llm_model: str,
    ocr_lang: str,
    ocr_config: str,
    tesseract_cmd: str,
    pdf_model_size: str,
) -> Tuple[str, str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
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
        ocr_llm_model,
        ocr_lang,
        ocr_config,
        tesseract_cmd,
        pdf_model_size,
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
            result = _run_on_pdf(file_path, mode, values.get("PDF_MODEL_SIZE") or "large")
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


def build_app() -> gr.Blocks:
    vals = merged_env_values(ENV_FILE_DEFAULT)

    with gr.Blocks(title="ChemEagle Self-Hosted GUI") as demo:
        gr.Markdown("# ChemEagle Self-Hosted GUI")
        gr.Markdown("Configure run mode, OCR backend, and PDF extraction settings before launching the current ChemEagle pipeline.")

        with gr.Group():
            gr.Markdown("## Run")
            with gr.Row():
                upload = gr.File(label="Upload image or PDF", file_count="single", scale=2)
                env_path = gr.Textbox(label="Env file path", value=str(ENV_FILE_DEFAULT), scale=2)
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
            with gr.Row():
                preflight_btn = gr.Button("Run Precheck")
                run_btn = gr.Button("Run ChemEagle", variant="primary")

        with gr.Tabs():
            with gr.Tab("LLM"):
                gr.Markdown("Provider credentials used by the planner and, when selected, by `llm_vision` OCR.")
                with gr.Row():
                    llm_provider = gr.Dropdown(
                        choices=LLM_PROVIDER_CHOICES,
                        value=vals.get("LLM_PROVIDER", "azure") or "azure",
                        allow_custom_value=True,
                        label="LLM provider",
                    )
                    llm_model = gr.Textbox(label="LLM_MODEL", value=vals.get("LLM_MODEL", "gpt-5-mini"))
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
                gr.Markdown("`auto` resolves to `llm_vision` in cloud mode and `easyocr` in local_os mode. Tesseract stays available as an explicit fallback.")
                with gr.Row():
                    ocr_backend = gr.Dropdown(
                        choices=OCR_BACKEND_CHOICES,
                        value=vals.get("OCR_BACKEND", "auto") or "auto",
                        label="OCR backend",
                    )
                    ocr_llm_model = gr.Textbox(
                        label="OCR_LLM_MODEL",
                        value=vals.get("OCR_LLM_MODEL", ""),
                        placeholder="Optional override for llm_vision OCR",
                    )
                with gr.Row():
                    ocr_lang = gr.Textbox(label="OCR_LANG", value=vals.get("OCR_LANG", "eng"))
                    ocr_config = gr.Textbox(label="OCR_CONFIG", value=vals.get("OCR_CONFIG", ""))
                with gr.Row():
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

        with gr.Group():
            gr.Markdown("## Output")
            status = gr.Textbox(label="Status", lines=12)
            with gr.Row():
                output = gr.Code(label="JSON output", language="json")
                diagnostics = gr.Code(label="Diagnostics / preflight", language="json")

        app_inputs = [
            env_path,
            mode,
            chemeagle_device,
            upload,
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
            ocr_llm_model,
            ocr_lang,
            ocr_config,
            tesseract_cmd,
            pdf_model_size,
        ]

        run_btn.click(
            fn=run_pipeline,
            inputs=app_inputs,
            outputs=[status, output, diagnostics],
        )
        preflight_btn.click(
            fn=run_preflight,
            inputs=app_inputs,
            outputs=[status, output, diagnostics],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "true").strip().lower() in {"1", "true", "yes", "on"},
    )
