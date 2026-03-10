#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asset_registry import build_asset_preflight_report

PROMPT_FILES = [
    "prompt/prompt_plan.txt",
    "prompt/prompt_final_simple_version.txt",
    "prompt/prompt_getreaction.txt",
    "prompt/prompt_getmolecular.txt",
    "prompt/prompt_getmolecular_correctR.txt",
]


def resolve_ocr_backend(requested_backend: str | None, run_mode: str | None) -> str:
    backend = (requested_backend or os.getenv("OCR_BACKEND") or "auto").strip().lower()
    mode = (run_mode or os.getenv("CHEMEAGLE_RUN_MODE") or "cloud").strip().lower()
    aliases = {
        "vision": "llm_vision",
        "llm": "llm_vision",
        "easy_ocr": "easyocr",
    }
    backend = aliases.get(backend, backend)
    if backend == "auto":
        return "llm_vision" if mode == "cloud" else "easyocr"
    return backend


def check_file(path: str):
    p = ROOT / path
    return p.exists(), str(p)


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return True, out
    except Exception as e:
        return False, str(e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChemEagle preflight check")
    parser.add_argument("--mode", default=os.getenv("CHEMEAGLE_RUN_MODE", "cloud"), choices=["cloud", "open-source"])
    parser.add_argument("--ocr-backend", default=os.getenv("OCR_BACKEND", "auto"), choices=["auto", "llm_vision", "easyocr", "tesseract"])
    parser.add_argument("--file-kind", default="image", choices=["image", "pdf"])
    parser.add_argument("--pdf-model-size", default=os.getenv("PDF_MODEL_SIZE", "large"), choices=["base", "large"])
    parser.add_argument("--tool", action="append", default=[], help="Optional planned tool name, e.g. --tool text_extraction_agent")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== ChemEAGLE preflight check ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"mode: {args.mode}")
    print(f"ocr_backend: {resolve_ocr_backend(args.ocr_backend, args.mode)}")
    print(f"file_kind: {args.file_kind}")

    ok_all = True

    py = sys.version_info
    if (py.major, py.minor) != (3, 10):
        ok_all = False
        print(f"[WARN] Unsupported Python {py.major}.{py.minor}. Use Python 3.10 for this repo (torch==2.2.0 and pinned deps).")
        print("[HINT] Recreate env with: python3.10 -m venv .venv  (or conda create -n chemeagle python=3.10)")

    if platform.system() != "Linux":
        ok_all = False
        print("[WARN] Non-Linux platform detected; guide targets Ubuntu.")

    if shutil.which("nvidia-smi"):
        ok, out = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        print("[OK] nvidia-smi found" if ok else "[WARN] nvidia-smi exists but query failed")
        if out:
            print(out.splitlines()[0])
    else:
        print("[WARN] nvidia-smi not found in PATH.")

    for key in ["LLM_PROVIDER", "API_KEY", "AZURE_ENDPOINT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VLLM_BASE_URL", "CHEMEAGLE_ASSET_ROOT"]:
        if os.getenv(key):
            print(f"[OK] env {key} is set")

    prompt_missing = []
    for rel in PROMPT_FILES:
        ok, abs_path = check_file(rel)
        if ok:
            print(f"[OK] {rel}")
        else:
            prompt_missing.append(rel)
            print(f"[MISS] {rel}")

    resolved_ocr_backend = resolve_ocr_backend(args.ocr_backend, args.mode)
    asset_report = build_asset_preflight_report(
        mode=args.mode,
        ocr_backend=resolved_ocr_backend,
        file_kind=args.file_kind,
        pdf_model_size=args.pdf_model_size,
        selected_tools=args.tool,
    )
    print(f"\nAsset root: {asset_report['asset_root']}")
    for item in asset_report["assets"]:
        requirement = "blocking" if item["blocking"] else ("optional" if item["required_for_current_run"] else "unused")
        status = "[OK]" if item["present"] else "[MISS]"
        print(f"{status} {item['asset_id']} ({requirement}) -> {item['expected_path']}")
        if item["present"] and item["resolved_from"] != "asset_root":
            print(f"      resolved from {item['resolved_from']}: {item['resolved_path']}")

    if prompt_missing:
        ok_all = False
        print("\nMissing prompt files:")
        for rel in prompt_missing:
            print(f" - {rel}")

    if asset_report["blocking_errors"]:
        ok_all = False
        print("\nBlocking asset issues:")
        for issue in asset_report["blocking_errors"]:
            print(f" - {issue}")

    if asset_report["warnings"]:
        print("\nOptional asset warnings:")
        for issue in asset_report["warnings"]:
            print(f" - {issue}")

    print("\nRESULT:", "PASS" if ok_all else "CHECK REQUIRED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
