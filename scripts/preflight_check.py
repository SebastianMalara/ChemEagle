#!/usr/bin/env python3
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "rxn.ckpt",
    "ner.ckpt",
    "molnextr.pth",
    "moldet.ckpt",
    "corefdet.ckpt",
    "prompt/prompt_plan.txt",
    "prompt/prompt_final_simple_version.txt",
    "prompt/prompt_getreaction.txt",
    "prompt/prompt_getmolecular.txt",
    "prompt/prompt_getmolecular_correctR.txt",
]

def check_file(path: str):
    p = ROOT / path
    return p.exists(), str(p)

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return True, out
    except Exception as e:
        return False, str(e)

def main():
    print("=== ChemEAGLE preflight check ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

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
        ok_all = False
        print("[WARN] nvidia-smi not found in PATH.")

    for key in ["LLM_PROVIDER", "API_KEY", "AZURE_ENDPOINT", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VLLM_BASE_URL"]:
        if os.getenv(key):
            print(f"[OK] env {key} is set")

    missing = []
    for rel in REQUIRED_FILES:
        ok, abs_path = check_file(rel)
        if ok:
            print(f"[OK] {rel}")
        else:
            missing.append(rel)
            print(f"[MISS] {rel}")

    if missing:
        ok_all = False
        print("\nMissing required files:")
        for m in missing:
            print(f" - {m}")
        print("Run: python installer.py")

    print("\nRESULT:", "PASS" if ok_all else "CHECK REQUIRED")
    return 0 if ok_all else 1

if __name__ == "__main__":
    raise SystemExit(main())
