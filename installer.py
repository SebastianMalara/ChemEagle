#!/usr/bin/env python3
"""ChemEagle setup assistant.

Guides users through:
1) Downloading required model checkpoints.
2) Optionally cloning external repositories.
3) Creating an LLM environment file for Azure/OpenAI/Anthropic/local endpoints.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

MODEL_REPO = "CYF200127/ChemEAGLEModel"
REQUIRED_MODEL_FILES = [
    "rxn.ckpt",
    "ner.ckpt",
    "molnextr.pth",
    "moldet.ckpt",
    "corefdet.ckpt",
]

OPTIONAL_REPOS = {
    "ChemRxnExtractor": "https://github.com/CrystalEye42/ChemRxnExtractor.git",
}


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    reply = input(f"{question} {suffix} ").strip().lower()
    if not reply:
        return default
    return reply in {"y", "yes"}


def ask_input(question: str, default: Optional[str] = None, secret: bool = False) -> str:
    prompt = f"{question}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "

    value = input(prompt).strip()
    if not value and default is not None:
        return default
    return value


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_models(model_dir: Path, dry_run: bool = False) -> List[Path]:
    ensure_dir(model_dir)
    downloaded_paths: List[Path] = []
    print(f"\n[Installer] Download target directory: {model_dir}")

    for filename in REQUIRED_MODEL_FILES:
        target = model_dir / filename
        if target.exists():
            print(f"  - exists, skipping: {target}")
            downloaded_paths.append(target)
            continue

        if dry_run:
            print(f"  - would download {filename} from {MODEL_REPO}")
            downloaded_paths.append(target)
            continue

        print(f"  - downloading {filename} ...")
        downloaded = Path(hf_hub_download(repo_id=MODEL_REPO, filename=filename, local_dir=str(model_dir)))
        downloaded_paths.append(downloaded)

    return downloaded_paths


def mirror_model_files_to_project(model_dir: Path, project_root: Path, dry_run: bool = False) -> None:
    print("\n[Installer] Mirroring required files into project root (for default relative paths)...")
    for filename in REQUIRED_MODEL_FILES:
        src = model_dir / filename
        dst = project_root / filename

        if not src.exists() and not dry_run:
            print(f"  - missing source file, skip: {src}")
            continue

        if dst.exists():
            print(f"  - exists, keeping: {dst}")
            continue

        if dry_run:
            print(f"  - would copy {src} -> {dst}")
            continue

        shutil.copy2(src, dst)
        print(f"  - copied: {dst}")


def clone_repo(name: str, url: str, target_dir: Path, dry_run: bool = False) -> None:
    destination = target_dir / name
    if destination.exists():
        print(f"  - {name} already exists at {destination}, skipping clone")
        return

    if dry_run:
        print(f"  - would clone {url} into {destination}")
        return

    print(f"  - cloning {name} from {url}")
    subprocess.run(["git", "clone", url, str(destination)], check=True)


def build_provider_env(provider: str) -> Dict[str, str]:
    provider = provider.lower().strip()
    env: Dict[str, str] = {"LLM_PROVIDER": provider}

    if provider == "azure":
        env["API_KEY"] = ask_input("Azure API key")
        env["AZURE_ENDPOINT"] = ask_input("Azure endpoint (e.g., https://xxx.openai.azure.com)")
        env["API_VERSION"] = ask_input("Azure API version", default="2024-06-01")
        env["LLM_MODEL"] = ask_input("Azure model/deployment name", default="gpt-5-mini")
    elif provider == "openai":
        env["OPENAI_API_KEY"] = ask_input("OpenAI API key")
        env["LLM_MODEL"] = ask_input("OpenAI model", default="gpt-5-mini")
        base_url = ask_input("Optional custom OPENAI_BASE_URL (leave blank for default)", default="")
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
    elif provider in {"lmstudio", "openai_compatible", "local_openai"}:
        env["LLM_BASE_URL"] = ask_input("Local/OpenAI-compatible base URL", default="http://127.0.0.1:1234/v1")
        env["LLM_API_KEY"] = ask_input("Local API key", default="lm-studio")
        env["LLM_MODEL"] = ask_input("Local model name", default="your-local-model-name")
    elif provider == "anthropic":
        env["ANTHROPIC_API_KEY"] = ask_input("Anthropic API key")
        env["LLM_MODEL"] = ask_input("Anthropic model", default="claude-3-5-sonnet-latest")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return env


def write_env_file(env_vars: Dict[str, str], output_path: Path, dry_run: bool = False) -> None:
    lines = [f"{k}={v}" for k, v in env_vars.items() if v is not None]
    body = "\n".join(lines) + "\n"

    if dry_run:
        print(f"\n[Installer] Would write env file: {output_path}\n{body}")
        return

    output_path.write_text(body, encoding="utf-8")
    print(f"\n[Installer] Wrote environment file: {output_path}")


def write_activation_script(env_file: Path, output_path: Path, dry_run: bool = False) -> None:
    script = f"#!/usr/bin/env bash\nset -a\nsource \"{env_file}\"\nset +a\necho \"Loaded ChemEagle env from {env_file}\"\n"

    if dry_run:
        print(f"[Installer] Would write activation script: {output_path}")
        return

    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o755)
    print(f"[Installer] Wrote activation script: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ChemEagle installer")
    parser.add_argument("--model-dir", default="./models", help="Directory for downloaded model files")
    parser.add_argument("--repos-dir", default="./external", help="Directory for optional cloned repos")
    parser.add_argument("--provider", choices=["azure", "openai", "anthropic", "lmstudio", "openai_compatible", "local_openai"], help="Skip provider prompt")
    parser.add_argument("--env-file", default=".env.chemeagle", help="Where to write provider environment variables")
    parser.add_argument("--no-clone", action="store_true", help="Skip optional repository cloning")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    model_dir = (project_root / args.model_dir).resolve()
    repos_dir = (project_root / args.repos_dir).resolve()
    env_file = (project_root / args.env_file).resolve()
    activation_script = (project_root / "load_chemeagle_env.sh").resolve()

    print("=== ChemEagle Installer ===")

    if ask_yes_no("Download required ChemEagle model files now?", default=True):
        download_models(model_dir, dry_run=args.dry_run)
        mirror_model_files_to_project(model_dir, project_root, dry_run=args.dry_run)

    if not args.no_clone and ask_yes_no("Clone optional helper repositories?", default=False):
        ensure_dir(repos_dir)
        for name, url in OPTIONAL_REPOS.items():
            if ask_yes_no(f"Clone {name}?", default=True):
                clone_repo(name, url, repos_dir, dry_run=args.dry_run)

    if ask_yes_no("Configure LLM provider environment variables?", default=True):
        provider = args.provider or ask_input(
            "Select provider (azure/openai/anthropic/lmstudio/openai_compatible/local_openai)",
            default="azure",
        )
        env_vars = build_provider_env(provider)
        write_env_file(env_vars, env_file, dry_run=args.dry_run)
        write_activation_script(env_file, activation_script, dry_run=args.dry_run)

    print("\n[Installer] Done.")
    print(f"Use: source {activation_script}  # to load provider env vars")


if __name__ == "__main__":
    main()
