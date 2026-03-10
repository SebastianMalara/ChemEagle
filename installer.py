#!/usr/bin/env python3
"""ChemEagle setup assistant.

Installs the canonical offline asset bundle under ./assets by default, verifies
bundle completeness, and optionally configures provider environment variables.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional

from asset_registry import (
    ALL_ASSET_IDS,
    DEFAULT_ASSET_ROOT,
    asset_bundle_report,
    get_asset_root,
    install_assets,
    write_asset_manifest,
)

OPTIONAL_REPOS = {
    "ChemRxnExtractor": "https://github.com/CrystalEye42/ChemRxnExtractor.git",
}


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    reply = input(f"{question} {suffix} ").strip().lower()
    if not reply:
        return default
    return reply in {"y", "yes"}


def ask_input(question: str, default: Optional[str] = None) -> str:
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


def install_asset_bundle(asset_root: Path, dry_run: bool = False) -> list[dict]:
    ensure_dir(asset_root)
    print(f"\n[Installer] Canonical asset root: {asset_root}")
    results = install_assets(ALL_ASSET_IDS, asset_root=asset_root, dry_run=dry_run)
    for item in results:
        action = item["action"]
        if action == "skipped":
            print(f"  - exists, skipping: {item['target_path']}")
        elif action == "would_materialize":
            print(f"  - would copy existing asset: {item['source_path']} -> {item['target_path']}")
        elif action == "materialized":
            print(f"  - copied existing asset: {item['source_path']} -> {item['installed_path']}")
        elif action == "would_download":
            print(f"  - would download: {item['asset_id']} -> {item['target_path']}")
        else:
            print(f"  - downloaded: {item['asset_id']} -> {item['installed_path']}")
    manifest_path = write_asset_manifest(results, asset_root=asset_root, dry_run=dry_run)
    if dry_run:
        print(f"[Installer] Would write asset manifest: {manifest_path}")
    else:
        print(f"[Installer] Wrote asset manifest: {manifest_path}")
    return results


def verify_asset_bundle(asset_root: Path) -> bool:
    report = asset_bundle_report(asset_root=asset_root)
    print(f"\n[Verify] Canonical asset root: {report['asset_root']}")
    for item in report["assets"]:
        status = "[OK]" if item["in_bundle"] else "[MISS]"
        resolved = f" resolved_from={item['resolved_from']}" if item["resolved_from"] != "missing" else ""
        print(f"{status} {item['asset_id']} -> {item['expected_path']}{resolved}")
    if report["missing_in_bundle"]:
        print("\nMissing from canonical bundle:")
        for asset_id in report["missing_in_bundle"]:
            print(f" - {asset_id}")
        print("Run: python installer.py install-all")
        return False
    print("\n[Verify] Offline bundle is complete.")
    return True


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
    parser = argparse.ArgumentParser(description="ChemEagle offline asset installer")
    parser.add_argument(
        "command",
        nargs="?",
        default="install-all",
        choices=["install-all", "verify", "repair"],
        help="install-all downloads the full offline bundle; verify checks bundle completeness; repair fills missing bundle assets.",
    )
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT), help="Canonical asset bundle directory")
    parser.add_argument("--repos-dir", default="./external", help="Directory for optional cloned repos")
    parser.add_argument("--provider", choices=["azure", "openai", "anthropic", "lmstudio", "openai_compatible", "local_openai"], help="Skip provider prompt")
    parser.add_argument("--env-file", default=".env.chemeagle", help="Where to write provider environment variables")
    parser.add_argument("--no-clone", action="store_true", help="Skip optional repository cloning")
    parser.add_argument("--skip-provider-setup", action="store_true", help="Skip provider environment prompts")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    asset_root = Path(args.asset_root).expanduser()
    if not asset_root.is_absolute():
        asset_root = (project_root / asset_root).resolve()
    repos_dir = (project_root / args.repos_dir).resolve()
    env_file = (project_root / args.env_file).resolve()
    activation_script = (project_root / "load_chemeagle_env.sh").resolve()

    print("=== ChemEagle Installer ===")
    print(f"[Installer] Command: {args.command}")
    print(f"[Installer] Effective asset root: {asset_root}")
    if args.asset_root == str(DEFAULT_ASSET_ROOT):
        print(f"[Installer] Default asset root from environment would be: {get_asset_root()}")

    if args.command in {"install-all", "repair"}:
        install_asset_bundle(asset_root, dry_run=args.dry_run)
        verify_asset_bundle(asset_root)
    elif args.command == "verify":
        ok = verify_asset_bundle(asset_root)
        raise SystemExit(0 if ok else 1)

    if not args.no_clone and args.command in {"install-all", "repair"} and ask_yes_no("Clone optional helper repositories?", default=False):
        ensure_dir(repos_dir)
        for name, url in OPTIONAL_REPOS.items():
            if ask_yes_no(f"Clone {name}?", default=True):
                clone_repo(name, url, repos_dir, dry_run=args.dry_run)

    if args.command in {"install-all", "repair"} and not args.skip_provider_setup and ask_yes_no("Configure LLM provider environment variables?", default=True):
        provider = args.provider or ask_input(
            "Select provider (azure/openai/anthropic/lmstudio/openai_compatible/local_openai)",
            default="azure",
        )
        env_vars = build_provider_env(provider)
        write_env_file(env_vars, env_file, dry_run=args.dry_run)
        write_activation_script(env_file, activation_script, dry_run=args.dry_run)

    print("\n[Installer] Done.")
    if args.command in {"install-all", "repair"}:
        print(f"Use: source {activation_script}  # to load provider env vars")


if __name__ == "__main__":
    main()
