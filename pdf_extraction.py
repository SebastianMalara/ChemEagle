import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pdf_images"


def load_config(config_file):
    """Load a JSON config file and resolve relative paths from its own directory."""
    config_path = Path(config_file).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    base_dir = config_path.parent
    for key in ("default_image_dir", "default_json_dir", "default_graph_dir", "image_dir"):
        value = config.get(key)
        if not value:
            continue
        path_value = Path(value).expanduser()
        if not path_value.is_absolute():
            path_value = (base_dir / path_value).resolve()
        config[key] = str(path_value)
    return config


def run_pdf(config_file=None, pdf_dir=None, image_dir=None, model_size="large"):
    """Run VisualHeist table/figure extraction for a single PDF."""
    from pdfmodel.methods import _pdf_to_figures_and_tables

    config = load_config(config_file) if config_file else {}
    resolved_pdf_path = Path(pdf_dir).expanduser().resolve() if pdf_dir else None

    if resolved_pdf_path is None:
        raise ValueError("pdf_dir is required")
    if not resolved_pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {resolved_pdf_path}")

    resolved_model_size = (model_size or config.get("model_size") or "large").strip().lower()
    if resolved_model_size not in {"base", "large"}:
        raise ValueError(f"Unsupported PDF_MODEL_SIZE: {resolved_model_size}")

    resolved_image_dir = image_dir or config.get("image_dir") or config.get("default_image_dir")
    if resolved_image_dir:
        resolved_image_dir = Path(resolved_image_dir).expanduser().resolve()
    else:
        resolved_image_dir = DEFAULT_OUTPUT_DIR.resolve()
    resolved_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing the PDF: {resolved_pdf_path}")
    print(f"Using {'LARGE' if resolved_model_size == 'large' else 'BASE'} model")
    _pdf_to_figures_and_tables(
        str(resolved_pdf_path),
        str(resolved_image_dir),
        large_model=(resolved_model_size == "large"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract figures and tables from a PDF with VisualHeist.")
    parser.add_argument("--pdf", dest="pdf_dir", required=True, help="Path to the input PDF file")
    parser.add_argument("--image-dir", dest="image_dir", default=None, help="Directory to save extracted crops")
    parser.add_argument("--config-file", dest="config_file", default=None, help="Optional JSON config file")
    parser.add_argument(
        "--model-size",
        dest="model_size",
        default="large",
        choices=["base", "large"],
        help="VisualHeist model size",
    )
    args = parser.parse_args()
    run_pdf(
        config_file=args.config_file,
        pdf_dir=args.pdf_dir,
        image_dir=args.image_dir,
        model_size=args.model_size,
    )
