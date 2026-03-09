#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import gradio as gr
except ImportError as e:
    raise ImportError("gradio is required for gui_app.py. Install dependencies with: pip install -r requirements.txt") from e

ENV_FILE_DEFAULT = Path('.env.chemeagle')

ENV_KEYS = [
    'LLM_PROVIDER',
    'LLM_MODEL',
    'API_KEY',
    'AZURE_ENDPOINT',
    'API_VERSION',
    'OPENAI_API_KEY',
    'OPENAI_BASE_URL',
    'ANTHROPIC_API_KEY',
    'VLLM_BASE_URL',
    'VLLM_API_KEY',
]


def parse_env_file(env_path: Path) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if not env_path.exists():
        return vals
    for line in env_path.read_text(encoding='utf-8').splitlines():
        raw = line.strip()
        if not raw or raw.startswith('#') or '=' not in raw:
            continue
        k, v = raw.split('=', 1)
        vals[k.strip()] = v.strip()
    return vals


def merged_env_values(env_path: Path) -> Dict[str, str]:
    from_file = parse_env_file(env_path)
    out: Dict[str, str] = {}
    for k in ENV_KEYS:
        out[k] = os.getenv(k, from_file.get(k, ''))
    return out


def save_env_file(env_path: Path, values: Dict[str, str]) -> str:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={values.get(k, '')}" for k in ENV_KEYS if values.get(k, '')]
    env_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return f"Saved {env_path} with {len(lines)} keys."


def apply_runtime_env(values: Dict[str, str]) -> None:
    for k, v in values.items():
        if v:
            os.environ[k] = v


def _run_on_image(image_path: str, mode: str) -> dict:
    from main import ChemEagle, ChemEagle_OS
    if mode == 'cloud':
        return ChemEagle(image_path)
    return ChemEagle_OS(image_path)


def _run_on_pdf(pdf_path: str, mode: str) -> List[dict]:
    from pdf_extraction import run_pdf
    with tempfile.TemporaryDirectory(prefix='chemeagle_pdf_') as tmpdir:
        run_pdf(pdf_dir=pdf_path, image_dir=tmpdir)
        results: List[dict] = []
        for fname in sorted(os.listdir(tmpdir)):
            if not fname.lower().endswith('.png'):
                continue
            img_path = os.path.join(tmpdir, fname)
            try:
                r = _run_on_image(img_path, mode)
                r['image_name'] = fname
                results.append(r)
            except Exception as e:
                results.append({'image_name': fname, 'error': str(e)})
        return results


def run_pipeline(
    env_path_str: str,
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
    upload,
    save_env: bool,
) -> Tuple[str, str]:
    env_path = Path(env_path_str).expanduser() if env_path_str else ENV_FILE_DEFAULT
    values = {
        'LLM_PROVIDER': llm_provider,
        'LLM_MODEL': llm_model,
        'API_KEY': api_key,
        'AZURE_ENDPOINT': azure_endpoint,
        'API_VERSION': api_version,
        'OPENAI_API_KEY': openai_api_key,
        'OPENAI_BASE_URL': openai_base_url,
        'ANTHROPIC_API_KEY': anthropic_api_key,
        'VLLM_BASE_URL': vllm_base_url,
        'VLLM_API_KEY': vllm_api_key,
    }

    apply_runtime_env(values)
    status_bits = [f"Runtime env applied from form. mode={mode}"]

    if save_env:
        status_bits.append(save_env_file(env_path, values))

    if upload is None:
        return '\n'.join(status_bits + ['No file uploaded.']), '{}'

    file_path = upload if isinstance(upload, str) else getattr(upload, 'name', None)
    if not file_path:
        return '\n'.join(status_bits + ['Could not resolve upload path.']), '{}'

    suffix = Path(file_path).suffix.lower()
    try:
        if suffix == '.pdf':
            result = _run_on_pdf(file_path, mode)
        elif suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}:
            result = _run_on_image(file_path, mode)
        else:
            return '\n'.join(status_bits + [f'Unsupported file type: {suffix}']), '{}'
    except Exception as e:
        return '\n'.join(status_bits + [f'Pipeline failed: {e}']), '{}'

    return '\n'.join(status_bits + [f'Completed for file: {Path(file_path).name}']), json.dumps(result, ensure_ascii=False, indent=2)


def build_app() -> gr.Blocks:
    vals = merged_env_values(ENV_FILE_DEFAULT)
    with gr.Blocks(title='ChemEagle Self-Hosted GUI') as demo:
        gr.Markdown('# ChemEagle Self-Hosted GUI')
        gr.Markdown('Configure env, upload image/PDF, and run current ChemEagle pipeline.')

        with gr.Row():
            env_path = gr.Textbox(label='Env file path', value=str(ENV_FILE_DEFAULT), scale=2)
            save_env = gr.Checkbox(label='Save form values to env file', value=True)
            mode = gr.Radio(['cloud', 'local_os'], value='cloud', label='Run mode')

        with gr.Accordion('Environment settings', open=True):
            with gr.Row():
                llm_provider = gr.Textbox(label='LLM_PROVIDER', value=vals.get('LLM_PROVIDER', 'azure'))
                llm_model = gr.Textbox(label='LLM_MODEL', value=vals.get('LLM_MODEL', 'gpt-5-mini'))
                api_version = gr.Textbox(label='API_VERSION', value=vals.get('API_VERSION', '2024-06-01'))
            with gr.Row():
                api_key = gr.Textbox(label='API_KEY (Azure)', type='password', value=vals.get('API_KEY', ''))
                azure_endpoint = gr.Textbox(label='AZURE_ENDPOINT', value=vals.get('AZURE_ENDPOINT', ''))
            with gr.Row():
                openai_api_key = gr.Textbox(label='OPENAI_API_KEY', type='password', value=vals.get('OPENAI_API_KEY', ''))
                openai_base_url = gr.Textbox(label='OPENAI_BASE_URL', value=vals.get('OPENAI_BASE_URL', ''))
            with gr.Row():
                anthropic_api_key = gr.Textbox(label='ANTHROPIC_API_KEY', type='password', value=vals.get('ANTHROPIC_API_KEY', ''))
                vllm_base_url = gr.Textbox(label='VLLM_BASE_URL', value=vals.get('VLLM_BASE_URL', 'http://localhost:8000/v1'))
                vllm_api_key = gr.Textbox(label='VLLM_API_KEY', type='password', value=vals.get('VLLM_API_KEY', 'EMPTY'))

        upload = gr.File(label='Upload image (.png/.jpg/...) or PDF', file_count='single')
        run_btn = gr.Button('Run ChemEagle', variant='primary')

        status = gr.Textbox(label='Status', lines=4)
        output = gr.Code(label='JSON output', language='json')

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                env_path,
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
                upload,
                save_env,
            ],
            outputs=[status, output],
        )
    return demo


if __name__ == '__main__':
    app = build_app()
    app.launch(server_name='0.0.0.0', server_port=int(os.getenv('PORT', '7860')))
