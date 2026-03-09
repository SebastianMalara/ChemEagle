# ChemEAGLE


![visualization](examples/overview.png)
<div align="center",width="100">
</div> 

This is the official code of the [paper](https://arxiv.org/abs/2507.20230) "A Multi-Agent System Enables Versatile Information Extraction from the Chemical Literature".

## :sparkles: Highlights
<p align="justify">
In this work, we present ChemEAGLE, a multimodal large language model (MLLM)-based multi-agent system that integrates diverse chemical information extraction tools to extract multimodal chemical reactions. By integrating ten expert-designed tools and seven chemical information extraction agents, ChemEAGLE not only processes individual modalities but also utilizes MLLMs' reasoning capabilities to unify extracted data, ensuring more accurate and comprehensive reaction representations. By bridging multimodal gaps, our approach significantly improves automated chemical knowledge extraction, facilitating more robust AI-driven chemical research.

[comment]: <> ()
![visualization](examples/chemeagle1.png)
<div align="center"> An example workflow of ChemEAGLE. Each agent handles a specific sub-task, from reaction template parsing and molecular recognition to SMILES reconstruction and condition role interpretation, ensuring accurate, structured chemical data integration. </div>
  
### 🧩 Agents Overview
| Agent Name                                          | Category            | Main Function                                                       |
| --------------------------------------------------- | ------------------- | ------------------------------------------------------------------- |
| **Planner**                                   | Planning  | Analyzes input, plans extraction steps, assigns sub-tasks to agents |
| **Plan Observer**                          | Validation | Monitors extraction workflow, ensures logical plan                  |
| **Action Observer**                           | Validation | Oversees agent actions, validates consistency and correctness       |
| **Reaction Template Parsing Agent**                 | Extraction          | Parses reaction templates, integrates R-group substitutions         |
| **Molecular Recognition Agent**                     | Extraction          | Detects and interprets all molecules in graphics                      |
| **Structure-based Table R-group Substitution Agent** | Extraction          | Substitutes R-groups and reconstructs reactant SMILES from product variant structure-based tables        |
| **Text-based Table R-group Substitution Agent**     | Extraction          | Substitutes R-groups and reconstructs SMILES from text-based tables  |
| **Condition Interpretation Agent**                  | Extraction          | Extracts and categorizes reaction conditions (solvent, temp, etc.)  |
| **Text Extraction Agent**                           | Extraction          | Extracts and aligns reaction info from associated texts             |
| **Data Structure Agent**                            | Output         | Compiles structured output for downstream applications              |


### 🛠️ Toolkits Used in ChemEAGLE
| Tool Name               | Category                          | Description                                            |
| ----------------------- | --------------------------------- | ------------------------------------------------------ |
| **TesseractOCR**        | Computer Vision                   | Optical character recognition for text in graphics       |
| **TableParser**         | Computer Vision                   | Table structure detection and parsing                  |
| **MolDetector**         | Computer Vision                   | Locates and segments molecules within graphics           |
| **MolNER**              | Text-based Information Extraction | Chemical named entity recognition from text            |
| **ChemRxnExtractor**    | Text-based Information Extraction | Extracts chemical reactions and roles from text        |
| **Image2Graph**         | Molecular Recognition             | Converts molecular sub-images to graph representations     |
| **Graph2SMILES**        | Molecular Recognition             | Converts molecular graphs to SMILES strings            |
| **SMILESReconstructor** | Molecular Recognition             | Reconstructs reactant SMILES from product variants     |
| **RxnImgParser**        | Reaction Image Parsing            | Parsing reaction template images into bounding boxes and components |
| **RxnConInterpreter**   | Reaction Image Parsing            | Assigns condition roles to extracted condition text     |




## :rocket: Using the code for ChemEAGLE
### Quick installer (models + API configuration)

You can run the interactive installer to download required model files, optionally clone helper repos, and generate an LLM environment file:

```bash
python installer.py
```

What it does:
- Downloads required model weights from `CYF200127/ChemEAGLEModel` into `./models` (and mirrors key files in project root for default paths).
- Optionally clones helper repositories (for example, ChemRxnExtractor) into `./external`.
- Guides you through selecting an LLM provider and writes `.env.chemeagle` plus `load_chemeagle_env.sh`.

Then load your LLM settings:

```bash
source ./load_chemeagle_env.sh
```

Useful options:

```bash
python installer.py --provider openai
python installer.py --model-dir ./models --repos-dir ./external
python installer.py --dry-run
```

### Using the code
Clone the following repositories:
```
git clone https://github.com/CYF2000127/ChemEagle
```
#### Option A: Using Azure OpenAI (Cloud-based)

1. First create and activate a [conda](https://numdifftools.readthedocs.io/en/stable/how-to/create_virtual_env_with_conda.html) environment with the following command in a Linux, Windows, or MacOS environment (Linux is the most recommended):
```bash
conda create -n chemeagle python=3.10
conda activate chemeagle
```

2. Then install requirements:
```bash
pip install -r requirements.txt
```

3. Download the necessary [models](https://huggingface.co/CYF200127/ChemEAGLEModel/tree/main) and put in the main path.

4. Configure the LLM provider via environment variables.

- **Azure OpenAI** (default):
```bash
export LLM_PROVIDER=azure
export API_KEY=your-azure-openai-api-key
export AZURE_ENDPOINT=your-azure-endpoint
export API_VERSION=your-api-version
export LLM_MODEL=gpt-5-mini
```

- **OpenAI**:
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-openai-api-key
export LLM_MODEL=gpt-5-mini
# optional
# export OPENAI_BASE_URL=https://api.openai.com/v1
```

- **Anthropic**:
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-anthropic-api-key
export LLM_MODEL=claude-3-5-sonnet-latest
```


- **LM Studio / OpenAI-compatible local server**:
```bash
export LLM_PROVIDER=lmstudio
export LLM_BASE_URL=http://127.0.0.1:1234/v1
export LLM_API_KEY=lm-studio  # LM Studio typically accepts any non-empty key
export LLM_MODEL=your-local-model-name
```

5. Run the following code to extract machine-readable chemical data from chemical graphics:
```python
from main import ChemEagle
image_path = './examples/1.png'
results = ChemEagle(image_path)
print(results)
```

6. Alternatively, run the following code to extract machine-readable chemical data from chemical literature (PDF files) directly:
```python
import os
from main import ChemEagle
from pdf_extraction import run_pdf
pdf_path   = 'your/pdf/path'
output_dir = 'your/output/dir'
run_pdf(pdf_dir=pdf_path, image_dir=output_dir)
results = []
for fname in sorted(os.listdir(output_dir)):
    if not fname.lower().endswith('.png'):
        continue
    img_path = os.path.join(output_dir, fname)
    try:
        r = ChemEagle(img_path)
        r['image_name'] = fname
        results.append(r)
    except Exception as e:
        results.append({'image_name': fname, 'error': str(e)})
print(results)
```

#### Option B: Using ChemEagle_OS (Local Deployment with vLLM)

**ChemEagle_OS** is an open-source version that runs locally using vLLM, eliminating the need for cloud API keys.

##### Prerequisites
- NVIDIA GPU with CUDA support (recommended)
- Docker installed (for Windows vLLM deployment)
- Download the Qwen3-VL model weights (e.g., `Qwen3-VL-32B-Instruct-AWQ`) from [HuggingFace](https://huggingface.co/QuantTrio/Qwen3-VL-32B-Instruct-AWQ)

1. Setup Python Environment
```bash
conda create -n chemeagle python=3.10
conda activate chemeagle
pip install -r requirements.txt
```

2. Download the necessary [models](https://huggingface.co/CYF200127/ChemEAGLEModel/tree/main) and put in the main path.

3. Deploy vLLM Server

**For Linux:**
```
pip install vllm
vllm serve /path/to/Qwen3-VL-32B-Instruct-AWQ \
    --port 8000 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 27200 \
    --limit-mm-per-prompt video=0
```

**For Windows (PowerShell):**

```powershell
docker run -d --gpus all `
    -p 8000:8000 `
    -v /path/to/Qwen3-VL-32B-Instruct-AWQ:/models/Qwen3-VL-32B-Instruct-AWQ `
    --name vllm-server `
    vllm/vllm-openai:latest `
    --model /models/Qwen3-VL-32B-Instruct-AWQ `
    --port 8000 `
    --trust-remote-code `
    --enable-auto-tool-choice `
    --tool-call-parser hermes `
    --max-model-len 27200 `
    --limit-mm-per-prompt.video 0
```

**Note:** 
- Replace `/path/to/Qwen3-VL-32B-Instruct-AWQ` (Linux) or `F:/chemeagle/Qwen3-VL-32B-Instruct-AWQ` (Windows) with your actual model path.
- The vLLM server will be available at `http://localhost:8000/v1` by default.



4. After the vLLM server is running, you can use the open source version of ChemEAGLE as follows:

```python
from main import ChemEagle_OS

# Using default local vLLM server (http://localhost:8000/v1)
image_path = './examples/1.png'
results = ChemEagle_OS(image_path)
print(results)
```

5. Alternatively, run the following code to extract machine-readable chemical data from chemical literature (PDF files) directly:
```python
import os
from main import ChemEagle_OS
from pdf_extraction import run_pdf
pdf_path   = 'your/pdf/path'
output_dir = 'your/output/dir'
run_pdf(pdf_dir=pdf_path, image_dir=output_dir)
results = []
for fname in sorted(os.listdir(output_dir)):
    if not fname.lower().endswith('.png'):
        continue
    img_path = os.path.join(output_dir, fname)
    try:
        r = ChemEagle_OS(img_path)
        r['image_name'] = fname
        results.append(r)
    except Exception as e:
        results.append({'image_name': fname, 'error': str(e)})
print(results)
```


### Self-hosted GUI (local deployment)

A built-in self-hosted GUI is included in `gui_app.py`. It allows you to:
- load env values from `.env.chemeagle` and edit them in the UI
- optionally save updated env values back to file
- upload an image or PDF
- run the existing pipeline (`ChemEagle` cloud mode or `ChemEagle_OS` local mode)

Start it with:

```bash
python gui_app.py
```

Then open: `http://localhost:7860`

### Benchmarking
Benchmark datasets, predictions, and ground truth can be found in our [Huggingface Repo](https://huggingface.co/datasets/CYF200127/ChemEagle/blob/main/Dataset.zip).

## 🤗 Chemical information extraction using [ChemEAGLE.Web](https://app.chemeagle.net/) 

Go to our [ChemEAGLE.Web app demo](https://app.chemeagle.net/) to directly use our tool online! (Use for both image and PDF input)

When the input is a multimodal chemical reaction graphic:
![visualization](examples/reaction9.png)
<div align="center",width="100">
</div> 

The output dictionary is a complete machine-readable reaction list with reactant SMILES, product SMILES, detailed conditions and additional information for every reaction in the graphics:

``` 
{"reactions":[
{"reaction_id":"0_1","reactants":[{"smiles":"[Ar]C([R])=C=O","label":"1"},{"smiles":"Cc1ccc(S(=O)(=O)N2OC2c2ccccc2Cl)cc1","label":"2"}],
"conditions":[{"role":"reagent","text":"10 mol% B17 or B27","smiles":"C(C=CC=C1)=C1C[N+]2=CN3[C@H](C(C4=CC=CC=C4)(C5=CC=CC=C5)O[Si](C)(C)C(C)(C)C)CCC3=N2.F[B-](F)(F)F","label":"B17"},
{"role":"reagent","text":"10 mol% B17 or B27","smiles":"CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F","label":"B27"},
{"role":"reagent","text":"10 mol% Cs2CO3","smiles":"[Cs+].[Cs+].[O-]C(=O)[O-]"},{"role":"solvent","text":"PhMe","smiles":"Cc1ccccc1"},{"role":"temperature","text":"rt"},{"role":"yield","text":"38 - 78%"}],
"products":[{"smiles":"[Ar]C1([R])O[C@H](c2ccccc2Cl)N(S(=O)(=O)c2ccc(C)cc2)C1=O","label":"3"}]},

{"reaction_id":"1_1","reactants":[{"smiles":"CCC(=C=O)c1ccccc1","label":"1a"},{"smiles":"Cc1ccc(S(=O)(=O)N2OC2c2ccccc2Cl)cc1","label":"2a"}],
"conditions":[{"role":"reagent","text":"10 mol% B17 or B27","smiles":"C(C=CC=C1)=C1C[N+]2=CN3[C@H](C(C4=CC=CC=C4)(C5=CC=CC=C5)O[Si](C)(C)C(C)(C)C)CCC3=N2.F[B-](F)(F)F","label":"B17"},
{"role":"reagent","text":"10 mol% B17 or B27","smiles":"CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F","label":"B27"},
{"role":"reagent","text":"10 mol% Cs2CO3","smiles":"[Cs+].[Cs+].[O-]C(=O)[O-]"},{"role":"solvent","text":"PhMe","smiles":"Cc1ccccc1"},{"role":"temperature","text":"rt"},{"role":"yield","text":"71%"}],
"products":[{"smiles":"CC[C@@]1(c2ccccc2)O[C@H](c2ccccc2Cl)N(S(=O)(=O)c2ccc(C)cc2)C1=O","label":"3a"}],"additional_info":[{"text":"14:1 dr, 91% ee"}]},

{"reaction_id":"2_1",... ###More detailed reactions}
],
"text_extraction":"..."
}
```
The input can be any chemical graphics; feel free to try more examples! 
(NOTE: If the local CPU is larger than 8 cores, it is recommended to use the local code, which will be faster than the web demo. It will also be significantly faster using a GPU. The current version runs by default on the CPU for stability. The online version may not be as updated as the GitHub version.)

![visualization](examples/reaction5.png)
![visualization](examples/reaction6.png)
![visualization](examples/reaction1.jpg)
![visualization](examples/reaction2.png)
![visualization](examples/reaction4.png)

## :warning: Acknowledgement
1. We use api_version="2024-10-21" with the HKUST Azure OpenAI endpoint as our official version.
2. Our code is based on [MolNexTR](https://github.com/CYF2000127/MolNexTR), [MolScribe](https://github.com/thomas0809/MolScribe), [RxnIM](https://github.com/CYF2000127/RxnIM), [RxnScribe](https://github.com/thomas0809/RxNScribe), [ChemNER](https://github.com/Ozymandias314/ChemIENER), [ChemRxnExtractor](https://github.com/jiangfeng1124/ChemRxnExtractor), [AutoAgents](https://github.com/Link-AGI/AutoAgents), and [Azure OpenAI](https://azure.microsoft.com/).


## 🖥️ Dell XPS (Ubuntu + NVIDIA dGPU) deployment guide

### Readiness verdict
ChemEAGLE is **ready to run on a Dell XPS with Ubuntu and NVIDIA discrete graphics**.

### Recommended target environment
- Ubuntu 22.04/24.04 LTS
- NVIDIA proprietary driver with a working `nvidia-smi`
- Python 3.10
- Optional for local high-throughput inference: CUDA-compatible stack + vLLM

### 0) Clone repository

```bash
git clone https://github.com/CYF2000127/ChemEagle
cd ChemEagle
```

### 1) Install Ubuntu system packages

```bash
sudo apt update
sudo apt install -y \
  git build-essential python3-dev python3-venv \
  poppler-utils tesseract-ocr libgl1 libglib2.0-0
```

Why these packages:
- `python3-venv` is needed if you use Python venv (recommended alternative to conda).
- `poppler-utils` is needed by PDF extraction flow (`pdf2image`).
- `tesseract-ocr` is needed for OCR.
- `libgl1` and `libglib2.0-0` are often required by OpenCV runtime.

### 2) Verify NVIDIA driver and GPU visibility

```bash
nvidia-smi
```

If this fails, fix NVIDIA driver installation first.

### 3) Create Python environment (choose one)

#### Option A: Conda

```bash
conda create -n chemeagle python=3.10 -y
conda activate chemeagle
python -m pip install --upgrade pip setuptools wheel
```

#### Option B: Python venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 4) Install Python dependencies

> ⚠️ **Important:** This repository is currently pinned for **Python 3.10**.
> If you use Python 3.12/3.13, `pip install -r requirements.txt` may fail (for example with `torch==2.2.0` not found).

```bash
pip install -r requirements.txt
```

Optional (GPU PyTorch wheel, example CUDA 12.1):

```bash
pip install --upgrade --force-reinstall torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify torch install:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
```

### 5) Run ChemEAGLE installer (models + env helpers)

Interactive installer:

```bash
python installer.py
```

Non-interactive examples:

```bash
python installer.py --provider openai
python installer.py --provider azure
python installer.py --provider anthropic
python installer.py --model-dir ./models --repos-dir ./external
```

The installer downloads model files and writes:
- `.env.chemeagle`
- `load_chemeagle_env.sh`

Load environment variables:

```bash
source ./load_chemeagle_env.sh
```

### 6) Run preflight check before go-live

```bash
python scripts/preflight_check.py
```

Preflight verifies platform basics, GPU visibility, and required model/prompt files.

### 7) Start ChemEAGLE

#### Mode A: Cloud LLM mode (`ChemEagle`)

Use one provider configuration (Azure/OpenAI/Anthropic), then run:

```bash
python - <<'PY'
from main import ChemEagle
print(ChemEagle('./examples/1.png'))
PY
```

#### Mode B: Local open-source mode (`ChemEagle_OS` + vLLM)

Install and start vLLM:

```bash
pip install vllm
vllm serve /path/to/Qwen3-VL-32B-Instruct-AWQ \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 27200 \
  --limit-mm-per-prompt video=0
```

Then run ChemEAGLE_OS:

```bash
python - <<'PY'
from main import ChemEagle_OS
print(ChemEagle_OS('./examples/1.png'))
PY
```

### 7.5) Optional: launch self-hosted GUI

```bash
python gui_app.py
```

Open `http://localhost:7860` on your Dell XPS.

### 8) Run on PDF input

```bash
python - <<'PY'
import os
from main import ChemEagle
from pdf_extraction import run_pdf

pdf_path='your/pdf/path'
output_dir='your/output/dir'
run_pdf(pdf_dir=pdf_path, image_dir=output_dir)

results=[]
for fname in sorted(os.listdir(output_dir)):
    if fname.lower().endswith('.png'):
        path=os.path.join(output_dir, fname)
        out=ChemEagle(path)
        out['image_name']=fname
        results.append(out)
print(results)
PY
```

### 9) Go-live troubleshooting checklist

- If `nvidia-smi` fails: reinstall/fix NVIDIA driver.
- If `python scripts/preflight_check.py` reports missing models: run `python installer.py`.
- If `pip install -r requirements.txt` fails with `No matching distribution found for torch==2.2.0`, you are likely on Python 3.12/3.13. Recreate the env with Python 3.10 and reinstall.
- If OpenCV import fails: verify `libgl1` and `libglib2.0-0` are installed.
- If PDF conversion fails: verify `poppler-utils` is installed and in `PATH`.
- If OCR quality is poor: verify `tesseract-ocr` and language packs.
- If local model startup OOMs: use a smaller quantized model, lower context, or use cloud mode.

## 🍎 Apple Silicon (M1/M2/M3) deployment guide (uv + MPS/Metal)

This section is for running ChemEAGLE on Apple Silicon Macs (for example, M1 Pro) using **uv-managed virtual environments**.

### 0) Clone repository

```bash
git clone https://github.com/CYF2000127/ChemEagle
cd ChemEagle
```

### 1) Install system dependencies

Install Homebrew if needed, then:

```bash
brew install tesseract poppler
```

Why:
- `tesseract` is needed for OCR.
- `poppler` is needed for PDF to image conversion (`pdf2image`).

### 2) Create a Python 3.10 environment with uv

> ⚠️ ChemEAGLE dependencies are pinned around Python 3.10.

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
```

### 3) Install Apple Silicon requirements (Metal-enabled PyTorch)

Use the dedicated requirements file:

```bash
uv pip install -r requirements.apple-silicon.txt
```

### 4) Verify Metal (MPS) is visible to PyTorch

```bash
python -c "import torch; print('torch', torch.__version__); print('mps_built', torch.backends.mps.is_built()); print('mps_available', torch.backends.mps.is_available())"
```

Expected: `mps_built=True` and (typically) `mps_available=True`.

### 5) Install models and configure LLM provider

```bash
python installer.py
source ./load_chemeagle_env.sh
```

### 6) Run a quick smoke test

```bash
python - <<'PY'
from main import ChemEagle
print(ChemEagle('./examples/1.png'))
PY
```

### 7) Optional: run GUI locally

```bash
python gui_app.py
```

Then open `http://localhost:7860` in your browser.
