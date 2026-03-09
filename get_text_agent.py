from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import time
from typing import Any, Optional

import pytesseract
from PIL import Image
from chemiener import ChemNER
from chemrxnextractor import RxnExtractor
from huggingface_hub import hf_hub_download
from openai import OpenAI

from llm_wrapper import LLMWrapper
from runtime_device import (
    easyocr_uses_acceleration,
    resolve_ocr_backend,
    resolve_torch_device,
    warn_once,
)

model_dir = "./cre_models_v0.1"
ckpt_path = "./ner.ckpt"

_CHEMNER_MODELS: dict[str, ChemNER] = {}
_RXN_EXTRACTORS: dict[str, RxnExtractor] = {}
_EASYOCR_READERS: dict[tuple[tuple[str, ...], str], Any] = {}
_OCR_TEXT_CACHE: dict[tuple[str, ...], str] = {}

def _get_azure_client():
    return LLMWrapper.from_env(default_model=os.getenv("LLM_MODEL") or "gpt-5-mini").as_chat_completion_client()

def _resolve_text_agent_model(default_model: str = "gpt-5-mini") -> str:
    return os.getenv("TEXT_AGENT_MODEL") or os.getenv("LLM_MODEL") or default_model


def _resolve_ocr_llm_model(default_model: str = "gpt-5-mini") -> str:
    return os.getenv("OCR_LLM_MODEL") or _resolve_text_agent_model(default_model)


def _get_text_agent_client():
    return LLMWrapper.from_env(default_model=_resolve_text_agent_model()).as_chat_completion_client()


def _get_rxn_extractor() -> RxnExtractor:
    device = resolve_torch_device()
    cache_key = "cuda" if device.type == "cuda" else "cpu"
    if cache_key not in _RXN_EXTRACTORS:
        if device.type == "mps":
            warn_once("Text reaction extractor does not support MPS directly. Using CPU for RxnExtractor.")
        _RXN_EXTRACTORS[cache_key] = RxnExtractor(model_dir, use_cuda=(device.type == "cuda"))
    return _RXN_EXTRACTORS[cache_key]


def _get_chemner_model() -> ChemNER:
    device = resolve_torch_device()
    cache_key = device.type
    if cache_key in _CHEMNER_MODELS:
        return _CHEMNER_MODELS[cache_key]

    local_ckpt_path = ckpt_path
    if not os.path.exists(local_ckpt_path):
        local_ckpt_path = hf_hub_download("CYF200127/ChemEAGLEModel", "ner.ckpt")

    _CHEMNER_MODELS[cache_key] = ChemNER(local_ckpt_path, device=device)
    return _CHEMNER_MODELS[cache_key]


def find_tesseract_cmd() -> str | None:
    current_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if current_cmd and os.path.exists(current_cmd):
        return current_cmd

    env_tesseract_cmd = os.getenv("TESSERACT_CMD") or os.getenv("CHEMEAGLE_TESSERACT_CMD")
    if env_tesseract_cmd:
        normalized_env_cmd = os.path.normpath(env_tesseract_cmd)
        if os.path.exists(normalized_env_cmd):
            return normalized_env_cmd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        shutil.which("tesseract"),
        os.path.join(script_dir, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.path.dirname(script_dir), "Tesseract-OCR", "tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        r"C:\Users\Administrator\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]

    for path in possible_paths:
        if not path:
            continue
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            return normalized_path
    return None


def configure_tesseract() -> bool:
    tesseract_cmd = find_tesseract_cmd()
    if not tesseract_cmd:
        warn_once("Tesseract OCR executable was not found. Select EasyOCR or LLM vision, or set TESSERACT_CMD.")
        return False

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return True


def _normalize_ocr_text(raw_text: str) -> str:
    return "\n".join(line.rstrip() for line in raw_text.splitlines()).strip()


def _paragraph_from_raw_text(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return " ".join(lines).strip()


def _parse_ocr_langs_for_easyocr() -> list[str]:
    raw_lang = (os.getenv("OCR_LANG") or "eng").strip()
    mapping = {
        "eng": "en",
        "chi_sim": "ch_sim",
        "chi_tra": "ch_tra",
        "deu": "de",
        "fra": "fr",
        "spa": "es",
        "ita": "it",
        "por": "pt",
        "rus": "ru",
        "jpn": "ja",
        "kor": "ko",
    }

    langs: list[str] = []
    for token in re.split(r"[+,;\s]+", raw_lang):
        normalized = token.strip().lower()
        if not normalized:
            continue
        langs.append(mapping.get(normalized, normalized))
    return langs or ["en"]


def _get_easyocr_reader():
    import easyocr

    device = resolve_torch_device()
    use_gpu = easyocr_uses_acceleration(device)
    cache_key = (tuple(_parse_ocr_langs_for_easyocr()), device.type if use_gpu else "cpu")

    if cache_key not in _EASYOCR_READERS:
        _EASYOCR_READERS[cache_key] = easyocr.Reader(
            list(cache_key[0]),
            gpu=use_gpu,
            verbose=False,
        )
    return _EASYOCR_READERS[cache_key]


def _ocr_text_via_tesseract(image_path: str) -> str:
    if not configure_tesseract():
        raise RuntimeError("Tesseract OCR unavailable. Install Tesseract or set TESSERACT_CMD.")

    img = Image.open(image_path)
    lang = os.getenv("OCR_LANG", "eng")
    config = os.getenv("OCR_CONFIG", "").strip()
    kwargs = {"lang": lang}
    if config:
        kwargs["config"] = config
    return pytesseract.image_to_string(img, **kwargs)


def _ocr_text_via_easyocr(image_path: str) -> str:
    reader = _get_easyocr_reader()
    results = reader.readtext(image_path, detail=0, paragraph=False)
    return "\n".join(str(item).strip() for item in results if str(item).strip())


def _ocr_text_via_llm_vision(image_path: str) -> str:
    model_name = _resolve_ocr_llm_model()
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

    with open(image_path, "rb") as handle:
        b64_image = base64.b64encode(handle.read()).decode("utf-8")

    llm = LLMWrapper.from_env(default_model=model_name, scope="ocr")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an OCR engine for chemistry documents. Return only the text visible in the image. "
                "Preserve useful line breaks. Do not explain, summarize, or infer missing content."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the text from this image. Return raw text only."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
            ],
        },
    ]
    return llm.chat_completion_text(messages, model=model_name, temperature=0)


def _extract_image_text(image_path: str) -> tuple[str, str]:
    backend = resolve_ocr_backend()
    cache_key = (
        os.path.abspath(image_path),
        backend,
        os.getenv("OCR_LANG", ""),
        os.getenv("OCR_CONFIG", ""),
        os.getenv("OCR_LLM_MODEL", ""),
        os.getenv("LLM_MODEL", ""),
    )

    if cache_key in _OCR_TEXT_CACHE:
        return _OCR_TEXT_CACHE[cache_key], backend

    if backend == "tesseract":
        raw_text = _ocr_text_via_tesseract(image_path)
    elif backend == "easyocr":
        raw_text = _ocr_text_via_easyocr(image_path)
    elif backend == "llm_vision":
        raw_text = _ocr_text_via_llm_vision(image_path)
    else:
        raise ValueError(f"Unsupported OCR_BACKEND: {backend}")

    normalized = _normalize_ocr_text(raw_text)
    _OCR_TEXT_CACHE[cache_key] = normalized
    return normalized, backend


def _chat_completion_with_json_fallback(client, **kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        error_text = str(e).lower()
        if "response_format" in error_text:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("response_format", None)
            return client.chat.completions.create(**fallback_kwargs)
        raise


def _assistant_message_to_dict(message) -> dict:
    """Convert SDK message objects into plain dicts safe for re-submission."""
    tool_calls_payload = []
    for tool_call in (getattr(message, "tool_calls", None) or []):
        function_payload = getattr(tool_call, "function", None)
        tool_calls_payload.append(
            {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": getattr(function_payload, "name", None),
                    "arguments": getattr(function_payload, "arguments", "{}") or "{}",
                },
            }
        )

    payload = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
    }
    if tool_calls_payload:
        payload["tool_calls"] = tool_calls_payload
    return payload


def _extract_json_from_text(raw_content: Optional[str]):
    if not raw_content:
        return None

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        pass

    try:
        from get_R_group_sub_agent import extract_json_from_text_with_reasoning
        return extract_json_from_text_with_reasoning(raw_content)
    except Exception:
        return None


def _repair_json_via_model(client, model_name: str, raw_content: str, max_attempts: int = 2):
    """Ask the model to convert malformed output to strict JSON only."""
    candidate = raw_content
    for attempt in range(max_attempts):
        repair_messages = [
            {
                "role": "system",
                "content": "You repair malformed assistant output. Return valid JSON only, no markdown, no explanation.",
            },
            {
                "role": "user",
                "content": (
                    "Convert the following content to valid JSON. Preserve all information. "
                    "If information is missing, wrap the original text as {\"content\": <text>}.\n\n"
                    f"{candidate}"
                ),
            },
        ]
        response = _chat_completion_with_json_fallback(
            client,
            model=model_name,
            messages=repair_messages,
            response_format={"type": "json_object"},
        )
        candidate = response.choices[0].message.content or ""
        parsed = _extract_json_from_text(candidate)
        if parsed is not None:
            if attempt > 0:
                print(f"✓ JSON 修复成功（第 {attempt + 1} 次尝试）")
            return parsed
    return None


def _safe_parse_agent_json(client, model_name: str, raw_content: Optional[str]) -> dict:
    if not raw_content:
        return {}

    parsed = _extract_json_from_text(raw_content)
    if parsed is not None:
        return parsed

    print("⚠️ 检测到非 JSON 输出，正在自动修复...")
    repaired = _repair_json_via_model(client, model_name, raw_content)
    if repaired is not None:
        return repaired

    print("⚠️ JSON 自动修复失败，返回原始文本")
    return {"content": raw_content}


def merge_sentences(sentences):
    """
    合并一个句子片段列表为一个连贯的段落字符串。
    """
    # 去除每条片段前后空白，并剔除空串
    cleaned = [s.strip() for s in sentences if s.strip()]
    # 用空格拼接，恢复成完整段落
    paragraph = [" ".join(cleaned)]
    return paragraph


def split_text_into_sentences(text: str) -> list:
    """
    将文本分割成句子，避免文本过长导致的问题。
    使用简单的标点符号分割，保留句子边界。
    """
    # 按句号、问号、感叹号分割，但保留这些标点
    sentences = re.split(r'([.!?]+)', text)
    # 合并标点和前面的文本
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        if sentence:
            result.append(sentence)
    
    # 如果没有找到句子边界，尝试按换行符分割
    if not result:
        result = [line.strip() for line in text.splitlines() if line.strip()]
    
    # 如果还是没有，返回整个文本（但限制长度）
    if not result:
        # 限制单个句子长度，避免超过模型限制
        max_length = 500  # 字符数限制
        if len(text) > max_length:
            # 按空格分割成更小的块
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            result = chunks
        else:
            result = [text]
    
    return result


def extract_reactions_from_text_in_image(image_path: str) -> dict:
    """
    从化学反应图像中提取文本并识别反应。

    参数：
      image_path: 图像文件路径

    返回：
      {
        'raw_text': OCR 提取的完整文本（str),
        'paragraph': 合并后的段落文本 (str),
        'reactions': RxnExtractor 输出的反应列表 (list)
      }
    """
    try:
        raw_text, backend = _extract_image_text(image_path)
    except Exception as e:
        return {"error": f"OCR unavailable: {e}"}

    paragraph = _paragraph_from_raw_text(raw_text)
    sentences = split_text_into_sentences(paragraph)

    try:
        rxn_extractor = _get_rxn_extractor()
    except Exception as e:
        return {"error": f"RxnExtractor unavailable: {e}"}

    all_reactions = []
    try:
        reactions = rxn_extractor.get_reactions(sentences)
        all_reactions = reactions
    except AssertionError as e:
        print(f"Warning: batch reaction extraction failed, retrying sentence-by-sentence: {e}")
        all_reactions = []
        for sent in sentences:
            try:
                sent_reactions = rxn_extractor.get_reactions([sent])
                all_reactions.extend(sent_reactions)
            except Exception as sent_e:
                print(f"Warning: skipping sentence after extraction failure: {sent[:50]}... error: {sent_e}")
                continue

    return {
        "ocr_backend": backend,
        "raw_text": raw_text,
        "paragraph": paragraph,
        "sentences": sentences,
        "reactions": all_reactions,
    }

def NER_from_text_in_image(image_path: str) -> dict:
    try:
        raw_text, backend = _extract_image_text(image_path)
    except Exception as e:
        return {"error": f"OCR unavailable: {e}"}

    paragraph = _paragraph_from_raw_text(raw_text)

    try:
        model2 = _get_chemner_model()
    except Exception as e:
        return {"error": f"ChemNER unavailable: {e}"}

    predictions = model2.predict_strings([paragraph])

    return {
        "ocr_backend": backend,
        "raw_text": raw_text,
        "paragraph": paragraph,
        "entities": predictions,
    }




def text_extraction_agent(image_path: str) -> dict:
    """
    Agent that calls two tools:
      1) extract_reactions_from_text_in_image
      2) NER_from_text_in_image
    to perform OCR, reaction extraction, and chemical NER on a single image.
    Returns a merged JSON result.
    """
    client = _get_text_agent_client()
    model_name = _resolve_text_agent_model()

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`(“NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst B27" in text to the molecule B27 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates and oxaziridines was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes were used as precursors of azolium enolates. A bifunctional NHC precatalyst B27 bearing a free hydroxyl group was employed."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "NHC-bound azolium enolates" -> Chemical (Plural/Class)
    * "oxaziridines" -> Chemical (Systematic Name)
    * "Aryl(alkyl)-disubstituted ketenes" -> Chemical (Reactant precursor)
    * "NHC" -> Abbreviation
    * "B27" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "B27" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for B27 from the tool/image context: `CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F`.
    * "azolium enolates" and "oxaziridines" are participating reactants in a [3+2] cycloaddition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates [reactant][MULTIPLE] and oxaziridines [reactant][SYSTEMATIC] was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes [reactant] were used as precursors of azolium enolates. A bifunctional NHC [ABBREVIATION] precatalyst B27 [reagent][IDENTIFIERS][SMILES:CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F] bearing a free hydroxyl group was employed."
}
"""

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    response1 = _chat_completion_with_json_fallback(
        client,
        model=model_name,
        messages=messages,
        tools=tools,
        #        response_format={"type": "json_object"}
    )

    # Get assistant message with tool calls
    assistant_message = response1.choices[0].message
    
    # Execute each requested tool
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        # If no tool calls, return the response directly
        return _safe_parse_agent_json(client, model_name, response1.choices[0].message.content)
    
    tool_results_msgs = []
    for call in tool_calls:
        name = call.function.name
        tool_call_id = call.id
        
        if name == "extract_reactions_from_text_in_image":
            result = extract_reactions_from_text_in_image(image_path)
        elif name == "NER_from_text_in_image":
            result = NER_from_text_in_image(image_path)
        else:
            continue
        
        # Correct format for tool messages: need tool_call_id, not tool_name
        tool_results_msgs.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False)
        })

    # Second API call: pass tool outputs back to GPT for final response
    # Add assistant message and tool results to messages
    messages.append(_assistant_message_to_dict(assistant_message))
    messages.extend(tool_results_msgs)
    
    response2 = _chat_completion_with_json_fallback(
        client,
        model=model_name,
        messages=messages,
        #           response_format={"type": "json_object"}
    )

    return _safe_parse_agent_json(client, model_name, response2.choices[0].message.content)


def retry_api_call(func, max_retries=3, base_delay=2, backoff_factor=2, *args, **kwargs):
    """
    通用的 API 调用重试函数，支持指数退避策略。
    
    Args:
        func: 要调用的函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        backoff_factor: 退避因子（每次重试延迟时间 = base_delay * backoff_factor^attempt）
        *args, **kwargs: 传递给 func 的参数
    
    Returns:
        func 的返回值
    
    Raises:
        最后一次尝试的异常
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            error_message = str(e)
            
            # Retry transient provider errors across OpenAI-compatible and Anthropic clients.
            if error_code in {429, 500, 502, 503, 529} or 'overloaded' in error_message.lower() or '503' in error_message or 'rate limit' in error_message.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"⚠️ API 调用失败 (503/过载)，第 {attempt + 1}/{max_retries} 次尝试。{delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ API 调用失败，已达到最大重试次数 ({max_retries})")
                    raise
            else:
                # 其他类型的错误，直接抛出
                raise
        except BaseException:
            raise
    
    # 如果所有重试都失败了
    if last_exception:
        raise last_exception
    raise RuntimeError("API 调用失败，未知错误")


def text_extraction_agent_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`("NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst B27" in text to the molecule B27 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates and oxaziridines was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes were used as precursors of azolium enolates. A bifunctional NHC precatalyst B27 bearing a free hydroxyl group was employed."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "NHC-bound azolium enolates" -> Chemical (Plural/Class)
    * "oxaziridines" -> Chemical (Systematic Name)
    * "Aryl(alkyl)-disubstituted ketenes" -> Chemical (Reactant precursor)
    * "NHC" -> Abbreviation
    * "B27" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "B27" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for B27 from the tool/image context: `CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F`.
    * "azolium enolates" and "oxaziridines" are participating reactants in a [3+2] cycloaddition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates [reactant][MULTIPLE] and oxaziridines [reactant][SYSTEMATIC] was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes [reactant] were used as precursors of azolium enolates. A bifunctional NHC [ABBREVIATION] precatalyst B27 [reagent][IDENTIFIERS][SMILES:CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F] bearing a free hydroxyl group was employed."
}
```

"""

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    # Note: vLLM may not support response_format and tools simultaneously
    try:
        response1 = retry_api_call(
            client.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
                        # response_format={"type": "json_object"},  # vLLM 不支持同时使用 response_format 和 tools
        )
    except Exception as e:
        error_msg = str(e)
        if "tool" in error_msg.lower() or "tool-call" in error_msg.lower():
            print(f"⚠️ 警告: vLLM 不支持工具调用: {e}")
            print("提示: 请重新启动 vLLM 容器，添加以下参数:")
            print("  --enable-auto-tool-choice --tool-call-parser auto")
            print("或者继续使用 Ollama（原生支持工具调用）")
            raise
        else:
            raise

    # Get assistant message with tool calls
    assistant_message = response1.choices[0].message
    
    # Execute each requested tool
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        # If no tool calls, try to parse response directly
        raw_content = response1.choices[0].message.content
        if raw_content:
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                try:
                    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                    result = extract_json_from_text_with_reasoning(raw_content)
                    if result is not None:
                        return result
                except ImportError:
                    pass
                return {"content": raw_content}
        return {}
    
    tool_results_msgs = []
    for call in tool_calls:
        name = call.function.name
        tool_call_id = call.id
        
        if name == "extract_reactions_from_text_in_image":
            result = extract_reactions_from_text_in_image(image_path)
        elif name == "NER_from_text_in_image":
            result = NER_from_text_in_image(image_path)
        else:
            continue
        
        # Correct format for tool messages: need tool_call_id and name (for some APIs)
        tool_results_msgs.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,  # Some APIs (like Gemini) require name field
            "content": json.dumps(result, ensure_ascii=False)
        })

    # Second API call: pass tool outputs back to GPT for final response
    # Add assistant message and tool results to messages
    messages.append(_assistant_message_to_dict(assistant_message))
    messages.extend(tool_results_msgs)
    
    response2 = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
        messages=messages,
                # response_format={"type": "json_object"},  # vLLM 可能不支持
    )

    # Parse response (support extracting JSON from text with reasoning)
    raw_content = response2.choices[0].message.content
    
    try:
        # First try direct JSON parsing
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from text
        try:
            from get_R_group_sub_agent import extract_json_from_text_with_reasoning
            result = extract_json_from_text_with_reasoning(raw_content)
            if result is not None:
                return result
        except ImportError:
            pass
        
        # If all else fails, return raw content wrapped in dict
        print(f"⚠️ 警告: 无法解析 JSON，返回原始内容")
        print(f"Raw content (last 500 chars):\n{raw_content[-500:]}")
        return {"content": raw_content}
