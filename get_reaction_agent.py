import sys
import torch
import json
from chemietoolkit import ChemIEToolkit
import cv2
from PIL import Image
import json
import sys
import torch
from rxnim import RxnIM
import json
from molnextr.chemistry import _convert_graph_to_smiles

from openai import OpenAI
import base64
import numpy as np
from chemietoolkit import utils
from PIL import Image
import os
from typing import Optional
import time
from asset_registry import ensure_asset_available
from llm_wrapper import LLMWrapper
from runtime_guards import (
    RuntimeStageError,
    assistant_message as guarded_assistant_message,
    first_dict_item,
    first_item,
    message_content,
    safe_json_loads,
    tool_calls_or_empty,
)
from runtime_device import resolve_torch_device

def retry_api_call(func, max_retries=3, base_delay=2, backoff_factor=2, *args, **kwargs):
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

_RUNTIME_DEVICE_TYPE: Optional[str] = None
_RUNTIME_RXNIM: Optional[RxnIM] = None


def _get_rxnim() -> RxnIM:
    global _RUNTIME_DEVICE_TYPE, _RUNTIME_RXNIM

    device = resolve_torch_device()
    if _RUNTIME_RXNIM is None or _RUNTIME_DEVICE_TYPE != device.type:
        _RUNTIME_RXNIM = RxnIM(str(ensure_asset_available("rxn_ckpt")), device=device)
        _RUNTIME_DEVICE_TYPE = device.type
        print(f"[ChemEagle] Reaction agent runtime using {device.type.upper()}.")
    return _RUNTIME_RXNIM

def _get_azure_client():
    return LLMWrapper.from_env(default_model=os.getenv("LLM_MODEL") or "gpt-5-mini").as_chat_completion_client()


def _normalize_chat_message(message):
    """Convert SDK chat message objects into plain dicts for follow-up requests."""
    if isinstance(message, dict):
        return message

    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=True, by_alias=True)

    normalized = {"role": getattr(message, "role", "assistant")}

    content = getattr(message, "content", None)
    if content is not None:
        normalized["content"] = content

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        normalized["tool_calls"] = [
            tc.model_dump(mode="json", exclude_none=True, by_alias=True)
            if hasattr(tc, "model_dump") else tc
            for tc in tool_calls
        ]

    return normalized


def _empty_reaction_sections() -> dict:
    return {
        "reactants": [],
        "conditions": [],
        "products": [],
    }


def _first_prediction_or_empty(raw_prediction, *, context: str) -> dict:
    if isinstance(raw_prediction, dict):
        return raw_prediction
    raw_root = first_item(raw_prediction, context=context, default=None)
    if isinstance(raw_root, dict):
        return raw_root
    print(f"warning: {context}: empty result list")
    return _empty_reaction_sections()

def get_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    image = Image.open(image_file)

    image_file = image_path
    raw_prediction = _get_rxnim().predict_image_file(image_file, molnextr=True, ocr=True)
    raw_root = _first_prediction_or_empty(
        raw_prediction,
        context="reaction_agent.predict_image_file",
    )
    #print(f'raw_prediction:{raw_prediction}')

    # Ensure raw_prediction is treated as a list directly
    structured_output = _empty_reaction_sections()
    for section_key in ['reactants', 'conditions', 'products']:
        if section_key in raw_root:
            structured_output[section_key] = []
            for item in raw_root[section_key]:
                if section_key in ['reactants', 'products']:
                    # Extract smiles and bbox for molecules
                    structured_output[section_key].append({
                        "smiles": item.get("smiles", ""),
                        "bbox": item.get("bbox", []),
                        "symbols": item.get("symbols", [])  
                    })
                elif section_key == 'conditions':
                    # Extract smiles, text, and bbox for conditions
                    condition_data = {"bbox": item.get("bbox", [])}
                    if "smiles" in item:
                        condition_data["smiles"] = item.get("smiles", "")
                        condition_data["symbols"] = item.get("symbols", [])
                    if "text" in item:
                        condition_data["text"] = item.get("text", [])
                    structured_output[section_key].append(condition_data)
    #print(f'structured_output:{structured_output}')

    return structured_output



def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    raw_prediction = _get_rxnim().predict_image_file(image_file, molnextr=True, ocr=True)
    for reaction in raw_prediction:
        for section in ("reactants", "products", "conditions"):
            for entry in reaction.get(section, []):
                # 1) 保留 coords 三位小数
                coords = entry.get("coords")
                if isinstance(coords, list):
                    entry["coords"] = [
                        [round(val, 3) for val in point]
                        for point in coords
                    ]
                # 2) 删除不需要的字段
                for key in ("molfile", "atoms", "bonds"):
                    entry.pop(key, None)

    raw_prediction =json.dumps(raw_prediction)
    return raw_prediction



def get_reaction_withatoms(image_path: str) -> dict:
    """
    输入化学反应图像路径，通过 GPT 模型和 OpenChemIE 提取反应信息并返回整理后的反应数据。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
    # 初始化 OpenChemIE 模型和 Azure OpenAI 客户端
    client = _get_azure_client()

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT 工具调用配置
    tools = [
        {
        'type': 'function',
        'function': {
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
    ]

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_getreaction.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口
    response = client.chat.completions.create(
    model = os.getenv("LLM_MODEL", 'gpt-5-mini'),
    response_format={ 'type': 'json_object' },
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]},
    ],
    tools = tools)
    
# Step 1: 工具映射表
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: 处理多个工具调用
    assistant_message = guarded_assistant_message(
        response,
        context="reaction_agent.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_calls = tool_calls_or_empty(assistant_message, context="reaction_agent.tool_selection")
    if not tool_calls:
        raise RuntimeStageError(
            "assistant message has no tool calls",
            context="reaction_agent.tool_selection",
            retry_trigger="auto_recovery_retry",
        )
    results = []

    # 遍历每个工具调用
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # 调用工具并获取结果
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # 保存每个工具调用结果
        results.append({
            'role': 'tool',
            'name': tool_name,  # Gemini API 要求必须包含 name 字段
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })


# Prepare the chat completion payload
    completion_payload = {
        'model': os.getenv("LLM_MODEL", 'gpt-5-mini'),
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}'
                        }
                    }
                ]
            },
            _normalize_chat_message(assistant_message),
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
    )


    
    # 获取 GPT 生成的结果
    gpt_output = safe_json_loads(
        message_content(
            response,
            context="reaction_agent.final_response",
            required=True,
            retry_trigger="auto_recovery_retry",
        ),
        context="reaction_agent.final_response",
        retry_trigger="auto_recovery_retry",
    )
    #print(f"gpt_output1:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''
        image_file = image_path
        raw_prediction = _get_rxnim().predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'products']:
            for item in input1.get(key, []):
                bbox = tuple(item['bbox'])  # 使用 bbox 作为唯一标识
                symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'products']:
            for item in input2.get(key, []):
                bbox = tuple(item['bbox'])  # 获取 bbox 作为匹配键

                # 如果 bbox 存在于 input1 的映射中，则更新 symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # 更新 atoms 的 atom_symbol
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # 如果 coords 和 edges 存在，调用转换函数生成新的 smiles 和 molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        try:
                            new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        except Exception as exc:
                            print(f"warning: reaction graph conversion skipped for bbox {bbox}: {exc}")
                            continue
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [
        update_input_with_symbols(
            gpt_output,
            _first_prediction_or_empty(
                input2,
                context="reaction_agent.get_reaction_full",
            ),
            _convert_graph_to_smiles,
        )
    ]

    return updated_data

 


def get_reaction_withatoms_correctR(image_path: str) -> dict:
    """
    输入化学反应图像路径，通过 GPT 模型和 OpenChemIE 提取反应信息并返回整理后的反应数据。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
    # 配置 API Key 和 Azure Endpoint
    

    client = _get_azure_client()

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT 工具调用配置
    tools = [
        {
        'type': 'function', 
        'function': {
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
    ]

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_Rxn_Tem.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口
    response = client.chat.completions.create(
    model = os.getenv("LLM_MODEL", 'gpt-5-mini'),
    #    response_format={ 'type': 'json_object' },
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]},
    ],
    tools = tools)
    
# Step 1: 工具映射表
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: 处理多个工具调用
    assistant_message = guarded_assistant_message(
        response,
        context="reaction_agent_correctR.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_calls = tool_calls_or_empty(assistant_message, context="reaction_agent_correctR.tool_selection")
    if not tool_calls:
        raise RuntimeStageError(
            "assistant message has no tool calls",
            context="reaction_agent_correctR.tool_selection",
            retry_trigger="auto_recovery_retry",
        )
    results = []

    # 遍历每个工具调用
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # 调用工具并获取结果
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # 保存每个工具调用结果
        results.append({
            'role': 'tool',
            'name': tool_name,  # Gemini API 要求必须包含 name 字段
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })


# Prepare the chat completion payload
    completion_payload = {
        'model': os.getenv("LLM_MODEL", 'gpt-5-mini'),
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}'
                        }
                    }
                ]
            },
            _normalize_chat_message(assistant_message),
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
    )


    
    # 获取 GPT 生成的结果
    gpt_output = safe_json_loads(
        message_content(
            response,
            context="reaction_agent_correctR.final_response",
            required=True,
            retry_trigger="auto_recovery_retry",
        ),
        context="reaction_agent_correctR.final_response",
        retry_trigger="auto_recovery_retry",
    )
    print(f"gpt_output_rxn:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''

        image_file = image_path
        raw_prediction = _get_rxnim().predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'conditions', 'products']:
            for item in input1.get(key, []):
                # 只处理有 symbols 和 bbox 字段的项（conditions 可能只有 text 字段而没有 symbols）
                if 'symbols' in item and 'bbox' in item:
                    bbox = tuple(item['bbox'])  # 使用 bbox 作为唯一标识
                    symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'conditions', 'products']:
            for item in input2.get(key, []):
                if 'bbox' not in item:
                    continue
                bbox = tuple(item['bbox'])  # 获取 bbox 作为匹配键

                # 如果 bbox 存在于 input1 的映射中，则更新 symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # 更新 atoms 的 atom_symbol
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # 如果 coords 和 edges 存在，调用转换函数生成新的 smiles 和 molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        try:
                            new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        except Exception as exc:
                            print(f"warning: reaction graph conversion skipped for bbox {bbox}: {exc}")
                            continue
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [
        update_input_with_symbols(
            gpt_output,
            _first_prediction_or_empty(
                input2,
                context="reaction_agent_correctR_os.get_reaction_full",
            ),
            _convert_graph_to_smiles,
        )
    ]
    print(f"rxn_agent_output:{updated_data}")

    return updated_data


def get_reaction_withatoms_correctR_OS(
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

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT 工具调用配置
    tools = [
        {
            'type': 'function', 
            'function': {
                'name': 'get_reaction',
                'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {
                            'type': 'string',
                            'description': 'The path to the reaction image.',
                        },
                    },
                    'required': ['image_path'],
                    'additionalProperties': False,
                },
            },
        },
    ]

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_Rxn_Tem.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口（带重试机制）
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
                #response_format={'type': 'json_object'},  # vLLM 不支持同时使用 response_format 和 tools
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    # Step 1: 工具映射表
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: 处理多个工具调用
    assistant_message = guarded_assistant_message(
        response,
        context="reaction_agent_os.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_calls = tool_calls_or_empty(assistant_message, context="reaction_agent_os.tool_selection")
    if not tool_calls:
        raise RuntimeStageError(
            "assistant message has no tool calls",
            context="reaction_agent_os.tool_selection",
            retry_trigger="auto_recovery_retry",
        )
    results = []

    # 遍历每个工具调用
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # 调用工具并获取结果
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # 保存每个工具调用结果
        results.append({
            'role': 'tool',
            'name': tool_name,  # Gemini API 要求必须包含 name 字段
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })

    # Prepare the chat completion payload
    completion_payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}'
                        }
                    }
                ]
            },
            _normalize_chat_message(assistant_message),
            *results
            ],
    }

    # Generate new response（带重试机制）
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        #response_format={'type': 'json_object'},  # vLLM 可能不支持
            )

    # 获取 GPT 生成的结果（支持从包含思考过程的文本中提取）
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
    
    raw_content = message_content(
        response,
        context="reaction_agent_os.final_response",
        required=True,
        retry_trigger="auto_recovery_retry",
    )
    
    try:
        # 首先尝试直接解析
        gpt_output = json.loads(raw_content)
        print(f"DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        # 如果直接解析失败，使用智能提取函数
        print(f"WARNING [OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        
        if gpt_output is not None:
            print(f"DEBUG [OS]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                f"Could not parse JSON from model response. Content may not be valid JSON.",
                raw_content, 0
            )
    
    print(f"gpt_output_rxn:{gpt_output}")

    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''

        image_file = image_path
        raw_prediction = _get_rxnim().predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)

    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'conditions', 'products']:
            for item in input1.get(key, []):
                # 只处理有 symbols 和 bbox 字段的项（conditions 可能只有 text 字段而没有 symbols）
                if 'symbols' in item and 'bbox' in item:
                    bbox = tuple(item['bbox'])  # 使用 bbox 作为唯一标识
                    symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'conditions', 'products']:
            for item in input2.get(key, []):
                if 'bbox' not in item:
                    continue
                bbox = tuple(item['bbox'])  # 获取 bbox 作为匹配键

                # 如果 bbox 存在于 input1 的映射中，则更新 symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # 更新 atoms 的 atom_symbol
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # 如果 coords 和 edges 存在，调用转换函数生成新的 smiles 和 molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        try:
                            new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        except Exception as exc:
                            print(f"warning: reaction graph conversion skipped for bbox {bbox}: {exc}")
                            continue
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [
        update_input_with_symbols(
            gpt_output,
            _first_prediction_or_empty(
                input2,
                context="reaction_agent_os.get_reaction_full",
            ),
            _convert_graph_to_smiles,
        )
    ]
    print(f"rxn_agent_output:{updated_data}")

    return updated_data
