import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
from openai import OpenAI
import numpy as np
from PIL import Image
import json
import os
import sys
from rxnim import RxnIM
import json
import base64
import re
from typing import Optional
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR
from get_reaction_agent import get_reaction_withatoms_correctR
from get_R_group_sub_agent import process_reaction_image_with_table_R_group, process_reaction_image_with_product_variant_R_group,get_full_reaction_template_OS,get_full_reaction_template, get_multi_molecular_full, process_reaction_image_with_table_R_group_OS,process_reaction_image_with_product_variant_R_group_OS,get_full_reaction_OS,get_reaction_OS
from get_observer import action_observer_agent, plan_observer_agent,action_observer_agent_OS, plan_observer_agent_OS
from get_text_agent import text_extraction_agent, text_extraction_agent_OS
from llm_wrapper import LLMWrapper


def _resolve_device() -> torch.device:
    pref = os.getenv("CHEMEAGLE_DEVICE", "auto").strip().lower()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("CHEMEAGLE_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device('cpu')
    if pref in {"metal", "mps"}:
        if mps_available:
            return torch.device('mps')
        print("CHEMEAGLE_DEVICE=metal requested but MPS is unavailable. Falling back to CPU.")
        return torch.device('cpu')
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        if mps_available:
            return torch.device('mps')
    return torch.device('cpu')


device = _resolve_device()
model = ChemIEToolkit(device=device)
ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device=device)


def _normalize_tool_args(raw_args: Optional[dict], image_path: str) -> dict:
    if not isinstance(raw_args, dict):
        return {"image_path": image_path}
    normalized = dict(raw_args)
    placeholder_values = {"[img]", "<img>", "[image]", "<image>", "<<<IMAGE>>>", "IMAGE_PATH", "image.png","image_path"}
    if normalized.get("image_path") in placeholder_values or normalized.get("image_path") is None:
        normalized["image_path"] = image_path
    return normalized

TOOL_NAME_ALIASES = {
    "structure_based_r_group_substitution_agent": "process_reaction_image_with_product_variant_R_group",
    "structure_based_rgroup_substitution_agent": "process_reaction_image_with_product_variant_R_group",
    "structurebased_rgroupsubstitutionagent": "process_reaction_image_with_product_variant_R_group",
    "text_based_r_group_substitution_agent": "process_reaction_image_with_table_R_group",
    "text_based_rgroup_substitution_agent": "process_reaction_image_with_table_R_group",
    "textbased_rgroupsubstitutionagent": "process_reaction_image_with_table_R_group",
    "reaction_template_parsing_agent": "get_full_reaction_template",
    "reactiontemplateparsingagent": "get_full_reaction_template",
    "molecular_recognition_agent": "get_multi_molecular_full",
    "molecularrecognitionagent": "get_multi_molecular_full",
    "text_extraction_agent": "text_extraction_agent",
    "textextractionagent": "text_extraction_agent",
}


def _normalize_tool_name(raw_name: Optional[str]) -> str:
    if not isinstance(raw_name, str):
        return ""

    name = raw_name.strip().strip('"\'')
    canonical_name = re.sub(r"[\s-]+", "_", name.lower())
    collapsed_name = canonical_name.replace("_", "")

    if name in TOOL_NAME_ALIASES.values():
        return name
    if canonical_name in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[canonical_name]
    if collapsed_name in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[collapsed_name]
    return canonical_name


def _select_primary_tool(agent_list: list[str]) -> str:
    normalized_agents = {_normalize_tool_name(agent) for agent in agent_list}

    if "process_reaction_image_with_product_variant_R_group" in normalized_agents:
        return "process_reaction_image_with_product_variant_R_group"
    if "process_reaction_image_with_table_R_group" in normalized_agents:
        return "process_reaction_image_with_table_R_group"
    if "get_full_reaction_template" in normalized_agents:
        return "get_full_reaction_template"
    if "get_multi_molecular_full" in normalized_agents:
        return "get_multi_molecular_full"

    print("warning: no agents")
    return "get_full_reaction_template"


def ChemEagle(
    image_path: str,
    *,
    use_plan_observer: bool = True,
    use_action_observer: bool = True,
) -> dict:
    """
    """
    llm = LLMWrapper.from_env(default_model="gpt-5-mini")

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
            'name': 'process_reaction_image_with_product_variant_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the set of product variants, and the original molecular list.',
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
            {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_table_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the R-group table',
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
            {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction_template',
            'description': 'After you carefully check the image, if this is a reaction image that contains only a text-based table and does not involve any R-group replacement, or this is a reaction image does not contain any tables or sets of product variants, then just call this simplified tool.',
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
            {
        'type': 'function',
        'function': {
            'name': 'get_multi_molecular_full',
            'description': 'After you carefully check the image, if this is a single molecule image or a multiple molecules image, then need to call this molecular recognition tool.',
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
        {
        'type': 'function',
        'function': {
            'name': 'text_extraction_agent',
            'description': 'Extract the text from the image.',
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
    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    # Step 1: 调用 planner 获取 agent 列表
    planner_output = llm.chat_completion_text(
        model=os.getenv("LLM_MODEL", "gpt-5-mini"),
        messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert.After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': planner_user_message},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ]
    )

    # 解析 planner 返回的 agent 列表
    print(f"[D] Planner output: {planner_output}")
    
    # 提取 agent 名称（移除可能的括号、花括号等）
    # 移除 { } 和多余的空白
    planner_output = re.sub(r'[{}]', '', planner_output).strip()
    # 分割为 agent 列表
    agent_list = [agent.strip() for agent in planner_output.split(',') if agent.strip()]
    print(f"[D] Parsed agents: {agent_list}")
    
    
    selected_tool = _select_primary_tool(agent_list)
    agent_names_lower = [agent.lower() for agent in agent_list]
    
    print(f"[D] Selected tool: {selected_tool}")
    
    # Step 3: 工具映射表
    TOOL_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full,
        'text_extraction_agent': text_extraction_agent
    }
    
    # Step 4: 构建执行计划（支持 observer）
    # 检查是否有 text_extraction_agent
    has_text_extraction = "text extraction agent" in agent_names_lower or "text_extraction_agent" in agent_names_lower
    
    serialized_calls = [{
        "id": "tool_call_0",
        "name": selected_tool,
        "arguments": {"image_path": image_path}
    }]
    
    # 如果有 text_extraction_agent，添加第二个工具
    if has_text_extraction:
        serialized_calls.append({
            "id": "tool_call_1",
            "name": "text_extraction_agent",
            "arguments": {"image_path": image_path}
        })
        print(f"[D] Added text_extraction_agent as second tool")
    
    # Plan Observer: 审查和修改工具调用计划
    if use_plan_observer:
        reviewed_plan = plan_observer_agent(image_path, serialized_calls)
        if not isinstance(reviewed_plan, list) or not reviewed_plan:
            plan_to_execute = serialized_calls
        else:
            plan_to_execute = []
            for idx, item in enumerate(reviewed_plan):
                name = item.get("name") or item.get("tool_name")
                if not name:
                    continue
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id = item.get("id") or f"observer_call_{idx}"
                plan_to_execute.append({
                    "id": call_id,
                    "name": name,
                    "arguments": args,
                })
            if not plan_to_execute:
                plan_to_execute = serialized_calls
    else:
        plan_to_execute = serialized_calls

    print(f"[D] plan_to_execute:{plan_to_execute}")
    execution_logs = []
    results = []

    # Step 5: 执行工具调用
    for idx, plan_item in enumerate(plan_to_execute):
        raw_tool_name = plan_item.get("name") or plan_item.get("tool_name")
        tool_name = _normalize_tool_name(raw_tool_name)
        if not tool_name:
            print(f"warning: plan_item {idx} no name ，skip: {plan_item}")
            continue
        
        tool_call_id = plan_item.get("id") or f"observer_call_{idx}"
        tool_args = _normalize_tool_args(plan_item.get("arguments", {}), image_path)

        if tool_name in TOOL_MAP:
            tool_func = TOOL_MAP[tool_name]
            tool_result = tool_func(**tool_args)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")

        execution_logs.append({
            "id": tool_call_id,
            "name": tool_name,
            "arguments": tool_args,
            "result": tool_result,
        })

        # 保存每个工具调用结果
        # NOTE: We execute tools locally (without model-emitted tool_calls), so
        # these must not be sent with role="tool" to OpenAI chat completions.
        # Instead, inject structured tool outputs as assistant context.
        results.append({
            'role': 'assistant',
            'content': json.dumps({
                'tool_call_id': tool_call_id,
                'tool_name': tool_name,
                'image_path': image_path,
                'tool_result': tool_result,
            }),
        })

    # Action Observer: 检查执行结果，如果失败则重新执行
    if use_action_observer and action_observer_agent(image_path, execution_logs):
        return {
            "redo": True,
            "plan": plan_to_execute,
            "execution_logs": execution_logs,
        }

    # Prepare the chat completion payload
    # 构建 assistant 消息，包含 planner 的输出和工具调用信息
    executed_tools = [selected_tool]
    if has_text_extraction:
        executed_tools.append("text_extraction_agent")
    assistant_message = {
        "role": "assistant",
        "content": f"Selected agents: {', '.join(agent_list)}\nExecuted tools: {', '.join(executed_tools)}"
    }
    
    completion_payload = {
        'model': os.getenv("LLM_MODEL", "gpt-5-mini"),
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
            assistant_message,
            *results
            ],
    }

    # Generate new response
    response_text = llm.chat_completion_text(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
    )

    # 获取 GPT 生成的结果
    gpt_output = json.loads(response_text)
    print(gpt_output)
    return gpt_output

def ChemEagle_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:
    """
    Open source version of ChemEagle
    """
    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    def encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    # Step 1: 调用 planner 获取 agent 列表
    planner_response = client.chat.completions.create(
        model=model_name,
                messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert.After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': planner_user_message},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ]
    )
    
    # 解析 planner 返回的 agent 列表
    planner_output = planner_response.choices[0].message.content.strip()
    print(f"[OS_D] Planner output: {planner_output}")
    
    # 提取 agent 名称（移除可能的括号、花括号等）
    # 移除 { } 和多余的空白
    planner_output = re.sub(r'[{}]', '', planner_output).strip()
    # 分割为 agent 列表
    agent_list = [agent.strip() for agent in planner_output.split(',') if agent.strip()]
    print(f"[OS_D] Parsed agents: {agent_list}")
    
    selected_tool = _select_primary_tool(agent_list)
    agent_names_lower = [agent.lower() for agent in agent_list]
    
    print(f"[OS_D] Selected tool: {selected_tool}")
    
    TOOL_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group_OS,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group_OS,
        'get_full_reaction_template': get_full_reaction_template_OS,
        'get_multi_molecular_full': get_multi_molecular_full,
        'text_extraction_agent': text_extraction_agent_OS
    }
    
    # Step 3: 构建执行计划（支持 observer）
    # 检查是否有 text_extraction_agent
    has_text_extraction = "text extraction agent" in agent_names_lower or "text_extraction_agent" in agent_names_lower
    
    serialized_calls = [{
        "id": "tool_call_0",
        "name": selected_tool,
        "arguments": {"image_path": image_path}
    }]
    
    # 如果有 text_extraction_agent，添加第二个工具
    if has_text_extraction:
        serialized_calls.append({
            "id": "tool_call_1",
            "name": "text_extraction_agent",
            "arguments": {"image_path": image_path}
        })
        print(f"[OS_D] Added text_extraction_agent as second tool")
    
    # Plan Observer: 审查和修改工具调用计划
    if use_plan_observer:
        reviewed_plan = plan_observer_agent_OS(image_path, serialized_calls)
        if not isinstance(reviewed_plan, list) or not reviewed_plan:
            plan_to_execute = serialized_calls
        else:
            plan_to_execute = []
            for idx, item in enumerate(reviewed_plan):
                name = item.get("name") or item.get("tool_name")
                if not name:
                    continue
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id = item.get("id") or f"observer_call_{idx}"
                plan_to_execute.append({
                    "id": call_id,
                    "name": name,
                    "arguments": args,
                })
            if not plan_to_execute:
                plan_to_execute = serialized_calls
    else:
        plan_to_execute = serialized_calls

    print(f"[OS_D] plan_to_execute:{plan_to_execute}")
    execution_logs = []
    results = []

    # Step 4: 执行工具调用
    for idx, plan_item in enumerate(plan_to_execute):
        raw_tool_name = plan_item.get("name") or plan_item.get("tool_name")
        tool_name = _normalize_tool_name(raw_tool_name)
        if not tool_name:
            print(f"warning: plan_item {idx} no name ，skip: {plan_item}")
            continue
        
        tool_call_id = plan_item.get("id") or f"observer_call_{idx}"
        tool_args = _normalize_tool_args(plan_item.get("arguments", {}), image_path)

        if tool_name in TOOL_MAP:
            tool_func = TOOL_MAP[tool_name]
            tool_result = tool_func(**tool_args)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")

        execution_logs.append({
            "id": tool_call_id,
            "name": tool_name,
            "arguments": tool_args,
            "result": tool_result,
        })

        # 确保 tool_name 不为空（OpenAI 兼容 API 标准要求）
        if not tool_name or not tool_name.strip():
            print(f"warning: tool_name is empty，skip")
            continue
            
        results.append({
            'role': 'assistant',
            'content': json.dumps({
                'tool_call_id': tool_call_id,
                'tool_name': tool_name.strip(),
                'image_path': image_path,
                'tool_result': tool_result,
            }),
        })
    
    print(f'[OS_D] results: {results}')
    
    # Action Observer: 检查执行结果，如果失败则重新执行
    if use_action_observer and action_observer_agent_OS(image_path, execution_logs):
        return {
            "redo": True,
            "plan": plan_to_execute,
            "execution_logs": execution_logs,
        }

    # Prepare the chat completion payload
    # 构建 assistant 消息，包含 planner 的输出和工具调用信息
    executed_tools = [selected_tool]
    if has_text_extraction:
        executed_tools.append("text_extraction_agent")
    assistant_message = {
        "role": "assistant",
        "content": f"Selected agents: {', '.join(agent_list)}\nExecuted tools: {', '.join(executed_tools)}"
    }
    
    completion_payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}' }}
                ],
            },
            assistant_message,
            *results
            ],
    }

    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        #response_format={ 'type': 'json_object' },
            )
    print(response)
    
    # 获取原始响应内容
    raw_content = response.choices[0].message.content
    
    # 尝试解析 JSON（支持从包含思考过程的文本中提取）
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
    
    try:
        # 首先尝试直接解析
        gpt_output = json.loads(raw_content)
        print("DEBUG [OS_D]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        # 如果直接解析失败，使用智能提取函数（支持思考模型输出）
        print("WARNING [OS_D]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        
        if gpt_output is not None:
            print("DEBUG [OS_D]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS_D]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            # 如果无法解析为 JSON，返回原始内容（保持向后兼容）
            print("WARNING [OS_D]: Returning raw content as fallback")
            return {"content": raw_content, "parsed": False}
    
    print(gpt_output)
    return gpt_output



if __name__ == "__main__":
    model = ChemEagle()
