import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
import numpy as np
from PIL import Image
import json
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR, process_reaction_image_with_multiple_products_and_text_correctmultiR_OS
from get_reaction_agent import get_reaction_withatoms_correctR, get_reaction_withatoms_correctR_OS
import sys
from rxnim import RxnIM
import json
import base64
import os
import base64
import torch
import json
from PIL import Image
import numpy as np
from openai import OpenAI
from typing import Optional
import copy
from molnextr.chemistry import _convert_graph_to_smiles 
import os
import io
import re
import time
from asset_registry import ensure_asset_available
from llm_wrapper import LLMWrapper
from runtime_guards import (
    RuntimeStageError,
    assistant_message as guarded_assistant_message,
    first_dict_item,
    first_tool_call,
    message_content,
    safe_json_loads,
    tool_calls_or_empty,
)
from runtime_device import resolve_torch_device

_RUNTIME_DEVICE_TYPE: Optional[str] = None
_RUNTIME_TOOLKIT: Optional[ChemIEToolkit] = None
_RUNTIME_RXNIM: Optional[RxnIM] = None


def _debug_enabled() -> bool:
    return os.getenv("CHEMEAGLE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_print(message: str) -> None:
    if _debug_enabled():
        print(message)


def _get_runtime_models() -> tuple[ChemIEToolkit, RxnIM]:
    global _RUNTIME_DEVICE_TYPE, _RUNTIME_TOOLKIT, _RUNTIME_RXNIM

    device = resolve_torch_device()
    if (
        _RUNTIME_TOOLKIT is None
        or _RUNTIME_RXNIM is None
        or _RUNTIME_DEVICE_TYPE != device.type
    ):
        _RUNTIME_TOOLKIT = ChemIEToolkit(device=device)
        _RUNTIME_RXNIM = RxnIM(str(ensure_asset_available("rxn_ckpt")), device=device)
        _RUNTIME_DEVICE_TYPE = device.type
        _debug_print(f"[ChemEagle] R-group agent runtime using {device.type.upper()}.")
    return _RUNTIME_TOOLKIT, _RUNTIME_RXNIM


def _normalize_chat_message(message) -> dict:
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



def _get_azure_client():
    return LLMWrapper.from_env(default_model=os.getenv("LLM_MODEL") or "gpt-5-mini").as_chat_completion_client()


def _extract_valid_coref_entries(bboxes, group):
    if not isinstance(group, (list, tuple)):
        return []
    valid_entries = []
    for raw_idx in group:
        if isinstance(raw_idx, int) and 0 <= raw_idx < len(bboxes):
            valid_entries.append((raw_idx, bboxes[raw_idx]))
    return valid_entries

def normalize_product_variant_output(data: dict) -> dict:
   
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary")
    
    normalized_reactions = []
    original_molecule_list = data.get('original_molecule_list', {})
    
    # 1. 处理 reaction_template（第一个 reaction，reaction_id = '0_1'）
    if 'reaction_template' in data:
        template = data['reaction_template']
        template_reactants = template.get('reactants', [])
        template_products = template.get('products', [])
        
        # 尝试从 original_molecule_list 中找到 product template 的 label
        template_product_labels = []
        if template_products:
            # 为每个 product 查找对应的 label
            for template_product_smiles in template_products:
                template_product_label = None
                # 在 original_molecule_list 中查找匹配的 SMILES，寻找 template 相关的条目
                if template_product_smiles in original_molecule_list:
                    info = original_molecule_list[template_product_smiles]
                    if isinstance(info, list) and len(info) > 0:
                        # 检查是否是 template 相关（info 中可能包含 'reactant template' 或 'product template'）
                        info_str = ' '.join(str(item).lower() for item in info)
                        if 'template' in info_str:
                            # 从 info[0] 获取 label
                            template_product_label = str(info[0]) if info[0] else None
                template_product_labels.append(template_product_label)
        
        normalized_reactions.append({
            'reaction_id': '0_1',
            'note': 'reaction template',
            'reactants': [{'smiles': smiles} for smiles in template_reactants],
            'conditions': [],
            'products': [{'smiles': smiles, 'label': label} if label else {'smiles': smiles}
                        for smiles, label in zip(template_products, template_product_labels)]
        })
    
    # 2. 处理 reactions 字典（从 '1_1' 开始编号）
    if 'reactions' in data:
        reactions_dict = data['reactions']
        # 按 key 排序以确保顺序一致
        sorted_reaction_keys = sorted(reactions_dict.keys())
        
        for idx, reaction_key in enumerate(sorted_reaction_keys, start=1):
            reaction_data = reactions_dict[reaction_key]
            reaction_reactants = reaction_data.get('reactants', [])
            reaction_products = reaction_data.get('products', [])
            
            # 提取 conditions：从 original_molecule_list 中通过 product SMILES 和 label 匹配
            conditions = []
            for product_smiles in reaction_products:
                # 在 original_molecule_list 中查找匹配的条目
                if product_smiles in original_molecule_list:
                    info = original_molecule_list[product_smiles]
                    if isinstance(info, list) and len(info) > 0:
                        # info 格式: ['20', '8 h, 87% yield', 'product', 'bbox_id=14', 'id=15']
                        # 需要找到 label 匹配的条目（info[0] 是 label）
                        info_label = str(info[0]) if info[0] else None
                        if info_label == str(reaction_key):
                            # 提取条件信息（跳过 label(索引0), 'product'/'reactant'/'template', bbox_id=, id=）
                            for item in info[1:]:
                                if item and isinstance(item, str):
                                    # 跳过固定关键词和 bbox_id=、id= 开头的项
                                    if item not in ['product', 'reactant', 'template'] and not item.startswith('bbox_id=') and not item.startswith('id='):
                                        if item not in conditions:  # 避免重复
                                            conditions.append(item)
            
            # 构建标准化格式
            normalized_reaction = {
                'reaction_id': f'{idx}_1',
                'reactants': [{'smiles': smiles} for smiles in reaction_reactants],
                'conditions': conditions if conditions else [],
                'products': [{'smiles': smiles, 'label': reaction_key} for smiles in reaction_products]
            }
            normalized_reactions.append(normalized_reaction)
    
    return {
        'reactions': normalized_reactions,
        'original_molecule_list': original_molecule_list
    }


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


def draw_mol_bboxes(image_path, coref_results, output_path=None):
    """
    Draws bounding boxes on the original image for each molecule in coref_results.
    
    Args:
        image_path (str): Path to the original image.
        coref_results (list): The coreference results data structure.
        output_path (str, optional): Path to save the annotated image. If None, returns the image array.
    
    Returns:
        np.ndarray: The annotated image if output_path is None, else None (saves file).
    """
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    height, width = img.shape[:2]

    # Handle the structure of coref_results
    # It is a list of dicts. We assume the first dict corresponds to the image if it's a list.
    if isinstance(coref_results, list) and len(coref_results) > 0:
        data = first_dict_item(coref_results, context="r_group_agent.draw_mol_bboxes", retry_trigger="auto_recovery_retry")
    elif isinstance(coref_results, dict):
        data = coref_results
    else:
        print("Error: Invalid coref_results format")
        return None

    bboxes = data.get('bboxes', [])
    
    # Iterate through all bounding boxes
    for item in bboxes:
        # Check if category is [Mol]
        if item.get('category') == '[Mol]':
            bbox = item.get('bbox')
            bbox_id = item.get('bbox_id')
            
            if bbox:
                # Bbox is [x1, y1, x2, y2] normalized
                # Convert to pixel coordinates
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # Draw rectangle (Green, thickness 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw bbox_id
                if bbox_id is not None:
                    label = str(bbox_id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    # Calculate text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Ensure text is within image bounds
                    text_x = x1
                    text_y = y1 - 5
                    
                    # If box is at the top, draw text inside or below
                    if text_y < text_height:
                        text_y = y1 + text_height + 5
                        
                    # Draw background rectangle for text
                    # cv2.rectangle(img, (text_x, text_y - text_height - 2), (text_x + text_width, text_y + baseline), (0, 255, 0), -1)
                    # Use a small filled box for the ID
                    cv2.rectangle(img, (x1, y1), (x1 + text_width + 4, y1 + text_height + 10), (255, 255, 255), -1)
                    
                    # Draw text (Black)
                    cv2.putText(img, label, (x1 + 2, y1 + text_height + 5), font, font_scale, (0, 0, 0), thickness)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")
        return None
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_json_from_text_with_reasoning(text):
    """
    从包含思考过程的文本中提取 JSON 对象。
    支持处理 thinking model 的输出，其中可能包含大量思考过程，JSON 在最后。
    
    Args:
        text: 包含 JSON 的文本，可能包含思考过程
        
    Returns:
        dict: 解析后的 JSON 对象，如果失败返回 None
    """
    # 方法1: 尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 方法2: 查找 </think> 或类似标记后的 JSON
    # 处理思考模型可能使用的标记
    markers = [
        r'</think>',
        r'</thinking>',
        r'</reasoning>',
        r'</think>',
        r'```json',
        r'```',
    ]
    
    for marker in markers:
        pattern = f'{marker}\\s*(.*?)(?:```|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                continue
    
    # 方法3: 从文本末尾开始查找完整的 JSON 对象
    # 找到最后一个 { 的位置，然后尝试匹配完整的 JSON
    last_brace_start = text.rfind('{')
    if last_brace_start != -1:
        # 从最后一个 { 开始，尝试找到匹配的 }
        brace_count = 0
        json_end = -1
        for i in range(last_brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end != -1:
            json_content = text[last_brace_start:json_end]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
    
    # 方法4: 查找第一个 { 到最后一个 } 之间的内容（简单方法）
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_content = text[first_brace:last_brace + 1]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass
    
    # 方法5: 查找包含 "reactions" 键的 JSON（针对特定格式）
    reactions_pattern = r'\{[^{}]*"reactions"[^{}]*\[.*?\].*?\}'
    match = re.search(reactions_pattern, text, re.DOTALL)
    if match:
        # 扩展匹配范围，找到完整的 JSON 对象
        start = match.start()
        # 向前找到 {，向后找到匹配的 }
        brace_count = 0
        json_start = start
        json_end = -1
        
        # 向前查找开始的 {
        for i in range(start, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    json_start = i
                    break
        
        if json_start != -1:
            brace_count = 0
            for i in range(json_start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end != -1:
                json_content = text[json_start:json_end]
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
    
    return None


def parse_coref_data_with_fallback(data):
    bboxes = data.get("bboxes", [])
    corefs = data.get("corefs", [])
    paired_indices = set()

    # 先处理有 coref 配对的
    results = []
    for group in corefs:
        valid_entries = _extract_valid_coref_entries(bboxes, group)
        if not valid_entries:
            continue

        smiles_entry = next((entry for _, entry in valid_entries if "smiles" in entry), None)
        text_entry = next((entry for _, entry in valid_entries if "text" in entry), None)
        if smiles_entry is None:
            continue

        smiles = smiles_entry.get("smiles", "")
        bbox_id = smiles_entry.get("bbox_id", "")
        
        # 如果 smiles_entry 有 sub_text，直接使用 sub_text；否则使用 text_entry 的 text
        if "sub_text" in smiles_entry:
            result_item = {
                "smiles": smiles,
                "text": smiles_entry["sub_text"],
                "bbox_id": bbox_id
            }
        else:
            texts = text_entry.get("text", []) if text_entry else []
            result_item = {
                "smiles": smiles,
                "texts": texts or ["There is no label or failed to detect, please recheck the image again"],
                "bbox_id": bbox_id
            }
        
        results.append(result_item)

        paired_indices.update(idx for idx, _ in valid_entries)

    # 处理未配对的 SMILES（补充进来）
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            # 如果 entry 有 sub_text，直接使用 sub_text；否则使用默认提示文本
            if "sub_text" in entry:
                result_item = {
                    "smiles": entry["smiles"],
                    "text": entry["sub_text"],
                    "bbox_id": entry.get("bbox_id", ""),
                }
            else:
                result_item = {
                    "smiles": entry["smiles"],
                    "texts": ["There is no label or failed to detect, please recheck the image again"],
                    "bbox_id": entry.get("bbox_id", ""),
                }
            results.append(result_item)

    return results

def parse_coref_data_with_fallback_with_box(data):
    bboxes = data.get("bboxes", [])
    corefs = data.get("corefs", [])
    paired_indices = set()

    # 先处理有 coref 配对的
    results = []
    for group in corefs:
        valid_entries = _extract_valid_coref_entries(bboxes, group)
        if not valid_entries:
            continue

        smiles_entry = next((entry for _, entry in valid_entries if "smiles" in entry), None)
        text_entry = next((entry for _, entry in valid_entries if "text" in entry), None)
        if smiles_entry is None:
            continue

        smiles = smiles_entry.get("smiles", "")
        bbox_coords = smiles_entry.get("bbox", [])
        texts = text_entry.get("text", []) if text_entry else []

        results.append({
            "smiles": smiles,
            "texts": texts or ["There is no label or failed to detect, please recheck the image again"],
            "bbox": bbox_coords
        })

        paired_indices.update(idx for idx, _ in valid_entries)

    # 处理未配对的 SMILES（补充进来）
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            results.append({
                "smiles": entry["smiles"],
                "texts": ["There is no label or failed to detect, please recheck the image again"],
                "bbox": entry["bbox"],
            })

    return results





############################### MOl
_process_multi_molecular_cache = {}

def get_cached_multi_molecular(image_path: str):
    """
    只会对同一个 image_path 真正调用一次
    process_reaction_image_with_multiple_products_and_text_correctR
    并缓存结果。
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    if image_path not in _process_multi_molecular_cache:
        ##print(f"[get_cached_multi_molecular] Processing image: {image_path}")
        _process_multi_molecular_cache[image_path] = (
            process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
            ################################model.extract_molecule_corefs_from_figures([image])#############################################################################################
            )
        ##print(f"original output: {model.extract_molecule_corefs_from_figures([image])}")
    return _process_multi_molecular_cache[image_path]


def get_multi_molecular_text_to_correct(image_path: str) -> list:
    """
    GPT-4o 注册的 tool。内部不再直接调用二级 Agent，
    而是复用缓存过的结果。
    """
    coref_results = copy.deepcopy(get_cached_multi_molecular(image_path))

    # 按需删掉不想返回给 LLM 的字段
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in [
                "category", "molfile", "symbols",
                "atoms", "bonds", "category_id", "score", "corefs",
                "coords", "edges"
            ]:
                bbox.pop(key, None)

    # 假设 parse_coref_data_with_fallback 需要传入单个 dict
    parsed = parse_coref_data_with_fallback(
        first_dict_item(coref_results, context="r_group_agent.get_multi_molecular_text_to_correct", retry_trigger="auto_recovery_retry")
    )
    print(f"[get_multi_molecular_text_to_correct] parsed: {json.dumps(parsed)}")
    return parsed

############################### MOl_OS
_process_multi_molecular_cache = {}

def get_cached_multi_molecular_OS(image_path: str):
    """
    只会对同一个 image_path 真正调用一次
    process_reaction_image_with_multiple_products_and_text_correctR
    并缓存结果。
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    if image_path not in _process_multi_molecular_cache:
        ##print(f"[get_cached_multi_molecular] Processing image: {image_path}")
        _process_multi_molecular_cache[image_path] = (
            process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(image_path)
            #######model.extract_molecule_corefs_from_figures([image])
            )
        ##print(f"original output: {model.extract_molecule_corefs_from_figures([image])}")
    return _process_multi_molecular_cache[image_path]


def get_multi_molecular_text_to_correct_OS(image_path: str) -> list:
    """
    GPT-4o 注册的 tool。内部不再直接调用二级 Agent，
    而是复用缓存过的结果。
    """
    coref_results = copy.deepcopy(get_cached_multi_molecular_OS(image_path))

    # 按需删掉不想返回给 LLM 的字段
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in [
                "category", "molfile", "symbols",
                "atoms", "bonds", "category_id", "score", "corefs",
                "coords", "edges"
            ]:
                bbox.pop(key, None)

    # 假设 parse_coref_data_with_fallback 需要传入单个 dict
    parsed = parse_coref_data_with_fallback(
        first_dict_item(coref_results, context="r_group_agent.get_multi_molecular_text_to_correct_os", retry_trigger="auto_recovery_retry")
    )
    print(f"[get_multi_molecular_text_to_correct] parsed: {json.dumps(parsed)}")
    return parsed



def get_multi_molecular_full(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
    toolkit, _ = _get_runtime_models()
    coref_results = toolkit.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = first_dict_item(coref_results, context="r_group_agent.get_multi_molecular_full", retry_trigger="auto_recovery_retry")
    parsed = parse_coref_data_with_fallback(data)

    
    ##print(f"coref_results:{json.dumps(parsed)}")
    #return json.dumps(parsed)
    return parsed
#get_multi_molecular_text_to_correct('./acs.joc.2c00176 example 1.png')



############################### Rxn
_raw_results_cache = {}

def get_cached_raw_results(image_path: str):
    """
    调用一次 get_reaction_withatoms_correctR 并缓存结果，
    后续复用同一份 raw_results。
    """
    if image_path not in _raw_results_cache:
        #print(f"[get_cached_raw_results] Processing image: {image_path}")
        _raw_results_cache[image_path] = get_reaction_withatoms_correctR(image_path)
        ###############################_raw_results_cache[image_path]= model1.predict_image_file(image_path, molnextr=True, ocr=True)####################################################################
    return _raw_results_cache[image_path]


# ----------------------------------------
# 工具函数：基于 raw_pred 构造精简输出
# ----------------------------------------
def get_reaction_from_raw(raw_pred: dict) -> dict:
    """
    Returns a structured dictionary of reactions extracted from the raw prediction,
    """
    structured = {}
    for section in ['reactants', 'conditions', 'products']:
        if section in raw_pred:
            structured[section] = []
            for item in raw_pred[section]:
                if section in ('reactants', 'products'):
                    structured[section].append({
                        "smiles": item.get("smiles", ""),
                        "bbox":   item.get("bbox",   [])
                    })
                else:  # conditions
                    structured[section].append({
                        "text":   item.get("text",   []),
                        "bbox":   item.get("bbox",   []),
                        "smiles": item.get("smiles", [])
                    })
    return structured

# ----------------------------------------
# LLM 工具：get_reaction
# ----------------------------------------
def get_reaction(image_path: str) -> dict:
    """    
    Returns a structured dictionary of reactions extracted from the image,
    """
    # 复用缓存的 raw_results
    raw_pred = first_dict_item(
        get_cached_raw_results(image_path),
        context="r_group_agent.get_reaction",
        retry_trigger="auto_recovery_retry",
    )
    return get_reaction_from_raw(raw_pred)

############################### Rxn_OS

def get_cached_raw_results_OS(image_path: str):
    """
    调用一次 get_reaction_withatoms_correctR 并缓存结果，
    后续复用同一份 raw_results。
    """
    if image_path not in _raw_results_cache:
        #print(f"[get_cached_raw_results] Processing image: {image_path}")
        _raw_results_cache[image_path]= get_reaction_withatoms_correctR_OS(image_path)
        ######_raw_results_cache[image_path]= model1.predict_image_file(image_path, molnextr=True, ocr=True)####################################################################
    return _raw_results_cache[image_path]



def get_reaction_OS(image_path: str) -> dict:
    """    
    Returns a structured dictionary of reactions extracted from the image,
    """
    # 复用缓存的 raw_results
    raw_pred = first_dict_item(
        get_cached_raw_results_OS(image_path),
        context="r_group_agent.get_reaction_os",
        retry_trigger="auto_recovery_retry",
    )
    return get_reaction_from_raw(raw_pred)




def get_reaction_full(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image, 
    including only reactants, conditions, and products with their smiles, bbox, or text.
    '''
    image_file = image_path
    _, rxnim_model = _get_runtime_models()
    raw_prediction = rxnim_model.predict_image_file(image_file, molnextr=True, ocr=True)
    #raw_prediction = get_reaction_withatoms_correctR(image_path)
    return raw_prediction

def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    #raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    # 使用原始数据，包含 coords 和 edges 等完整信息
    raw_prediction = get_cached_raw_results(image_path)
    # raw_prediction 是一个列表，每个元素是一个反应字典
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

    #raw_prediction =json.dumps(raw_prediction)
    _debug_print(f"raw_prediction:{raw_prediction}")

    # coref_results = model.extract_molecule_corefs_from_figures([image])
    # for item in coref_results:
    #     for bbox in item.get("bboxes", []):
    #         for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
    #             bbox.pop(key, None)  # 安全地移除键

    # data = coref_results[0]
    # parsed = parse_coref_data_with_fallback(data)
    
    parsed = get_multi_molecular_text_to_correct(image_path)

    combined_result = {
        "reaction_prediction": raw_prediction,  # 是个list
        "molecule_coref": parsed               # 结构化分子识别结果
    }
    _debug_print(f"combined_result:{combined_result}")
    return combined_result

def get_full_reaction_OS(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    #raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    # 使用原始数据，包含 coords 和 edges 等完整信息
    raw_prediction = get_cached_raw_results_OS(image_path)
    # raw_prediction 是一个列表，每个元素是一个反应字典
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

    #raw_prediction =json.dumps(raw_prediction)
    _debug_print(f"raw_prediction:{raw_prediction}")

    # coref_results = model.extract_molecule_corefs_from_figures([image])
    # for item in coref_results:
    #     for bbox in item.get("bboxes", []):
    #         for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
    #             bbox.pop(key, None)  # 安全地移除键

    # data = coref_results[0]
    # parsed = parse_coref_data_with_fallback(data)
    
    parsed = get_multi_molecular_text_to_correct_OS(image_path)

    combined_result = {
        "reaction_prediction": raw_prediction,  # 是个list
        "molecule_coref": parsed               # 结构化分子识别结果
    }
    _debug_print(f"combined_result:{combined_result}")
    return combined_result






def get_full_reaction_template(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    _, rxnim_model = _get_runtime_models()
    raw_prediction = rxnim_model.predict_image_file(image_file, molnextr=True, ocr=True)
    ####################raw_prediction = get_reaction_withatoms_correctR(image_path)###############################################################################################
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

    #raw_prediction =json.dumps(raw_prediction)
    _debug_print(f"raw_prediction:{raw_prediction}")
    #coref_results = model.extract_molecule_corefs_from_figures([image])
    coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = first_dict_item(coref_results, context="r_group_agent.get_full_reaction_template", retry_trigger="auto_recovery_retry")
    parsed = parse_coref_data_with_fallback(data)

    combined_result = {
        #"reaction_prediction": raw_prediction,  # 是个list
        "molecule_coref": parsed               # 结构化分子识别结果
    }
    _debug_print(f"combined_result:{combined_result}")
    return combined_result

def get_full_reaction_template_OS(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    _, rxnim_model = _get_runtime_models()
    raw_prediction = rxnim_model.predict_image_file(image_file, molnextr=True, ocr=True)
    ####################raw_prediction = get_reaction_withatoms_correctR(image_path)###############################################################################################
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

    #raw_prediction =json.dumps(raw_prediction)
    _debug_print(f"raw_prediction:{raw_prediction}")
    #coref_results = model.extract_molecule_corefs_from_figures([image])
    coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(image_path)
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = first_dict_item(coref_results, context="r_group_agent.get_full_reaction_template_os", retry_trigger="auto_recovery_retry")
    parsed = parse_coref_data_with_fallback(data)

    combined_result = {
        #"reaction_prediction": raw_prediction,  # 是个list
        "molecule_coref": parsed               # 结构化分子识别结果
    }
    _debug_print(f"combined_result:{combined_result}")
    return combined_result



def process_reaction_image_with_product_variant_R_group(image_path: str) -> dict:
    """
    输入化学反应图像路径，通过 GPT 模型和 OpenChemIE 提取反应信息并返回整理后的反应数据。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
 
    client = _get_azure_client()

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_image_from_array(img_array: np.ndarray) -> str:
        """
        将 numpy array (RGB格式) 转换为 base64 字符串
        """
        # 确保是 uint8 类型
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        
        # 使用 PIL 转换为 PNG 字节流
        img_pil = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        # 编码为 base64
        return base64.b64encode(img_bytes).decode('utf-8')
    base64_image = encode_image(image_path)

    # GPT 工具调用配置
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_multi_molecular_text_to_correct',
                'description': 'Extracts the SMILES string and text coref from molecular images.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {
                            'type': 'string',
                            'description': 'Path to the reaction image.'
                        }
                    },
                    'required': ['image_path'],
                    'additionalProperties': False
                }
            }
        },
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
    with open('./prompt/prompt_Str_R.txt', 'r', encoding='utf-8') as prompt_file:
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
        'get_multi_molecular_text_to_correct': get_multi_molecular_text_to_correct,
        'get_reaction': get_reaction
    }

    # Step 2: 处理多个工具调用
    assistant_message = guarded_assistant_message(
        response,
        context="r_group_agent.product_variant.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_calls = tool_calls_or_empty(assistant_message, context="r_group_agent.product_variant.tool_selection")
    if not tool_calls:
        raise RuntimeStageError(
            "assistant message has no tool calls",
            context="r_group_agent.product_variant.tool_selection",
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
    #print(f"tool_results:{tool_result}")

    coref_results = get_cached_multi_molecular(image_path)
    annotated_img = draw_mol_bboxes(image_path, coref_results, output_path=None)
    base64_image_1 = encode_image_from_array(annotated_img)
    
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
                            'url': f'data:image/png;base64,{base64_image_1}'
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
            context="r_group_agent.product_variant.final_response",
            required=True,
            retry_trigger="auto_recovery_retry",
        ),
        context="r_group_agent.product_variant.final_response",
        retry_trigger="auto_recovery_retry",
    )
    print("R_group_agent_output:", gpt_output)
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

 
    #coref_results = model.extract_molecule_corefs_from_figures([image_np])
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
    #coref_results = get_cached_multi_molecular(image_path)


    # reaction_results = model.extract_reactions_from_figures([image_np])
    #reaction_results = get_reaction_withatoms_correctR(image_path)[0]
    raw_results  = get_cached_raw_results(image_path)
    reaction_results = first_dict_item(raw_results, context="r_group_agent.product_variant.raw_results", retry_trigger="auto_recovery_retry")
    
    reaction = {
    "reactants": reaction_results.get('reactants', []),
    "conditions": reaction_results.get('conditions', []),
    "products": reaction_results.get('products', [])
    }
    reaction_results = [{"reactions": [reaction]}]
    #print(reaction_results)
    

    # 定义更新工具输出的函数
    def extract_smiles_details(smiles_data, raw_details):
        smiles_details = {}
        for smiles in smiles_data:
            for detail in raw_details:
                for bbox in detail.get('bboxes', []):
                    if bbox.get('smiles') == smiles:
                        smiles_details[smiles] = {
                            'category': bbox.get('category'),
                            'bbox': bbox.get('bbox'),
                            'category_id': bbox.get('category_id'),
                            'score': bbox.get('score'),
                            'molfile': bbox.get('molfile'),
                            'atoms': bbox.get('atoms'),
                            'bonds': bbox.get('bonds'),
                        }
                        break
        return smiles_details

# 获取结果
    smiles_details = extract_smiles_details(gpt_output, coref_results)
    #print('smiles_details:', smiles_details)

    reactants_array = []
    products = []

    first_reaction = first_dict_item(reaction_results.get('reactions', []), context="r_group_agent.product_variant.reactions", retry_trigger="auto_recovery_retry")
    for reactant in first_reaction.get('reactants', []):
        if 'smiles' in reactant:
            #print(f"SMILES:{reactant['smiles']}")
            ##print(reactant)
            reactants_array.append(reactant['smiles'])

    for product in first_reaction.get('products', []):
        ##print(product['smiles'])
        ##print(product)
        products.append(product['smiles'])
    # 输出结果
    #import p#print
    #p#print.p#print(smiles_details)

        # 整理反应数据
    toolkit, _ = _get_runtime_models()
    backed_out = utils.backout_without_coref(reaction_results, coref_results, gpt_output, smiles_details, toolkit.molnextr)
    backed_out.sort(key=lambda x: x[2])
    extracted_rxns = {}
    for reactants, products_, label in backed_out:
        extracted_rxns[label] = {'reactants': reactants, 'products': products_}
    
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = first_dict_item(coref_results, context="r_group_agent.process_variant.parse_coref", retry_trigger="auto_recovery_retry")
    parsed = parse_coref_data_with_fallback(data)
    
    toadd = {
        "reaction_template": {
            "reactants": reactants_array,
            "products": products
        },
        "reactions": extracted_rxns,
        "original_molecule_list": gpt_output
    }

# 按标签排序
    sorted_keys = sorted(toadd["reactions"].keys())
    toadd["reactions"] = {i: toadd["reactions"][i] for i in sorted_keys}
    toadd = normalize_product_variant_output(toadd)
    print(f"str_R_group_agent_output:{toadd}")
    return toadd


def process_reaction_image_with_product_variant_R_group_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
    # model_name="gemini-2.5-flash",
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
    # api_key="AIzaSyBL8j4MbHAPhq8cR4Y05o9tY5Zq6fMDU3g"  
) -> dict:
    """
    与 process_reaction_image_with_product_variant_R_group 流程保持一致，但改用兼容 OpenAI Chat Completions 协议的本地/自建模型（如 vLLM 或 Ollama）。

    Args:
        image_path: 反应图像路径。
        model_name: 本地模型名称（默认 `Qwen/Qwen3-VL-8B-Instruct`）。
        base_url: OpenAI 兼容接口地址，若为 None 则使用 `http://localhost:8000/v1` (vLLM 默认端口)。
        api_key: 接口密钥，可为任意非空字符串（vLLM 默认可填 `"EMPTY"`）。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
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
                'name': 'get_multi_molecular_text_to_correct_OS',
                'description': 'Extracts the SMILES string and text coref from molecular images.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {
                            'type': 'string',
                            'description': 'Path to the reaction image.'
                        }
                    },
                    'required': ['image_path'],
                    'additionalProperties': False
                }
            }
        },
        {
        'type': 'function',
        'function': {
            'name': 'get_reaction_OS',
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
    with open('./prompt/prompt_Str_R.txt', 'r', encoding='utf-8') as prompt_file:
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
        max_retries=5,  # 增加重试次数，因为可能同时有多个请求
        base_delay=3,   # 增加基础延迟，给 API 更多恢复时间
        backoff_factor=2,
        model=model_name,
                #response_format={'type': 'json_object'},
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    # Step 1: 工具映射表
    TOOL_MAP = {
        'get_multi_molecular_text_to_correct_OS': get_multi_molecular_text_to_correct_OS,
        'get_reaction_OS': get_reaction_OS
    }

    # Step 2: 处理多个工具调用
    assistant_message = guarded_assistant_message(
        response,
        context="r_group_agent.product_variant_os.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_calls = tool_calls_or_empty(assistant_message, context="r_group_agent.product_variant_os.tool_selection")
    if not tool_calls:
        raise RuntimeStageError(
            "assistant message has no tool calls",
            context="r_group_agent.product_variant_os.tool_selection",
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
        #response_format={'type': 'json_object'},
            )

    # 获取 GPT 生成的结果
    raw_content = message_content(
        response,
        context="r_group_agent.product_variant_os.final_response",
        required=True,
        retry_trigger="auto_recovery_retry",
    )
    
    # 检查内容是否为空
    if not raw_content or not raw_content.strip():
        print(f"ERROR [OS]: Model returned empty content")
        print(f"Full response object: {response}")
        raise ValueError("Model returned empty content. Please check the model response.")
    
    print(f"DEBUG [OS]: Raw content preview (first 500 chars):\n{raw_content[:500]}")
    
    # 尝试解析 JSON（支持从包含思考过程的文本中提取）
    gpt_output = None
    
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
    
    print("R_group_agent_output:", gpt_output)
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # 使用 OS 版本的缓存函数
    coref_results = get_cached_multi_molecular_OS(image_path)
    raw_results = get_cached_raw_results_OS(image_path)
    reaction_results = first_dict_item(raw_results, context="r_group_agent.product_variant_os.raw_results", retry_trigger="auto_recovery_retry")
    
    reaction = {
        "reactants": reaction_results.get('reactants', []),
        "conditions": reaction_results.get('conditions', []),
        "products": reaction_results.get('products', [])
    }
    reaction_results = [{"reactions": [reaction]}]

    # 定义更新工具输出的函数
    def extract_smiles_details(smiles_data, raw_details):
        smiles_details = {}
        for smiles in smiles_data:
            for detail in raw_details:
                for bbox in detail.get('bboxes', []):
                    if bbox.get('smiles') == smiles:
                        smiles_details[smiles] = {
                            'category': bbox.get('category'),
                            'bbox': bbox.get('bbox'),
                            'category_id': bbox.get('category_id'),
                            'score': bbox.get('score'),
                            'molfile': bbox.get('molfile'),
                            'atoms': bbox.get('atoms'),
                            'bonds': bbox.get('bonds'),
                        }
                        break
        return smiles_details

    # 获取结果
    smiles_details = extract_smiles_details(gpt_output, coref_results)

    reactants_array = []
    products = []

    first_reaction = first_dict_item(reaction_results.get('reactions', []), context="r_group_agent.product_variant_os.reactions", retry_trigger="auto_recovery_retry")
    for reactant in first_reaction.get('reactants', []):
        if 'smiles' in reactant:
            reactants_array.append(reactant['smiles'])

    for product in first_reaction.get('products', []):
        products.append(product['smiles'])

    # 整理反应数据
    toolkit, _ = _get_runtime_models()
    backed_out = utils.backout_without_coref(reaction_results, coref_results, gpt_output, smiles_details, toolkit.molnextr)
    backed_out.sort(key=lambda x: x[2])
    extracted_rxns = {}
    for reactants, products_, label in backed_out:
        extracted_rxns[label] = {'reactants': reactants, 'products': products_}
    
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]:
                bbox.pop(key, None)  # 安全地移除键

    data = first_dict_item(coref_results, context="r_group_agent.process_variant_os.parse_coref", retry_trigger="auto_recovery_retry")
    parsed = parse_coref_data_with_fallback(data)
    
    toadd = {
        "reaction_template": {
            "reactants": reactants_array,
            "products": products
        },
        "reactions": extracted_rxns,
        "original_molecule_list": gpt_output
    }

    # 按标签排序
    sorted_keys = sorted(toadd["reactions"].keys())
    toadd["reactions"] = {i: toadd["reactions"][i] for i in sorted_keys}
    toadd = normalize_product_variant_output(toadd)
    print(f"str_R_group_agent_output:{toadd}")
    return toadd


def process_reaction_image_with_table_R_group(image_path: str) -> dict:

    client = _get_azure_client()

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)
    with open('./prompt/prompt_Table_R.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction',
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
    tools = tools,
    )

    
    assistant_message = guarded_assistant_message(
        response,
        context="r_group_agent.table.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_call = first_tool_call(
        assistant_message,
        context="r_group_agent.table.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_name = tool_call.function.name  # 修改此处
    tool_arguments = tool_call.function.arguments  # 新增此处
    tool_call_id = tool_call.id

    tool_args = json.loads(tool_arguments)
    #image_path = tool_args.get('image_path', image_path)  # 使用模型提供的 image_path

    if tool_name == 'get_full_reaction':
        tool_result = get_full_reaction(image_path)

    else:
        raise ValueError(f"Unknown tool called: {tool_name}")
    #print(tool_result)

    # 构建工具调用结果消息
    function_call_result_message = {
        'role': 'tool',
        'name': tool_name,  # Gemini API 要求必须包含 name 字段
        'content': json.dumps({
            'image_path': image_path,
            f'{tool_name}':(tool_result),
    }),
        'tool_call_id': tool_call_id,
    }


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
            function_call_result_message,
        ],
    }

    # Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
    )

    #print(response)   


    def replace_symbols_and_generate_smiles(input1, input2):
        """
        通用函数，用于将输入2中的symbols替换到输入1中，并生成新的SMILES。
        返回的结果保持特定格式，不包含初始的反应数据。
        
        参数:
        input1: 包含reactants和products的初始输入数据
        input2: 包含不同反应的symbols信息的数据

        返回:
        一个新的包含每个reaction的字典，包含reaction_id、reactants和products。
        """
        
        reactions_output = {"reactions": []}  # 存储最终的反应输出
        
        # 遍历 input2 中的每个 reaction
        for reaction in input2['reactions']:
            reaction_id = reaction['reaction_id']
            
            # 构建新的 reaction 字典
            new_reaction = {"reaction_id": reaction_id, "reactants": [], "conditions":[], "products": [], "additional_info": []}

            # 遍历 input1 中的所有 reactants，保留文本类型，处理分子类型
            mol_idx = 0  # 用于跟踪 reaction['reactants'] 中的分子索引
            for j, original_reactant in enumerate(input1['reactants']):
                # 如果是文本类型，直接保留
                if 'coords' not in original_reactant or 'edges' not in original_reactant:
                    new_reactant = {
                        "category": original_reactant.get('category', '[Txt]'),
                        "bbox": original_reactant.get('bbox', []),
                        "text": original_reactant.get('text', []),
                    }
                    new_reaction["reactants"].append(new_reactant)
                else:
                    # 如果是分子类型，从 reaction['reactants'] 中获取对应的 symbols
                    if mol_idx < len(reaction['reactants']):
                        reactant = reaction['reactants'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_reactant = reactant['symbols']  # 替换为reaction中的symbols
                        new_smiles_reactant, __, __ = _convert_graph_to_smiles(original_reactant['coords'], new_symbols_reactant, original_reactant['edges'])  # 生成新的SMILES
                        
                        new_reactant = {
                            #"category": original_reactant['category'],
                            #"bbox": original_reactant['bbox'],
                            #"category_id": original_reactant['category_id'],
                            "smiles": new_smiles_reactant,
                            #"coords": original_reactant['coords'],
                            "symbols": new_symbols_reactant,
                            #"edges": original_reactant['edges']
                        }
                        new_reaction["reactants"].append(new_reactant)

            if 'conditions' in reaction:
                new_reaction['conditions'] = reaction['conditions']

            
            # 处理 products 中的每个分子
            # 遍历 input1 中的所有 products，保留文本类型，处理分子类型
            mol_idx = 0  # 用于跟踪 reaction['products'] 中的分子索引
            for k, original_product in enumerate(input1['products']):
                # 如果是文本类型，直接保留
                if 'coords' not in original_product or 'edges' not in original_product:
                    new_product = {
                        "category": original_product.get('category', '[Txt]'),
                        "bbox": original_product.get('bbox', []),
                        "text": original_product.get('text', []),
                    }
                    new_reaction["products"].append(new_product)
                else:
                    # 如果是分子类型，从 reaction['products'] 中获取对应的 symbols
                    if mol_idx < len(reaction['products']):
                        product = reaction['products'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_product = product['symbols']  # 替换为reaction中的symbols
                        new_smiles_product, __, __ = _convert_graph_to_smiles(original_product['coords'], new_symbols_product, original_product['edges'])  # 生成新的SMILES
                        
                        new_product = {
                            #"category": original_product['category'],
                            #"bbox": original_product['bbox'],
                            #"category_id": original_product['category_id'],
                            "smiles": new_smiles_product,
                            #"coords": original_product['coords'],
                            "symbols": new_symbols_product,
                            #"edges": original_product['edges']
                        }
                        new_reaction["products"].append(new_product)
            
            if 'additional_info' in reaction:
                new_reaction['additional_info'] = reaction['additional_info']

            reactions_output['reactions'].append(new_reaction)  

        return reactions_output
    

    reaction_preds = tool_result['reaction_prediction']
    if isinstance(reaction_preds, str):
        # 如果是字符串，就 parse
        tool_result_json = json.loads(reaction_preds)
    elif isinstance(reaction_preds, (dict, list)):
        # 已经是 dict 或 list，直接使用
        tool_result_json = reaction_preds
    else:
        raise TypeError(f"Unexpected tool_result type: {type(reaction_preds)}")

    input1 = first_dict_item(tool_result_json, context="r_group_agent.table.tool_result_json", retry_trigger="auto_recovery_retry")
    input2 = safe_json_loads(
        message_content(
            response,
            context="r_group_agent.table.final_response",
            required=True,
            retry_trigger="auto_recovery_retry",
        ),
        context="r_group_agent.table.final_response",
        retry_trigger="auto_recovery_retry",
    )
    updated_input = replace_symbols_and_generate_smiles(input1, input2)
    print(f"txt_R_group_agent_output:{updated_input}")
    return updated_input


def process_reaction_image_with_table_R_group_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
    # model_name="gemini-2.5-flash",
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
    # api_key="AIzaSyBL8j4MbHAPhq8cR4Y05o9tY5Zq6fMDU3g"  
) -> dict:
    """
    与 process_reaction_image_with_table_R_group 流程保持一致，但改用兼容 OpenAI Chat Completions 协议的本地/自建模型（如 vLLM 或 Ollama）。

    Args:
        image_path: 反应图像路径。
        model_name: 本地模型名称（默认 `Qwen/Qwen3-VL-8B-Instruct`）。
        base_url: OpenAI 兼容接口地址，若为 None 则使用 `http://localhost:8000/v1` (vLLM 默认端口)。
        api_key: 接口密钥，可为任意非空字符串（vLLM 默认可填 `"EMPTY"`）。

    Returns:
        dict: 整理后的反应数据，包含 R-group 表格信息。
    """
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
    with open('./prompt/prompt_Table_R.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_full_reaction_OS',
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

    # 调用 GPT 接口（带重试机制）
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
                #response_format={'type': 'json_object'},
        messages=[
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
        ],
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = guarded_assistant_message(
        response,
        context="r_group_agent.table_os.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_calls = tool_calls_or_empty(assistant_message, context="r_group_agent.table_os.tool_selection")
    if not tool_calls:
        raise RuntimeStageError(
            "assistant message has no tool calls",
            context="r_group_agent.table_os.tool_selection",
            retry_trigger="auto_recovery_retry",
        )

    tool_call = first_tool_call(
        assistant_message,
        context="r_group_agent.table_os.tool_selection",
        retry_trigger="auto_recovery_retry",
    )
    tool_name = tool_call.function.name
    tool_arguments = tool_call.function.arguments
    tool_call_id = tool_call.id

    tool_args = json.loads(tool_arguments)

    if tool_name == 'get_full_reaction_OS':
        tool_result = get_full_reaction_OS(image_path)
    else:
        raise ValueError(f"Unknown tool called: {tool_name}")

    # 构建工具调用结果消息
    function_call_result_message = {
        'role': 'tool',
        'name': tool_name,  # Gemini API 要求必须包含 name 字段
        'content': json.dumps({
            'image_path': image_path,
            f'{tool_name}':(tool_result),
        }),
        'tool_call_id': tool_call_id,
    }

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
            function_call_result_message,
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
        #response_format={'type': 'json_object'},
            )
    
    preview_content = message_content(
        response,
        context="r_group_agent.table_os.final_response",
        required=True,
        retry_trigger="auto_recovery_retry",
    )
    print(f"DEBUG [OS]: Model response content type: {type(preview_content)}")
    print(f"DEBUG [OS]: Model response content preview: {str(preview_content)[:500]}")

    def replace_symbols_and_generate_smiles(input1, input2):
        """
        通用函数，用于将输入2中的symbols替换到输入1中，并生成新的SMILES。
        返回的结果保持特定格式，不包含初始的反应数据。
        
        参数:
        input1: 包含reactants和products的初始输入数据
        input2: 包含不同反应的symbols信息的数据

        返回:
        一个新的包含每个reaction的字典，包含reaction_id、reactants和products。
        """
        
        reactions_output = {"reactions": []}  # 存储最终的反应输出
        
        # 验证 input2 格式
        if not isinstance(input2, dict):
            raise ValueError(f"Expected input2 to be a dict, but got {type(input2)}: {input2}")
        
        if 'reactions' not in input2:
            print(f"ERROR [OS]: 'reactions' key not found in input2.")
            print(f"Available keys: {list(input2.keys())}")
            print(f"Full input2 content:\n{json.dumps(input2, indent=2, ensure_ascii=False)}")
            raise KeyError(f"'reactions' key not found in model response. Available keys: {list(input2.keys())}. "
                          f"Please check the model response format and the prompt file './prompt/prompt_Table_R.txt'. "
                          f"The model may not be following the expected JSON schema.")
        
        # 遍历 input2 中的每个 reaction
        for reaction in input2['reactions']:
            reaction_id = reaction['reaction_id']
            
            # 构建新的 reaction 字典
            new_reaction = {"reaction_id": reaction_id, "reactants": [], "conditions":[], "products": [], "additional_info": []}

            # 遍历 input1 中的所有 reactants，保留文本类型，处理分子类型
            mol_idx = 0  # 用于跟踪 reaction['reactants'] 中的分子索引
            for j, original_reactant in enumerate(input1['reactants']):
                # 如果是文本类型，直接保留
                if 'coords' not in original_reactant or 'edges' not in original_reactant:
                    new_reactant = {
                        "category": original_reactant.get('category', '[Txt]'),
                        "bbox": original_reactant.get('bbox', []),
                        "text": original_reactant.get('text', []),
                    }
                    new_reaction["reactants"].append(new_reactant)
                else:
                    # 如果是分子类型，从 reaction['reactants'] 中获取对应的 symbols
                    if mol_idx < len(reaction['reactants']):
                        reactant = reaction['reactants'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_reactant = reactant['symbols']  # 替换为reaction中的symbols
                        new_smiles_reactant, __, __ = _convert_graph_to_smiles(original_reactant['coords'], new_symbols_reactant, original_reactant['edges'])  # 生成新的SMILES
                        
                        new_reactant = {
                            "smiles": new_smiles_reactant,
                            "symbols": new_symbols_reactant,
                        }
                        new_reaction["reactants"].append(new_reactant)

            if 'conditions' in reaction:
                new_reaction['conditions'] = reaction['conditions']

            
            # 处理 products 中的每个分子
            # 遍历 input1 中的所有 products，保留文本类型，处理分子类型
            mol_idx = 0  # 用于跟踪 reaction['products'] 中的分子索引
            for k, original_product in enumerate(input1['products']):
                # 如果是文本类型，直接保留
                if 'coords' not in original_product or 'edges' not in original_product:
                    new_product = {
                        "category": original_product.get('category', '[Txt]'),
                        "bbox": original_product.get('bbox', []),
                        "text": original_product.get('text', []),
                    }
                    new_reaction["products"].append(new_product)
                else:
                    # 如果是分子类型，从 reaction['products'] 中获取对应的 symbols
                    if mol_idx < len(reaction['products']):
                        product = reaction['products'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_product = product['symbols']  # 替换为reaction中的symbols
                        new_smiles_product, __, __ = _convert_graph_to_smiles(original_product['coords'], new_symbols_product, original_product['edges'])  # 生成新的SMILES
                        
                        new_product = {
                            "smiles": new_smiles_product,
                            "symbols": new_symbols_product,
                        }
                        new_reaction["products"].append(new_product)
            
            if 'additional_info' in reaction:
                new_reaction['additional_info'] = reaction['additional_info']

            reactions_output['reactions'].append(new_reaction)  

        return reactions_output
    

    reaction_preds = tool_result['reaction_prediction']
    if isinstance(reaction_preds, str):
        # 如果是字符串，就 parse
        tool_result_json = json.loads(reaction_preds)
    elif isinstance(reaction_preds, (dict, list)):
        # 已经是 dict 或 list，直接使用
        tool_result_json = reaction_preds
    else:
        raise TypeError(f"Unexpected tool_result type: {type(reaction_preds)}")

    input1 = first_dict_item(tool_result_json, context="r_group_agent.table_os.tool_result_json", retry_trigger="auto_recovery_retry")
    
    # 获取模型返回的原始内容
    raw_content = preview_content
    
    # 检查内容是否为空
    if not raw_content or not raw_content.strip():
        print(f"ERROR [OS]: Model returned empty content")
        print(f"Full response object: {response}")
        raise ValueError("Model returned empty content. Please check the model response.")
    
    print(f"DEBUG [OS]: Raw content type: {type(raw_content)}")
    print(f"DEBUG [OS]: Raw content length: {len(raw_content)}")
    print(f"DEBUG [OS]: Raw content preview (first 500 chars):\n{raw_content[:500]}")
    
    # 尝试解析 JSON（支持从包含思考过程的文本中提取）
    input2 = None
    
    try:
        # 首先尝试直接解析
        input2 = json.loads(raw_content)
        print(f"DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        # 如果直接解析失败，使用智能提取函数
        print(f"WARNING [OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        input2 = extract_json_from_text_with_reasoning(raw_content)
        
        if input2 is not None:
            print(f"DEBUG [OS]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                f"Could not parse JSON from model response. Content may not be valid JSON.",
                raw_content, 0
            )
    
    # 验证 input2 的格式
    print(f"DEBUG [OS]: input2 type: {type(input2)}")
    if isinstance(input2, dict):
        print(f"DEBUG [OS]: input2 keys: {list(input2.keys())}")
        print(f"DEBUG [OS]: input2 content preview (first 1000 chars):\n{json.dumps(input2, indent=2, ensure_ascii=False)[:1000]}")
    else:
        print(f"DEBUG [OS]: input2 is not a dict, value: {input2}")
    
    updated_input = replace_symbols_and_generate_smiles(input1, input2)
    print(f"txt_R_group_agent_output:{updated_input}")
    return updated_input
