from __future__ import annotations

import hashlib
import copy
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem

from llm_preflight import RunFailureController, RunFailureControllerState, classify_provider_exception
from review_artifacts import ArtifactStore, create_artifact_store_from_config
from review_db import ReviewRepository
from review_logging import RunLogSession, bind_log_context, get_log_download_ref, read_log_tail
from review_renderer import render_reaction_png
from review_tracking import RunMetricsCollector, bind_metrics_collector


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
PDF_SUFFIXES = {".pdf"}
RUN_PHASE_LABELS = {
    "queued": "Queued",
    "prepare_source": "Preparing source",
    "extract_pdf_images": "Extracting PDF figures",
    "process_image": "Analyzing image",
    "normalize": "Normalizing results",
    "render": "Normalizing results",
    "finalize": "Finalizing",
    "completed": "Completed",
    "failed": "Failed",
    "interrupted": "Interrupted",
}

PLACEHOLDER_TOKENS = ("R", "R1", "R2", "R3", "R4", "PG")
PLACEHOLDER_PATTERN = re.compile(r"\[(R(?:\d+)?|PG)\]")
TRANSIENT_RETRY_KINDS = {"timeout", "rate_limited", "provider_overloaded"}
LOCAL_RECOVERY_EXCEPTIONS = (IndexError, AttributeError, TypeError, KeyError, ValueError)
MAX_AUTO_ATTEMPTS_PER_DERIVED_IMAGE = 3
MAX_MANUAL_ATTEMPTS_PER_DERIVED_IMAGE = 5


@dataclass(frozen=True)
class ReactionCandidate:
    candidate_index: int
    source_path: str
    raw_reaction: Dict[str, Any]
    raw_candidate_hash: str


@dataclass(frozen=True)
class CandidateValidationResult:
    candidate: ReactionCandidate
    accepted: bool
    structure_quality: str
    acceptance_reason: str
    normalized_molecules: List[Dict[str, Any]]
    normalized_conditions: List[Dict[str, Any]]
    normalized_additional_info: List[Dict[str, Any]]
    rejection_reason: str


@dataclass(frozen=True)
class NormalizationSummary:
    accepted_reaction_count: int
    rejected_reaction_count: int
    normalization_status: str
    normalization_summary: str
    canonical_reactions: List[Dict[str, Any]]
    canonical_molecules: List[List[Dict[str, Any]]]
    canonical_conditions: List[List[Dict[str, Any]]]
    canonical_additional_info: List[List[Dict[str, Any]]]
    structure_qualities: List[str]
    acceptance_reasons: List[str]


class RunAbortedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        controller_state: Optional[RunFailureControllerState] = None,
        summary_delta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.controller_state = controller_state or RunFailureControllerState()
        self.summary_delta = summary_delta or {}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_set_fingerprint(paths: Sequence[str]) -> str:
    return _hash_text(json.dumps(sorted(str(Path(path).expanduser().resolve()) for path in paths), ensure_ascii=False))


def _config_hash(config: Dict[str, Any]) -> str:
    return _hash_text(json.dumps(config, sort_keys=True, ensure_ascii=False, default=str))


def _valid_smiles(smiles: str) -> bool:
    text = (smiles or "").strip()
    if not text or text.lower() == "none":
        return False
    return Chem.MolFromSmiles(text) is not None


def _normalized_smiles_text(smiles: Any) -> str:
    text = str(smiles or "").strip()
    return "" if not text or text.lower() == "none" else text


def _placeholder_smiles(smiles: str) -> str:
    return PLACEHOLDER_PATTERN.sub("[*]", smiles or "")


def _template_valid_smiles(smiles: str) -> bool:
    text = _normalized_smiles_text(smiles)
    if not text:
        return False
    if not PLACEHOLDER_PATTERN.search(text):
        return False
    normalized = _placeholder_smiles(text)
    return Chem.MolFromSmiles(normalized) is not None


def _validation_kind(smiles: Any) -> str:
    text = _normalized_smiles_text(smiles)
    if not text:
        return "missing"
    if _valid_smiles(text):
        return "rdkit_valid"
    if _template_valid_smiles(text):
        return "template_valid"
    return "invalid"


def _reaction_fingerprint(reaction: Dict[str, Any]) -> str:
    payload = {
        "reactants": reaction.get("reactants", []),
        "products": reaction.get("products", []),
        "conditions": reaction.get("conditions", []),
        "additional_info": reaction.get("additional_info", []),
    }
    return _hash_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _iter_reaction_lists(node: Any, source_path: str) -> Iterable[Tuple[str, List[Dict[str, Any]]]]:
    if isinstance(node, dict):
        reactions = node.get("reactions")
        if isinstance(reactions, list):
            items = [item for item in reactions if isinstance(item, dict)]
            if items:
                yield source_path, items
        for key, value in node.items():
            if key == "reactions":
                continue
            next_path = f"{source_path}.{key}" if source_path else str(key)
            yield from _iter_reaction_lists(value, next_path)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _iter_reaction_lists(item, f"{source_path}[{index}]")


def extract_reaction_candidates(payload: Dict[str, Any]) -> List[ReactionCandidate]:
    if not isinstance(payload, dict):
        return []
    candidates: List[ReactionCandidate] = []
    seen: set[str] = set()
    candidate_index = 0
    top_level = payload.get("reactions")
    if isinstance(top_level, list):
        for item in top_level:
            if not isinstance(item, dict):
                continue
            signature = _hash_text(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                ReactionCandidate(
                    candidate_index=candidate_index,
                    source_path="top_level",
                    raw_reaction=item,
                    raw_candidate_hash=signature,
                )
            )
            candidate_index += 1
    for entry_index, entry in enumerate(payload.get("execution_logs", []) or []):
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        for nested_path, reactions in _iter_reaction_lists(result, f"execution_logs[{entry_index}].result"):
            for item in reactions:
                signature = _hash_text(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str))
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(
                    ReactionCandidate(
                        candidate_index=candidate_index,
                        source_path=nested_path,
                        raw_reaction=item,
                        raw_candidate_hash=signature,
                    )
                )
                candidate_index += 1
    return candidates


def _normalize_additional_info(items: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    ordinal = 0
    for item in items or []:
        if isinstance(item, dict):
            item_type = str(item.get("type") or "")
            if "text" in item and len(item) == 1:
                normalized.append(
                    {
                        "ordinal": ordinal,
                        "kind": item_type or "text",
                        "key": "",
                        "value": "",
                        "raw_text": str(item.get("text") or ""),
                    }
                )
                ordinal += 1
                continue
            for key, value in item.items():
                if key == "type":
                    continue
                normalized.append(
                    {
                        "ordinal": ordinal,
                        "kind": item_type or ("kv" if key != "text" else "text"),
                        "key": "" if key == "text" else str(key),
                        "value": "" if key == "text" else str(value),
                        "raw_text": str(value) if key == "text" else f"{key}: {value}",
                    }
                )
                ordinal += 1
            continue
        normalized.append(
            {
                "ordinal": ordinal,
                "kind": "text",
                "key": "",
                "value": "",
                "raw_text": str(item),
            }
        )
        ordinal += 1
    return normalized


def _normalize_molecules(reaction: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for side in ("reactants", "products"):
        items = reaction.get(side, []) or []
        if isinstance(items, dict) or isinstance(items, str):
            items = [items]
        for ordinal, item in enumerate(items):
            if isinstance(item, dict):
                smiles_value = item.get("smiles")
                label_value = item.get("label")
            else:
                smiles_value = item
                label_value = ""
            smiles = _normalized_smiles_text(smiles_value)
            validation_kind = _validation_kind(smiles)
            normalized.append(
                {
                    "side": side[:-1],
                    "ordinal": ordinal,
                    "smiles": smiles,
                    "label": str(label_value or ""),
                    "valid_smiles": validation_kind in {"rdkit_valid", "template_valid"},
                    "validation_kind": validation_kind,
                }
            )
    return normalized


def _normalize_conditions(reaction: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    items = reaction.get("conditions", []) or []
    if isinstance(items, dict) or isinstance(items, str):
        items = [items]
    for ordinal, item in enumerate(items):
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("category") or "")
            text = str(item.get("text") or "")
            smiles = str(item.get("smiles") or "")
        else:
            role = ""
            text = str(item)
            smiles = ""
        normalized.append(
            {
                "ordinal": ordinal,
                "role": role,
                "text": text,
                "smiles": smiles,
            }
        )
    return normalized


def validate_reaction_candidate(candidate: ReactionCandidate) -> CandidateValidationResult:
    reaction = candidate.raw_reaction
    normalized_molecules = _normalize_molecules(reaction)
    normalized_conditions = _normalize_conditions(reaction)
    normalized_additional_info = _normalize_additional_info(reaction.get("additional_info", []))
    accepted_reactants = [
        item for item in normalized_molecules if item.get("side") == "reactant" and item.get("valid_smiles")
    ]
    accepted_products = [
        item for item in normalized_molecules if item.get("side") == "product" and item.get("valid_smiles")
    ]
    if accepted_reactants and accepted_products:
        structure_quality = (
            "template_valid"
            if any(item.get("validation_kind") == "template_valid" for item in accepted_reactants + accepted_products)
            else "rdkit_valid"
        )
        return CandidateValidationResult(
            candidate=candidate,
            accepted=True,
            structure_quality=structure_quality,
            acceptance_reason="accepted_structure_gate",
            normalized_molecules=normalized_molecules,
            normalized_conditions=normalized_conditions,
            normalized_additional_info=normalized_additional_info,
            rejection_reason="",
        )
    rejected_molecules = [item for item in normalized_molecules if item.get("validation_kind") in {"invalid", "missing"}]
    if not normalized_molecules or all(item.get("validation_kind") == "missing" for item in normalized_molecules):
        rejection_reason = "rejected_missing_smiles"
    elif rejected_molecules:
        rejection_reason = "rejected_invalid_smiles"
    else:
        rejection_reason = "rejected_missing_smiles"
    return CandidateValidationResult(
        candidate=candidate,
        accepted=False,
        structure_quality="",
        acceptance_reason="",
        normalized_molecules=normalized_molecules,
        normalized_conditions=normalized_conditions,
        normalized_additional_info=normalized_additional_info,
        rejection_reason=rejection_reason,
    )


def accept_reaction_candidate(validation_result: CandidateValidationResult) -> bool:
    return validation_result.accepted


def _extract_reactions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [candidate.raw_reaction for candidate in extract_reaction_candidates(payload)]


def _normalize_reaction_candidates(payload: Dict[str, Any]) -> NormalizationSummary:
    candidates = extract_reaction_candidates(payload)
    validations = [validate_reaction_candidate(candidate) for candidate in candidates]
    accepted = [item for item in validations if accept_reaction_candidate(item)]
    rejected = [item for item in validations if not item.accepted]
    rejected_reasons = {item.rejection_reason for item in rejected if item.rejection_reason}
    if accepted and rejected:
        normalization_status = "partial"
    elif accepted:
        normalization_status = "accepted"
    elif payload.get("redo"):
        normalization_status = "redo_pending"
    elif not candidates:
        normalization_status = "none_found"
    elif rejected_reasons == {"rejected_missing_smiles"}:
        normalization_status = "rejected_missing_smiles"
    else:
        normalization_status = "rejected_invalid_smiles"
    summary_bits = [
        f"candidates={len(candidates)}",
        f"accepted={len(accepted)}",
        f"rejected={len(rejected)}",
    ]
    if rejected_reasons:
        summary_bits.append("reasons=" + ",".join(sorted(rejected_reasons)))
    return NormalizationSummary(
        accepted_reaction_count=len(accepted),
        rejected_reaction_count=len(rejected),
        normalization_status=normalization_status,
        normalization_summary=" | ".join(summary_bits),
        canonical_reactions=[item.candidate.raw_reaction for item in accepted],
        canonical_molecules=[item.normalized_molecules for item in accepted],
        canonical_conditions=[item.normalized_conditions for item in accepted],
        canonical_additional_info=[item.normalized_additional_info for item in accepted],
        structure_qualities=[item.structure_quality for item in accepted],
        acceptance_reasons=[item.acceptance_reason for item in accepted],
    )


def _classify_payload(payload: Dict[str, Any], *, artifact_missing: bool = False) -> str:
    if not isinstance(payload, dict):
        return "failed"
    if artifact_missing:
        return "imported_without_artifact"
    if payload.get("error"):
        return "failed"
    normalization = _normalize_reaction_candidates(payload)
    if normalization.accepted_reaction_count > 0:
        return "succeeded"
    if payload.get("redo") or normalization.normalization_status == "redo_pending":
        return "needs_redo"
    if normalization.normalization_status == "none_found":
        return "empty"
    if extract_reaction_candidates(payload):
        return "failed"
    return "failed"


def _resolve_path_candidates(upload: Any) -> List[str]:
    if upload is None:
        return []
    if isinstance(upload, (str, os.PathLike)):
        return [str(upload)]
    if isinstance(upload, list):
        paths: List[str] = []
        for item in upload:
            paths.extend(_resolve_path_candidates(item))
        return paths
    name = getattr(upload, "name", "")
    return [name] if name else []


def _search_candidates(
    *,
    file_name: str,
    json_path: Path,
    recovery_roots: Sequence[str],
) -> List[Path]:
    candidate_roots = [
        json_path.parent,
        json_path.parent / "images",
        json_path.parent / "pdf_images",
        Path.cwd(),
        Path.cwd() / "debug",
        Path.cwd() / "debug" / "pdf_images",
    ]
    candidate_roots.extend(Path(root).expanduser().resolve() for root in recovery_roots if root)
    seen = set()
    found: List[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        root_resolved = root.resolve()
        if str(root_resolved) in seen:
            continue
        seen.add(str(root_resolved))
        direct = root_resolved / file_name
        if direct.exists():
            found.append(direct)
        found.extend(sorted(path for path in root_resolved.rglob(file_name) if path.is_file()))
    return found


def _recover_file(
    *,
    payload: Dict[str, Any],
    file_name: str,
    json_path: Path,
    recovery_roots: Sequence[str],
) -> Optional[Path]:
    for section in ("execution_logs", "plan"):
        for entry in payload.get(section, []) or []:
            arguments = entry.get("arguments", {}) if isinstance(entry, dict) else {}
            if not isinstance(arguments, dict):
                continue
            image_path = str(arguments.get("image_path") or "").strip()
            if image_path and Path(image_path).exists():
                return Path(image_path).resolve()
    for candidate in _search_candidates(file_name=file_name, json_path=json_path, recovery_roots=recovery_roots):
        return candidate.resolve()
    return None


def _store_json(store: ArtifactStore, key: str, payload: Any) -> None:
    store.put_bytes(key, _json_dumps(payload).encode("utf-8"), "application/json")


def _payload_display_ref(store: ArtifactStore, artifact_key: str) -> str:
    if not artifact_key:
        return ""
    try:
        return str(store.get_download_ref(artifact_key) or artifact_key)
    except Exception:
        return artifact_key


def _rewrite_payload_image_refs(
    payload: Any,
    *,
    store: ArtifactStore,
    artifact_key: str,
    artifact_backend: str,
) -> Any:
    artifact_ref = _payload_display_ref(store, artifact_key)

    def _rewrite(node: Any) -> Any:
        if isinstance(node, dict):
            rewritten: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "image_path" and isinstance(value, str):
                    rewritten[key] = artifact_ref or artifact_key or value
                    rewritten["original_image_path"] = value
                    rewritten["image_artifact_backend"] = artifact_backend
                    rewritten["image_artifact_key"] = artifact_key
                    rewritten["image_artifact_ref"] = artifact_ref or artifact_key
                else:
                    rewritten[key] = _rewrite(value)
            return rewritten
        if isinstance(node, list):
            return [_rewrite(item) for item in node]
        return node

    return _rewrite(payload)


def _artifact_to_local_path(store: ArtifactStore, artifact_key: str, *, suffix: str = ".png") -> Path:
    ref = store.get_download_ref(artifact_key)
    candidate = Path(str(ref)).expanduser()
    if candidate.exists():
        return candidate.resolve()
    temp = tempfile.NamedTemporaryFile(prefix="review_retry_", suffix=suffix, delete=False)
    temp.write(store.get_bytes(artifact_key))
    temp.flush()
    temp.close()
    return Path(temp.name).resolve()


def _initial_summary(total_sources: int = 0) -> Dict[str, Any]:
    return {
        "total_sources": total_sources,
        "total_derived_images": 0,
        "total_reactions": 0,
        "total_failures": 0,
        "total_redo": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": None,
        "usage_completeness": "none",
    }


def _merge_summary(summary: Dict[str, Any], delta: Dict[str, Any]) -> None:
    summary["total_derived_images"] += int(delta.get("derived_count", 0))
    summary["total_reactions"] += int(delta.get("reaction_count", 0))
    summary["total_failures"] += int(delta.get("failure_count", 0))
    summary["total_redo"] += int(delta.get("redo_count", 0))
    summary["prompt_tokens"] += int(delta.get("prompt_tokens", 0))
    summary["completion_tokens"] += int(delta.get("completion_tokens", 0))
    summary["total_tokens"] += int(delta.get("total_tokens", 0))
    cost = delta.get("estimated_cost_usd")
    if cost is not None:
        summary["estimated_cost_usd"] = round(float(summary.get("estimated_cost_usd") or 0.0) + float(cost), 8)
    summary["usage_completeness"] = _merge_usage_completeness(
        str(summary.get("usage_completeness", "none")),
        str(delta.get("usage_completeness", "none")),
    )


def _empty_metrics_summary() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": None,
        "usage_completeness": "none",
        "call_count": 0,
    }


def _exception_metrics(exc: Exception) -> RunMetricsCollector:
    collector = getattr(exc, "_metrics_collector", None)
    return collector if isinstance(collector, RunMetricsCollector) else RunMetricsCollector()


def _provider_diagnostics(exc: Exception) -> Dict[str, Any]:
    diagnostics = getattr(exc, "_provider_diagnostics", None)
    return diagnostics if isinstance(diagnostics, dict) else {}


def _diagnostic_error_summary(exc: Exception, diagnostics: Dict[str, Any]) -> str:
    parts = [f"{exc.__class__.__name__}: {exc}"]
    if diagnostics.get("llm_stage") or diagnostics.get("llm_phase"):
        parts.append(
            f"stage={diagnostics.get('llm_stage') or 'unknown'} phase={diagnostics.get('llm_phase') or 'unknown'}"
        )
    if diagnostics.get("provider") or diagnostics.get("model"):
        parts.append(
            f"provider={diagnostics.get('provider') or 'unknown'} model={diagnostics.get('model') or 'unknown'}"
        )
    if diagnostics.get("base_url"):
        parts.append(f"endpoint={diagnostics['base_url']}")
    if diagnostics.get("cause_class") or diagnostics.get("cause_message"):
        parts.append(
            f"cause={diagnostics.get('cause_class') or 'unknown'}: {diagnostics.get('cause_message') or ''}".strip()
        )
    return " | ".join(part for part in parts if part)


def _retry_backoff_seconds(attempt_no: int) -> float:
    return min(8.0, float(2 ** max(0, attempt_no - 1)))


def _execution_mode_for_trigger(trigger: str, *, fallback: str = "normal") -> str:
    if trigger in {"auto_no_agents_retry", "manual_no_agents_retry"}:
        return "no_agents"
    if trigger in {"auto_recovery_retry", "manual_reprocess"}:
        return "recovery"
    return fallback


def _should_retry_failure(failure, exc: Exception, attempt_no: int, *, current_execution_mode: str) -> Tuple[bool, str]:
    if attempt_no >= MAX_AUTO_ATTEMPTS_PER_DERIVED_IMAGE:
        return False, "retry_cap_reached"
    explicit_trigger = str(getattr(exc, "_retry_trigger", "") or "")
    if explicit_trigger:
        return True, explicit_trigger
    if failure.kind in TRANSIENT_RETRY_KINDS or failure.retryable:
        return True, "auto_transient_retry"
    if isinstance(exc, LOCAL_RECOVERY_EXCEPTIONS):
        if current_execution_mode == "no_agents":
            return True, "auto_recovery_retry"
        return True, "auto_recovery_retry"
    return False, ""


def _should_retry_redo(
    *,
    payload: Dict[str, Any],
    normalization: NormalizationSummary,
    attempt_no: int,
    current_execution_mode: str,
) -> Tuple[bool, str]:
    if attempt_no >= MAX_AUTO_ATTEMPTS_PER_DERIVED_IMAGE:
        return False, ""
    if payload.get("redo") and normalization.accepted_reaction_count == 0:
        if current_execution_mode == "normal":
            return True, "auto_no_agents_retry"
        if current_execution_mode == "no_agents":
            return True, "auto_recovery_retry"
    return False, ""


def _source_summary_status(source_summary: Dict[str, Any]) -> str:
    expected = int(source_summary.get("expected_derived_images", 0))
    failed = int(source_summary.get("failed_derived_images", 0))
    completed = int(source_summary.get("completed_derived_images", 0))
    if source_summary.get("status") in {"aborted", "failed"}:
        return str(source_summary["status"])
    if expected and completed >= expected and failed >= expected and int(source_summary.get("reaction_count", 0)) == 0:
        return "failed"
    return "completed"


def _run_phase_label(run: Dict[str, Any]) -> str:
    status = str(run.get("status") or "").strip().lower()
    if status in {"queued", "completed", "failed", "interrupted"}:
        return RUN_PHASE_LABELS.get(status, status.title() or "Queued")
    phase = str(run.get("current_phase") or "").strip().lower()
    return RUN_PHASE_LABELS.get(phase, RUN_PHASE_LABELS.get(status, "Queued"))


def _run_progress_snapshot(run: Dict[str, Any], *, has_logs: bool) -> Dict[str, Any]:
    completed_sources = int(run.get("completed_sources") or 0)
    failed_sources = int(run.get("failed_sources") or 0)
    total_sources = int(run.get("total_sources") or 0)
    progress_completed = completed_sources + failed_sources
    progress_fraction = float(progress_completed / total_sources) if total_sources else 0.0
    current_source = str(run.get("current_source_name") or "").strip()
    current_phase_label = _run_phase_label(run)
    status = str(run.get("status") or "").strip().lower()
    if status == "queued":
        status_summary = "Queued and waiting to start."
    elif status == "running":
        status_summary = f"{current_phase_label}{f' for {current_source}' if current_source else ''}."
    elif status == "completed":
        status_summary = f"Completed {progress_completed}/{total_sources or progress_completed} sources."
    elif status == "failed":
        status_summary = str(run.get("failure_summary") or run.get("last_error_summary") or "Run failed.")
    elif status == "interrupted":
        status_summary = str(run.get("failure_summary") or "Run interrupted.")
    else:
        status_summary = current_phase_label
    return {
        "is_active": status in {"queued", "running"},
        "progress_completed_sources": progress_completed,
        "progress_total_sources": total_sources,
        "progress_fraction": progress_fraction,
        "progress_label": f"{progress_completed}/{total_sources or 0} sources finished",
        "current_phase_label": current_phase_label,
        "current_source_label": current_source or "Waiting for next source",
        "status_summary": status_summary,
        "has_troubleshooting_logs": has_logs,
    }


@dataclass
class RunTask:
    run_id: str
    experiment_id: str
    profile_label: str
    ingest_mode: str
    config_snapshot: Dict[str, Any]
    live_source_paths: List[str] = field(default_factory=list)
    sideload_json_path: str = ""
    recovery_roots: List[str] = field(default_factory=list)


class ReviewDatasetService:
    def __init__(self, repository: ReviewRepository):
        self.repository = repository
        self._queue: "queue.Queue[RunTask]" = queue.Queue()
        self._recover_interrupted_runs()
        self._worker = threading.Thread(target=self._worker_loop, name="review-dataset-worker", daemon=True)
        self._worker.start()

    def _recover_interrupted_runs(self) -> None:
        recovered = self.repository.mark_running_runs_interrupted()
        for row in recovered:
            self._refresh_experiment_status(str(row.get("experiment_id") or ""))

    def submit_live_experiment(
        self,
        *,
        experiment_name: str,
        notes: str,
        source_paths: Sequence[str],
        profile_configs: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        resolved_paths = [str(Path(path).expanduser().resolve()) for path in source_paths]
        experiment_id = self.repository.create_experiment(
            name=experiment_name or "Live Experiment",
            notes=notes,
            source_set_fingerprint=_source_set_fingerprint(resolved_paths),
            status="queued",
        )
        run_ids: List[str] = []
        for index, config in enumerate(profile_configs):
            profile_label = str(config.get("profile_label") or f"profile-{index + 1}")
            run_id = self.repository.create_run(
                experiment_id=experiment_id,
                profile_label=profile_label,
                ingest_mode="live_batch",
                status="queued",
                config_snapshot=config,
                config_hash=_config_hash(config),
            )
            preflight_status = str(config.get("preflight_status") or "")
            preflight_summary = str(config.get("preflight_summary") or "")
            if preflight_status or preflight_summary:
                self.repository.update_run_preflight(
                    run_id,
                    preflight_status=preflight_status or "passed",
                    preflight_summary=preflight_summary or "Batch runtime preflight passed.",
                )
            run_ids.append(run_id)
            self._queue.put(
                RunTask(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    profile_label=profile_label,
                    ingest_mode="live_batch",
                    config_snapshot=dict(config),
                    live_source_paths=list(resolved_paths),
                )
            )
        return {"experiment_id": experiment_id, "run_ids": run_ids}

    def submit_sideload_experiment(
        self,
        *,
        experiment_name: str,
        notes: str,
        json_paths: Sequence[str],
        recovery_roots: Sequence[str],
        config_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved_jsons = [str(Path(path).expanduser().resolve()) for path in json_paths]
        experiment_id = self.repository.create_experiment(
            name=experiment_name or "Sideload Experiment",
            notes=notes,
            source_set_fingerprint=_source_set_fingerprint(resolved_jsons),
            status="queued",
        )
        run_ids: List[str] = []
        for json_path in resolved_jsons:
            run_id = self.repository.create_run(
                experiment_id=experiment_id,
                profile_label=Path(json_path).name,
                ingest_mode="sideload_json",
                status="queued",
                config_snapshot=config_snapshot,
                config_hash=_config_hash({**config_snapshot, "sideload_json_path": json_path}),
            )
            run_ids.append(run_id)
            self._queue.put(
                RunTask(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    profile_label=Path(json_path).name,
                    ingest_mode="sideload_json",
                    config_snapshot=dict(config_snapshot),
                    sideload_json_path=json_path,
                    recovery_roots=[str(Path(root).expanduser().resolve()) for root in recovery_roots if root],
                )
            )
        return {"experiment_id": experiment_id, "run_ids": run_ids}

    def list_experiments(self) -> List[Dict[str, Any]]:
        return self.repository.list_experiments()

    def list_runs(self, experiment_id: str = "") -> List[Dict[str, Any]]:
        return self.repository.list_runs(experiment_id=experiment_id)

    def list_run_sources(self, run_id: str) -> List[Dict[str, Any]]:
        return self.repository.list_run_sources(run_id)

    def list_reactions(
        self,
        *,
        experiment_id: str = "",
        run_id: str = "",
        review_status: str = "",
        outcome_class: str = "",
    ) -> List[Dict[str, Any]]:
        return self.repository.list_reactions(
            experiment_id=experiment_id,
            run_id=run_id,
            review_status=review_status,
            outcome_class=outcome_class,
        )

    def get_reaction_detail(self, reaction_uid: str) -> Dict[str, Any]:
        return self.repository.get_reaction_detail(reaction_uid)

    def get_run_source_monitor(self, run_source_id: str) -> Dict[str, Any]:
        return self.repository.get_run_source_detail(run_source_id)

    def list_retry_candidates(self, run_id: str) -> List[Dict[str, Any]]:
        return self.repository.list_retry_candidates(run_id)

    def _ensure_attempt_history(
        self,
        *,
        derived_image_id: str,
        detail: Dict[str, Any],
        config_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        attempts = list(detail.get("attempts", []) or [])
        if attempts:
            return detail
        bootstrap_status = str(detail.get("status") or "completed")
        if bootstrap_status == "processing":
            bootstrap_status = "failed"
        bootstrap_attempt = self.repository.create_derived_image_attempt(
            derived_image_id=derived_image_id,
            trigger="initial",
            execution_mode="normal",
            status=bootstrap_status,
            config_snapshot_json=_json_dumps(config_snapshot),
        )
        self.repository.finalize_derived_image_attempt(
            str(bootstrap_attempt["attempt_id"]),
            {
                "status": bootstrap_status,
                "failure_kind": "",
                "error_summary": str(detail.get("error_text") or ""),
                "raw_artifact_key": str(detail.get("raw_artifact_key") or ""),
                "usage_completeness": "none",
            },
        )
        self.repository.finalize_derived_image_summary(
            derived_image_id,
            {
                "attempt_count": int(bootstrap_attempt["attempt_no"]),
                "last_attempt_id": str(bootstrap_attempt["attempt_id"]),
                "last_retry_reason": "initial",
            },
        )
        return self.repository.get_derived_image(derived_image_id)

    def retry_derived_image(
        self,
        derived_image_id: str,
        *,
        trigger: str = "manual_retry",
        execution_mode: str = "normal",
        force: bool = False,
    ) -> str:
        detail = self.repository.get_derived_image(derived_image_id)
        attempts = detail.get("attempts", [])
        max_attempts = MAX_MANUAL_ATTEMPTS_PER_DERIVED_IMAGE if force else min(
            MAX_MANUAL_ATTEMPTS_PER_DERIVED_IMAGE,
            max(1, MAX_MANUAL_ATTEMPTS_PER_DERIVED_IMAGE - len(attempts) + 1),
        )
        if not detail.get("artifact_key"):
            raise RuntimeError(f"Derived image {derived_image_id} has no persisted artifact to retry.")
        run = self.repository.get_run(str(detail["run_id"]))
        if run is None:
            raise RuntimeError(f"Run {detail['run_id']} not found for derived image {derived_image_id}.")
        config_snapshot = json.loads(run.get("config_snapshot_json") or "{}")
        detail = self._ensure_attempt_history(
            derived_image_id=derived_image_id,
            detail=detail,
            config_snapshot=config_snapshot,
        )
        if execution_mode == "no_agents" and trigger == "manual_retry":
            trigger = "manual_no_agents_retry"
        store = create_artifact_store_from_config(config_snapshot)
        local_image = _artifact_to_local_path(store, str(detail["artifact_key"]), suffix=Path(str(detail["artifact_key"])).suffix or ".png")
        controller = RunFailureController()
        log_session = RunLogSession(run_id=str(detail["run_id"]), experiment_id=str(run["experiment_id"]))
        run_logger = log_session.logger.bind(phase="process_image", run_source_id=str(detail["run_source_id"]))
        try:
            result = self._process_derived_image(
                task=RunTask(
                    run_id=str(detail["run_id"]),
                    experiment_id=str(run["experiment_id"]),
                    profile_label=str(run.get("profile_label") or ""),
                    ingest_mode=str(run.get("ingest_mode") or "live_batch"),
                    config_snapshot=config_snapshot,
                ),
                store=store,
                run_source_id=str(detail["run_source_id"]),
                derived_image_id=derived_image_id,
                image_path=local_image,
                derived_artifact_key=str(detail["artifact_key"]),
                image_index=int(detail.get("image_index", 0)),
                controller=controller,
                source_index=0,
                source_name=str(detail.get("original_filename") or detail.get("page_hint") or derived_image_id),
                run_logger=run_logger,
                trigger=trigger,
                execution_mode=execution_mode,
                max_attempts=max_attempts,
            )
            self._refresh_source_and_run_counts(str(detail["run_source_id"]), str(detail["run_id"]))
            return (
                f"Retried {derived_image_id} using {execution_mode}: "
                f"reactions={result['reaction_count']} failures={result['failure_count']} redo={result['redo_count']}"
            )
        finally:
            log_session.close()

    def retry_failed_derived_images(
        self,
        run_id: str,
        *,
        include_needs_redo: bool = True,
        include_failed: bool = True,
    ) -> List[str]:
        retried: List[str] = []
        for row in self.repository.list_retry_candidates(run_id):
            outcome_class = str(row.get("outcome_class") or "")
            status = str(row.get("status") or "")
            if outcome_class == "needs_redo" and not include_needs_redo:
                continue
            if status == "failed" and not include_failed:
                continue
            detail = self.repository.get_derived_image(str(row["derived_image_id"]))
            attempts = list(detail.get("attempts", []) or [])
            last_attempt = attempts[-1] if attempts else {}
            execution_mode = "normal"
            trigger = "manual_retry"
            if outcome_class == "needs_redo" or str(row.get("normalization_status") or "") == "redo_pending":
                execution_mode = "no_agents"
                trigger = "manual_no_agents_retry"
            elif str(last_attempt.get("execution_mode") or "") == "no_agents":
                execution_mode = "recovery"
            self.retry_derived_image(str(row["derived_image_id"]), trigger=trigger, execution_mode=execution_mode)
            retried.append(str(row["derived_image_id"]))
        return retried

    def reprocess_normalization_for_derived_images(
        self,
        derived_image_ids: List[str],
        *,
        purge_existing: bool = True,
    ) -> Dict[str, Any]:
        summary = {"derived_images": 0, "accepted_reactions": 0, "rejected_reactions": 0}
        touched_runs: set[str] = set()
        touched_sources: set[str] = set()
        for derived_image_id in derived_image_ids:
            detail = self.repository.get_derived_image(derived_image_id)
            run = self.repository.get_run(str(detail["run_id"]))
            if run is None:
                continue
            config_snapshot = json.loads(run.get("config_snapshot_json") or "{}")
            store = create_artifact_store_from_config(config_snapshot)
            raw_key = str(detail.get("raw_artifact_key") or "")
            if not raw_key:
                continue
            payload = json.loads(store.get_bytes(raw_key).decode("utf-8"))
            attempt = self.repository.create_derived_image_attempt(
                derived_image_id=derived_image_id,
                trigger="manual_reprocess",
                execution_mode="recovery",
                status="completed",
                config_snapshot_json=_json_dumps(config_snapshot),
                retry_of_attempt_id=str(detail.get("last_attempt_id") or ""),
            )
            normalization = self._persist_reactions(
                store=store,
                run_id=str(detail["run_id"]),
                run_source_id=str(detail["run_source_id"]),
                derived_image_id=derived_image_id,
                attempt_id=str(attempt["attempt_id"]),
                payload=payload,
            )
            outcome_class = _classify_payload(payload, artifact_missing=(str(detail.get("artifact_status") or "") == "missing"))
            self.repository.finalize_derived_image_attempt(
                str(attempt["attempt_id"]),
                {"status": "completed", "raw_artifact_key": raw_key},
            )
            self.repository.finalize_derived_image_summary(
                derived_image_id,
                {
                    "status": "completed" if outcome_class != "failed" else "failed",
                    "status_message": "Normalization reprocessed.",
                    "outcome_class": outcome_class,
                    "raw_artifact_key": raw_key,
                    "error_text": str(payload.get("error") or ""),
                    "reaction_count": normalization.accepted_reaction_count,
                    "accepted_reaction_count": normalization.accepted_reaction_count,
                    "rejected_reaction_count": normalization.rejected_reaction_count,
                    "normalization_status": normalization.normalization_status,
                    "normalization_summary": normalization.normalization_summary,
                    "redo_count": 1 if outcome_class == "needs_redo" else 0,
                    "attempt_count": int(attempt["attempt_no"]),
                    "last_attempt_id": str(attempt["attempt_id"]),
                    "last_retry_reason": "manual_reprocess",
                },
            )
            touched_runs.add(str(detail["run_id"]))
            touched_sources.add(str(detail["run_source_id"]))
            summary["derived_images"] += 1
            summary["accepted_reactions"] += normalization.accepted_reaction_count
            summary["rejected_reactions"] += normalization.rejected_reaction_count
        for run_source_id in touched_sources:
            detail = self.repository.get_run_source_detail(run_source_id)
            self._refresh_source_and_run_counts(run_source_id, str(detail["run_id"]))
        return summary

    def reprocess_normalization_for_run(
        self,
        run_id: str,
        *,
        only_invalid_reactions: bool = True,
    ) -> Dict[str, Any]:
        candidates = []
        for row in self.repository.list_run_sources(run_id):
            detail = self.repository.get_run_source_detail(str(row["run_source_id"]))
            for derived in detail.get("derived_images", []):
                if only_invalid_reactions and int(derived.get("accepted_reaction_count", derived.get("reaction_count", 0)) or 0) > 0:
                    continue
                if not str(derived.get("raw_artifact_key") or ""):
                    continue
                candidates.append(str(derived["derived_image_id"]))
        return self.reprocess_normalization_for_derived_images(candidates, purge_existing=True)

    def run_dataset_maintenance(self, run_id: str = "") -> Dict[str, Any]:
        target_runs = [run_id] if run_id else [str(row["run_id"]) for row in self.repository.list_runs()]
        summary = {"runs": 0, "seeded_attempts": 0, "reprocessed_derived_images": 0, "accepted_reactions": 0}
        for current_run_id in target_runs:
            if not current_run_id:
                continue
            for source in self.repository.list_run_sources(current_run_id):
                detail = self.repository.get_run_source_detail(str(source["run_source_id"]))
                for derived in detail.get("derived_images", []):
                    attempts = derived.get("attempts", [])
                    if not attempts:
                        attempt = self.repository.create_derived_image_attempt(
                            derived_image_id=str(derived["derived_image_id"]),
                            trigger="initial",
                            execution_mode="normal",
                            status=str(derived.get("status") or "completed"),
                            config_snapshot_json="{}",
                        )
                        self.repository.finalize_derived_image_attempt(
                            str(attempt["attempt_id"]),
                            {
                                "status": str(derived.get("status") or "completed"),
                                "error_summary": str(derived.get("error_text") or ""),
                                "raw_artifact_key": str(derived.get("raw_artifact_key") or ""),
                            },
                        )
                        summary["seeded_attempts"] += 1
            run_summary = self.reprocess_normalization_for_run(current_run_id, only_invalid_reactions=False)
            summary["runs"] += 1
            summary["reprocessed_derived_images"] += int(run_summary.get("derived_images", 0))
            summary["accepted_reactions"] += int(run_summary.get("accepted_reactions", 0))
        return summary

    def get_run_monitor(
        self,
        run_id: str,
        *,
        tail_lines: int = 200,
        min_level: str = "INFO",
        raw: bool = False,
        include_suppressed: bool = False,
    ) -> Dict[str, Any]:
        run = self.repository.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        config = json.loads(run.get("config_snapshot_json") or "{}")
        log_tail = read_log_tail(
            run_id=run_id,
            config=config,
            log_artifact_key=str(run.get("log_artifact_key") or ""),
            tail_lines=tail_lines,
            min_level=min_level,
            raw=raw,
            include_suppressed=include_suppressed,
        )
        sources = self.repository.list_run_sources(run_id)
        progress = _run_progress_snapshot(
            run,
            has_logs=bool(log_tail.get("events") or run.get("log_artifact_key") or run.get("stdout_artifact_key") or run.get("stderr_artifact_key")),
        )
        return {
            "run": run,
            "progress": progress,
            "sources": sources,
            "log_tail": log_tail,
        }

    def get_run_log_tail(
        self,
        run_id: str,
        *,
        tail_lines: int = 200,
        min_level: str = "INFO",
        raw: bool = False,
        include_suppressed: bool = False,
    ) -> Dict[str, Any]:
        return self.get_run_monitor(
            run_id,
            tail_lines=tail_lines,
            min_level=min_level,
            raw=raw,
            include_suppressed=include_suppressed,
        )["log_tail"]

    def get_log_download_ref(self, run_id: str) -> str:
        run = self.repository.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        config = json.loads(run.get("config_snapshot_json") or "{}")
        return get_log_download_ref(
            run_id=run_id,
            config=config,
            log_artifact_key=str(run.get("log_artifact_key") or ""),
        )

    def update_reaction_review(self, reaction_uid: str, *, review_status: str, review_notes: str) -> None:
        self.repository.update_reaction_review(reaction_uid, review_status=review_status, review_notes=review_notes)

    def export_run_to_parquet(self, run_id: str, output_dir: str) -> Dict[str, str]:
        return self.repository.export_run_to_parquet(run_id, Path(output_dir))

    def _refresh_source_and_run_counts(self, run_source_id: str, run_id: str) -> None:
        source_detail = self.repository.get_run_source_detail(run_source_id)
        derived_images = source_detail.get("derived_images", [])
        source_summary = {
            "status": "completed",
            "status_message": "Source summary refreshed.",
            "expected_derived_images": len(derived_images),
            "completed_derived_images": len(derived_images),
            "successful_derived_images": sum(1 for row in derived_images if str(row.get("status") or "") != "failed"),
            "failed_derived_images": sum(1 for row in derived_images if str(row.get("status") or "") == "failed"),
            "reaction_count": sum(int(row.get("accepted_reaction_count", row.get("reaction_count", 0)) or 0) for row in derived_images),
            "redo_count": sum(1 for row in derived_images if str(row.get("outcome_class") or "") == "needs_redo"),
            "error_summary": next((str(row.get("error_text") or "") for row in reversed(derived_images) if row.get("error_text")), ""),
        }
        source_summary["status"] = _source_summary_status(source_summary)
        self.repository.finalize_run_source_summary(run_source_id, source_summary)

        run_sources = self.repository.list_run_sources(run_id)
        run_summary = _initial_summary(total_sources=len(run_sources))
        completed_sources = 0
        failed_sources = 0
        with self.repository.connect() as conn:
            usage_row = conn.execute(
                """
                SELECT COALESCE(SUM(usage_prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(usage_completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(usage_total_tokens), 0) AS total_tokens,
                       SUM(estimated_cost_usd) AS estimated_cost_usd
                FROM llm_call_metrics
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        run_summary["prompt_tokens"] = int(usage_row["prompt_tokens"] or 0)
        run_summary["completion_tokens"] = int(usage_row["completion_tokens"] or 0)
        run_summary["total_tokens"] = int(usage_row["total_tokens"] or 0)
        run_summary["estimated_cost_usd"] = usage_row["estimated_cost_usd"]
        for row in run_sources:
            if str(row.get("status") or "") == "failed":
                failed_sources += 1
            else:
                completed_sources += 1
            run_summary["total_derived_images"] += int(row.get("expected_derived_images", 0) or 0)
            run_summary["total_reactions"] += int(row.get("reaction_count", 0) or 0)
            run_summary["total_failures"] += int(row.get("failed_derived_images", 0) or 0)
            run_summary["total_redo"] += int(row.get("redo_count", 0) or 0)
        self.repository.update_run_summary(run_id, run_summary)
        self.repository.update_run_live_state(run_id, completed_sources=completed_sources, failed_sources=failed_sources)

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                self._run_task(task)
            except Exception as exc:  # pragma: no cover - defensive worker guard
                print(
                    f"[review_service] Unhandled run failure for {task.run_id}: {exc}\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                self.repository.update_run_status(task.run_id, "failed", finished=True)
                self._refresh_experiment_status(task.experiment_id)
            finally:
                self._queue.task_done()

    def _refresh_experiment_status(self, experiment_id: str) -> None:
        if not experiment_id:
            return
        runs = self.repository.list_runs(experiment_id=experiment_id)
        statuses = {run["status"] for run in runs}
        if "running" in statuses:
            status = "running"
        elif "queued" in statuses:
            status = "queued"
        elif "interrupted" in statuses and statuses.issubset({"completed", "interrupted"}):
            status = "interrupted"
        elif statuses == {"completed"}:
            status = "completed"
        elif "failed" in statuses or "interrupted" in statuses:
            status = "failed"
        else:
            status = "completed"
        self.repository.update_experiment_status(experiment_id, status)

    def _run_task(self, task: RunTask) -> None:
        store = create_artifact_store_from_config(task.config_snapshot)
        log_session = RunLogSession(run_id=task.run_id, experiment_id=task.experiment_id)
        run_logger = log_session.logger.bind(phase="run")
        self.repository.update_experiment_status(task.experiment_id, "running")
        self.repository.update_run_status(task.run_id, "running", started=True)
        self.repository.update_run_live_state(task.run_id, status_message="Run started.", current_phase="run_started")
        started = time.perf_counter()
        summary = _initial_summary(total_sources=len(task.live_source_paths) if task.ingest_mode == "live_batch" else 0)
        status = "completed"
        try:
            with bind_log_context(run_id=task.run_id, experiment_id=task.experiment_id):
                with log_session.capture_streams(run_logger):
                    run_logger.info("run_started", f"Run {task.run_id} started.")
                    if task.ingest_mode == "live_batch":
                        summary = self._execute_live_run(task, store, run_logger)
                    else:
                        summary = self._execute_sideload_run(task, store, run_logger)
                    run_logger.info("run_completed", f"Run {task.run_id} completed.")
        except RunAbortedError as exc:
            status = "failed"
            controller_state = exc.controller_state
            message = str(exc)
            run_row = self.repository.get_run(task.run_id) or {}
            summary = {
                **summary,
                "total_sources": int(run_row.get("total_sources", summary.get("total_sources", 0))),
                "total_derived_images": int(run_row.get("total_derived_images", summary.get("total_derived_images", 0))),
                "total_reactions": int(run_row.get("total_reactions", summary.get("total_reactions", 0))),
                "total_failures": int(run_row.get("total_failures", summary.get("total_failures", 0))),
                "total_redo": int(run_row.get("total_redo", summary.get("total_redo", 0))),
                "prompt_tokens": int(run_row.get("prompt_tokens", summary.get("prompt_tokens", 0))),
                "completion_tokens": int(run_row.get("completion_tokens", summary.get("completion_tokens", 0))),
                "total_tokens": int(run_row.get("total_tokens", summary.get("total_tokens", 0))),
                "estimated_cost_usd": run_row.get("estimated_cost_usd", summary.get("estimated_cost_usd")),
                "usage_completeness": str(run_row.get("usage_completeness", summary.get("usage_completeness", "none"))),
            }
            self.repository.update_run_abort(
                task.run_id,
                abort_reason=message,
                failure_summary=message,
                systemic_failure_kind=controller_state.systemic_failure_kind,
                systemic_failure_count=controller_state.systemic_failure_count,
            )
            run_logger.error("run_aborted", message)
        except Exception as exc:  # pragma: no cover - defensive worker guard
            status = "failed"
            failure_summary = f"{exc}\n{traceback.format_exc()}"
            self.repository.update_run_abort(task.run_id, abort_reason=str(exc), failure_summary=failure_summary)
            run_logger.exception("run_failed", f"Unhandled run failure for {task.run_id}: {exc}")
        finally:
            summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.repository.update_run_summary(task.run_id, summary)
            self.repository.update_run_status(task.run_id, status, finished=True)
            log_keys = log_session.finalize(store)
            self.repository.update_run_live_state(
                task.run_id,
                **log_keys,
                current_phase="completed" if status == "completed" else "failed",
                status_message="Run completed." if status == "completed" else "Run failed.",
            )
            log_session.close()
            self._refresh_experiment_status(task.experiment_id)

    def _execute_live_run(
        self,
        task: RunTask,
        store: ArtifactStore,
        run_logger,
    ) -> Dict[str, Any]:
        summary = _initial_summary(total_sources=len(task.live_source_paths))
        completed_sources = 0
        failed_sources = 0
        controller = RunFailureController()
        queued_sources = self._prequeue_live_sources(task=task, store=store)
        self.repository.update_run_live_state(task.run_id, total_sources=len(queued_sources), status_message="Queued sources prepared.")
        for item in queued_sources:
            path = item["path"]
            suffix = path.suffix.lower()
            index = int(item["input_order"])
            source_logger = run_logger.bind(phase="prepare_source", source_name=path.name)
            source_logger.info("source_started", f"Starting source {index + 1}/{len(task.live_source_paths)}: {path.name}")
            try:
                if suffix in IMAGE_SUFFIXES:
                    outcome = self._process_live_image(
                        task,
                        store,
                        path,
                        queued_source=item,
                        controller=controller,
                        run_logger=run_logger,
                    )
                elif suffix in PDF_SUFFIXES:
                    outcome = self._process_live_pdf(
                        task,
                        store,
                        path,
                        queued_source=item,
                        controller=controller,
                        run_logger=run_logger,
                    )
                else:
                    failed_sources += 1
                    run_logger.warning("source_skipped", f"Skipping unsupported source type: {path.name}")
                    self.repository.update_run_source_status(item["run_source_id"], status="skipped", status_message="Unsupported source type.", finished=True)
                    continue
            except RunAbortedError as exc:
                _merge_summary(summary, exc.summary_delta)
                failed_sources += 1
                self.repository.update_run_summary(task.run_id, summary)
                self.repository.update_run_live_state(
                    task.run_id,
                    completed_sources=completed_sources,
                    failed_sources=failed_sources,
                    current_phase="run_aborted",
                    status_message=str(exc),
                )
                raise
            _merge_summary(summary, outcome)
            if outcome["source_status"] == "completed":
                completed_sources += 1
            else:
                failed_sources += 1
            self.repository.update_run_summary(task.run_id, summary)
            self.repository.update_run_live_state(
                task.run_id,
                completed_sources=completed_sources,
                failed_sources=failed_sources,
                current_phase="source_completed",
                status_message=f"Finished source {path.name}",
            )
        return summary

    def _process_live_image(
        self,
        task: RunTask,
        store: ArtifactStore,
        image_path: Path,
        *,
        queued_source: Dict[str, Any],
        controller: RunFailureController,
        run_logger,
    ) -> Dict[str, Any]:
        run_source_id, source_logger = self._prepare_live_source(
            task=task,
            store=store,
            source_path=image_path,
            queued_source=queued_source,
            run_logger=run_logger,
        )
        source_summary = {
            "status": "processing",
            "status_message": f"Processing image {image_path.name}",
            "expected_derived_images": 1,
            "completed_derived_images": 0,
            "successful_derived_images": 0,
            "failed_derived_images": 0,
            "reaction_count": 0,
            "redo_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
            "usage_completeness": "none",
            "error_summary": "",
        }
        derived_image_id = self.repository.create_derived_image(
            run_source_id=run_source_id,
            page_hint=image_path.name,
            image_index=0,
            artifact_backend=store.backend_name,
            artifact_key=str(queued_source["source_key"]),
            artifact_status="present",
            outcome_class="queued",
            raw_artifact_key="",
        )
        try:
                delta = self._process_derived_image(
                    task=task,
                    store=store,
                    run_source_id=run_source_id,
                    derived_image_id=derived_image_id,
                    image_path=image_path,
                    derived_artifact_key=str(queued_source["source_key"]),
                    image_index=0,
                    controller=controller,
                    source_index=int(queued_source["input_order"]),
                    source_name=image_path.name,
                    run_logger=source_logger,
            )
        except RunAbortedError as exc:
            source_summary["status"] = "aborted"
            source_summary["completed_derived_images"] = 1
            source_summary["failed_derived_images"] = 1
            source_summary["error_summary"] = str(exc)
            source_summary["status_message"] = str(exc)
            self.repository.finalize_run_source_summary(run_source_id, source_summary)
            raise RunAbortedError(
                str(exc),
                controller_state=exc.controller_state,
                summary_delta={
                    "derived_count": 1,
                    "reaction_count": 0,
                    "failure_count": 1,
                    "redo_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": None,
                    "usage_completeness": "none",
                },
            ) from exc
        source_summary["completed_derived_images"] = 1
        source_summary["successful_derived_images"] = 0 if delta["failure_count"] else 1
        source_summary["failed_derived_images"] = delta["failure_count"]
        source_summary["reaction_count"] = delta["reaction_count"]
        source_summary["redo_count"] = delta["redo_count"]
        source_summary["prompt_tokens"] = delta["prompt_tokens"]
        source_summary["completion_tokens"] = delta["completion_tokens"]
        source_summary["total_tokens"] = delta["total_tokens"]
        source_summary["estimated_cost_usd"] = delta["estimated_cost_usd"]
        source_summary["usage_completeness"] = delta["usage_completeness"]
        source_summary["error_summary"] = delta.get("error_summary", "")
        source_summary["status"] = _source_summary_status(source_summary)
        source_summary["status_message"] = f"Image source {image_path.name} finished with {source_summary['status']}."
        self.repository.finalize_run_source_summary(run_source_id, source_summary)
        source_logger.info("source_completed", source_summary["status_message"], source_status=source_summary["status"])
        return {
            "source_status": source_summary["status"],
            "derived_count": 1,
            "reaction_count": delta["reaction_count"],
            "failure_count": delta["failure_count"],
            "redo_count": delta["redo_count"],
            "prompt_tokens": delta["prompt_tokens"],
            "completion_tokens": delta["completion_tokens"],
            "total_tokens": delta["total_tokens"],
            "estimated_cost_usd": delta["estimated_cost_usd"],
            "usage_completeness": delta["usage_completeness"],
        }

    def _process_live_pdf(
        self,
        task: RunTask,
        store: ArtifactStore,
        pdf_path: Path,
        *,
        queued_source: Dict[str, Any],
        controller: RunFailureController,
        run_logger,
    ) -> Dict[str, Any]:
        from pdf_extraction import run_pdf

        run_source_id, source_logger = self._prepare_live_source(
            task=task,
            store=store,
            source_path=pdf_path,
            queued_source=queued_source,
            run_logger=run_logger,
        )
        source_summary = {
            "status": "extracting",
            "status_message": f"Extracting PDF crops for {pdf_path.name}",
            "expected_derived_images": 0,
            "completed_derived_images": 0,
            "successful_derived_images": 0,
            "failed_derived_images": 0,
            "reaction_count": 0,
            "redo_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
            "usage_completeness": "none",
            "error_summary": "",
        }
        self.repository.update_run_source_status(
            run_source_id,
            status="extracting",
            current_phase="extract_pdf_images",
            status_message=source_summary["status_message"],
        )
        self.repository.update_run_live_state(
            task.run_id,
            current_phase="extract_pdf_images",
            current_run_source_id=run_source_id,
            current_source_name=pdf_path.name,
            status_message=source_summary["status_message"],
        )
        with tempfile.TemporaryDirectory(prefix="review_pdf_") as tmpdir:
            source_logger.info("pdf_extraction_started", f"Extracting crops from {pdf_path.name}")
            try:
                with bind_log_context(run_source_id=run_source_id, phase="extract_pdf_images"):
                    run_pdf(
                        pdf_dir=str(pdf_path),
                        image_dir=tmpdir,
                        model_size=str(task.config_snapshot.get("pdf_model_size") or task.config_snapshot.get("PDF_MODEL_SIZE") or "large"),
                    )
            except Exception as exc:
                source_summary["status"] = "failed"
                source_summary["status_message"] = f"PDF extraction failed for {pdf_path.name}: {exc}"
                source_summary["error_summary"] = str(exc)
                self.repository.finalize_run_source_summary(run_source_id, source_summary)
                source_logger.error("source_failed", source_summary["status_message"])
                return {
                    "source_status": "failed",
                    "derived_count": 0,
                    "reaction_count": 0,
                    "failure_count": 1,
                    "redo_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": None,
                    "usage_completeness": "none",
                }
            extracted_images = sorted(Path(tmpdir).glob("*.png"))
            source_summary["expected_derived_images"] = len(extracted_images)
            self.repository.update_run_source_progress(
                run_source_id,
                status="processing",
                current_phase="process_image",
                expected_derived_images=len(extracted_images),
                status_message=f"Discovered {len(extracted_images)} extracted image(s) for {pdf_path.name}",
            )
            source_logger.info(
                "pdf_extraction_finished",
                f"Extracted {len(extracted_images)} crop(s) from {pdf_path.name}",
                expected_derived_images=len(extracted_images),
            )
            queued_images = []
            for image_index, image_file in enumerate(extracted_images):
                derived_key = f"derived/{run_source_id}/{image_index}.png"
                store.put_file(derived_key, str(image_file), content_type="image/png")
                derived_image_id = self.repository.create_derived_image(
                    run_source_id=run_source_id,
                    page_hint=image_file.name,
                    image_index=image_index,
                    artifact_backend=store.backend_name,
                    artifact_key=derived_key,
                    artifact_status="present",
                    outcome_class="queued",
                    raw_artifact_key="",
                )
                source_logger.info(
                    "derived_image_queued",
                    f"Queued derived image {image_index + 1}/{len(extracted_images)} for {pdf_path.name}",
                    derived_image_id=derived_image_id,
                )
                queued_images.append((image_index, image_file, derived_image_id, derived_key))

            for image_index, image_file, derived_image_id, derived_key in queued_images:
                try:
                    delta = self._process_derived_image(
                        task=task,
                        store=store,
                        run_source_id=run_source_id,
                        derived_image_id=derived_image_id,
                        image_path=image_file,
                        derived_artifact_key=derived_key,
                        image_index=image_index,
                        controller=controller,
                        source_index=int(queued_source["input_order"]),
                        source_name=pdf_path.name,
                        run_logger=source_logger,
                    )
                except RunAbortedError as exc:
                    source_summary["status"] = "aborted"
                    source_summary["completed_derived_images"] += 1
                    source_summary["failed_derived_images"] += 1
                    source_summary["error_summary"] = str(exc)
                    source_summary["status_message"] = str(exc)
                    self.repository.finalize_run_source_summary(run_source_id, source_summary)
                    raise RunAbortedError(
                        str(exc),
                        controller_state=exc.controller_state,
                        summary_delta={
                            "derived_count": source_summary["expected_derived_images"],
                            "reaction_count": source_summary["reaction_count"],
                            "failure_count": source_summary["failed_derived_images"],
                            "redo_count": source_summary["redo_count"],
                            "prompt_tokens": source_summary["prompt_tokens"],
                            "completion_tokens": source_summary["completion_tokens"],
                            "total_tokens": source_summary["total_tokens"],
                            "estimated_cost_usd": source_summary["estimated_cost_usd"],
                            "usage_completeness": source_summary["usage_completeness"],
                        },
                    ) from exc
                source_summary["completed_derived_images"] += 1
                source_summary["successful_derived_images"] += 0 if delta["failure_count"] else 1
                source_summary["failed_derived_images"] += delta["failure_count"]
                source_summary["reaction_count"] += delta["reaction_count"]
                source_summary["redo_count"] += delta["redo_count"]
                source_summary["prompt_tokens"] += delta["prompt_tokens"]
                source_summary["completion_tokens"] += delta["completion_tokens"]
                source_summary["total_tokens"] += delta["total_tokens"]
                source_summary["usage_completeness"] = _merge_usage_completeness(
                    source_summary["usage_completeness"],
                    delta["usage_completeness"],
                )
                if delta["estimated_cost_usd"] is not None:
                    source_summary["estimated_cost_usd"] = round(
                        float(source_summary.get("estimated_cost_usd") or 0.0) + float(delta["estimated_cost_usd"]),
                        8,
                    )
                if delta.get("error_summary"):
                    source_summary["error_summary"] = str(delta["error_summary"])
                self.repository.update_run_source_progress(
                    run_source_id,
                    completed_derived_images=source_summary["completed_derived_images"],
                    successful_derived_images=source_summary["successful_derived_images"],
                    failed_derived_images=source_summary["failed_derived_images"],
                    reaction_count=source_summary["reaction_count"],
                    redo_count=source_summary["redo_count"],
                    prompt_tokens=source_summary["prompt_tokens"],
                    completion_tokens=source_summary["completion_tokens"],
                    total_tokens=source_summary["total_tokens"],
                    estimated_cost_usd=source_summary["estimated_cost_usd"],
                    usage_completeness=source_summary["usage_completeness"],
                    error_summary=source_summary["error_summary"],
                    status_message=f"Processed {source_summary['completed_derived_images']}/{source_summary['expected_derived_images']} crop(s) for {pdf_path.name}",
                )
        source_summary["status"] = _source_summary_status(source_summary)
        source_summary["status_message"] = f"PDF source {pdf_path.name} finished with {source_summary['status']}."
        self.repository.finalize_run_source_summary(run_source_id, source_summary)
        source_logger.info("source_completed", source_summary["status_message"], source_status=source_summary["status"])
        return {
            "source_status": source_summary["status"],
            "derived_count": source_summary["expected_derived_images"],
            "reaction_count": source_summary["reaction_count"],
            "failure_count": source_summary["failed_derived_images"],
            "redo_count": source_summary["redo_count"],
            "prompt_tokens": source_summary["prompt_tokens"],
            "completion_tokens": source_summary["completion_tokens"],
            "total_tokens": source_summary["total_tokens"],
            "estimated_cost_usd": source_summary["estimated_cost_usd"],
            "usage_completeness": source_summary["usage_completeness"],
        }

    def _execute_sideload_run(self, task: RunTask, store: ArtifactStore, run_logger) -> Dict[str, Any]:
        json_path = Path(task.sideload_json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        raw_run_source_id = _hash_text(str(json_path.resolve()))
        raw_key = f"raw/{task.run_id}/{raw_run_source_id}/pipeline.json"
        _store_json(store, raw_key, payload)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            image_name = str(item.get("image_name") or "")
            if "_image_" in image_name:
                group_key = image_name.split("_image_")[0]
            else:
                group_key = image_name or f"item-{index}"
            groups.setdefault(group_key, []).append(item)

        summary = _initial_summary(total_sources=len(groups))
        completed_sources = 0
        failed_sources = 0
        for input_order, (group_key, group_items) in enumerate(sorted(groups.items())):
            first_image_name = str(group_items[0].get("image_name") or "")
            inferred_pdf = "_image_" in first_image_name
            source_type = "pdf" if inferred_pdf else ("image" if first_image_name else "synthetic_missing")
            source_filename = f"{group_key}.pdf" if inferred_pdf else (first_image_name or f"{group_key}.json")
            source_path = _recover_file(
                payload={"execution_logs": [], "plan": []},
                file_name=source_filename,
                json_path=json_path,
                recovery_roots=task.recovery_roots,
            )
            artifact_status = "present" if source_path else "missing"
            source_asset_id = _hash_text(f"{json_path.resolve()}::{group_key}")
            source_key = ""
            if source_path is not None:
                source_key = f"sources/{source_asset_id}/original{source_path.suffix.lower()}"
                if not store.exists(source_key):
                    store.put_file(source_key, str(source_path))
            self.repository.upsert_source_asset(
                source_asset_id=source_asset_id,
                source_type=source_type if artifact_status == "present" else "synthetic_missing" if source_type == "synthetic_missing" else source_type,
                sha256=_hash_file(source_path) if source_path and source_path.is_file() else None,
                original_filename=source_filename,
                artifact_backend=store.backend_name if source_key else "",
                artifact_key=source_key,
                artifact_status="recovered" if source_path and source_type == "pdf" and not source_key else artifact_status,
            )
            run_source_id = self.repository.create_run_source(
                run_id=task.run_id,
                source_asset_id=source_asset_id,
                input_order=input_order,
                recovery_note=str(json_path),
                source_type=source_type,
            )
            self.repository.update_run_source_status(
                run_source_id,
                status="processing",
                current_phase="normalize",
                status_message=f"Importing sideload source {source_filename}",
                started=True,
            )
            self.repository.update_run_live_state(
                task.run_id,
                current_phase="normalize",
                current_run_source_id=run_source_id,
                current_source_name=source_filename,
                status_message=f"Importing sideload source {source_filename}",
            )
            run_logger.info("source_started", f"Importing sideload source {source_filename}", run_source_id=run_source_id)
            source_reactions = 0
            source_failures = 0
            source_redo = 0
            for image_index, item in enumerate(group_items):
                image_name = str(item.get("image_name") or f"{group_key}_{image_index}.png")
                recovered_image = _recover_file(
                    payload=item,
                    file_name=image_name,
                    json_path=json_path,
                    recovery_roots=task.recovery_roots,
                )
                derived_key = ""
                artifact_status = "missing"
                if recovered_image is not None:
                    derived_key = f"derived/{run_source_id}/{image_index}{recovered_image.suffix.lower() or '.png'}"
                    if not store.exists(derived_key):
                        store.put_file(derived_key, str(recovered_image))
                    artifact_status = "recovered"
                derived_image_id = self.repository.create_derived_image(
                    run_source_id=run_source_id,
                    page_hint=image_name,
                    image_index=image_index,
                    artifact_backend=store.backend_name if derived_key else "",
                    artifact_key=derived_key,
                    artifact_status=artifact_status,
                    status="completed",
                    outcome_class="queued",
                    raw_artifact_key="",
                )
                raw_item_key = f"raw/{task.run_id}/{run_source_id}/image_{image_index}.json"
                persisted_item = _rewrite_payload_image_refs(
                    item,
                    store=store,
                    artifact_key=derived_key,
                    artifact_backend=store.backend_name if derived_key else "",
                )
                _store_json(store, raw_item_key, persisted_item)
                attempt = self.repository.create_derived_image_attempt(
                    derived_image_id=derived_image_id,
                    trigger="initial",
                    execution_mode="normal",
                    status="completed",
                    config_snapshot_json=_json_dumps(task.config_snapshot),
                )
                self.repository.finalize_derived_image_attempt(
                    str(attempt["attempt_id"]),
                    {
                        "status": "completed",
                        "raw_artifact_key": raw_item_key,
                    },
                )
                outcome_class = _classify_payload(item, artifact_missing=(artifact_status == "missing"))
                normalization = self._persist_reactions(
                    store=store,
                    run_id=task.run_id,
                    run_source_id=run_source_id,
                    derived_image_id=derived_image_id,
                    attempt_id=str(attempt["attempt_id"]),
                    payload=item,
                    override_outcome="imported_without_artifact" if artifact_status == "missing" else "",
                )
                self.repository.finalize_derived_image_summary(
                    derived_image_id,
                    {
                        "status": "completed",
                        "status_message": f"Sideload item {image_name} imported.",
                        "outcome_class": outcome_class,
                        "raw_artifact_key": raw_item_key,
                        "error_text": str(item.get("error") or ""),
                        "reaction_count": normalization.accepted_reaction_count,
                        "accepted_reaction_count": normalization.accepted_reaction_count,
                        "rejected_reaction_count": normalization.rejected_reaction_count,
                        "normalization_status": normalization.normalization_status,
                        "normalization_summary": normalization.normalization_summary,
                        "redo_count": 1 if outcome_class == "needs_redo" else 0,
                        "attempt_count": int(attempt["attempt_no"]),
                        "last_attempt_id": str(attempt["attempt_id"]),
                        "last_retry_reason": "initial",
                    },
                )
                summary["total_derived_images"] += 1
                summary["total_reactions"] += normalization.accepted_reaction_count
                summary["total_failures"] += 1 if outcome_class == "failed" else 0
                summary["total_redo"] += 1 if outcome_class == "needs_redo" else 0
                source_reactions += normalization.accepted_reaction_count
                source_failures += 1 if outcome_class == "failed" else 0
                source_redo += 1 if outcome_class == "needs_redo" else 0
            self.repository.finalize_run_source_summary(
                run_source_id,
                {
                    "status": "failed" if source_failures and source_reactions == 0 else "completed",
                    "status_message": f"Sideload source {source_filename} imported.",
                    "expected_derived_images": len(group_items),
                    "completed_derived_images": len(group_items),
                    "successful_derived_images": len(group_items) - source_failures,
                    "failed_derived_images": source_failures,
                    "reaction_count": source_reactions,
                    "redo_count": source_redo,
                    "error_summary": "" if not source_failures else f"{source_failures} sideload item(s) failed.",
                },
            )
            if source_failures and source_reactions == 0:
                failed_sources += 1
            else:
                completed_sources += 1
            self.repository.update_run_summary(task.run_id, summary)
            self.repository.update_run_live_state(
                task.run_id,
                completed_sources=completed_sources,
                failed_sources=failed_sources,
                current_phase="source_completed",
                status_message=f"Finished sideload source {source_filename}",
            )
        return summary

    def _prepare_live_source(
        self,
        *,
        task: RunTask,
        store: ArtifactStore,
        queued_source: Dict[str, Any],
        source_path: Path,
        run_logger,
    ) -> tuple[str, Any]:
        run_source_id = str(queued_source["run_source_id"])
        self.repository.update_run_source_status(
            run_source_id,
            status="processing",
            current_phase="prepare_source",
            status_message=f"Preparing source {source_path.name}",
            started=True,
        )
        self.repository.update_run_live_state(
            task.run_id,
            current_phase="prepare_source",
            current_run_source_id=run_source_id,
            current_source_name=source_path.name,
            status_message=f"Preparing source {source_path.name}",
        )
        return run_source_id, run_logger.bind(run_source_id=run_source_id, source_name=source_path.name)

    def _prequeue_live_sources(self, *, task: RunTask, store: ArtifactStore) -> List[Dict[str, Any]]:
        queued_sources: List[Dict[str, Any]] = []
        for input_order, path_str in enumerate(task.live_source_paths):
            path = Path(path_str)
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                source_type = "image"
                content_type = None
            elif suffix in PDF_SUFFIXES:
                source_type = "pdf"
                content_type = "application/pdf"
            else:
                source_type = "unknown"
                content_type = None
            source_sha = _hash_file(path)
            source_key = f"sources/{source_sha}/original{path.suffix.lower()}"
            if source_type != "unknown" and not store.exists(source_key):
                store.put_file(source_key, str(path), content_type=content_type)
            source_asset_id = self.repository.upsert_source_asset(
                source_asset_id=source_sha,
                source_type=source_type if source_type != "unknown" else "synthetic_missing",
                sha256=source_sha,
                original_filename=path.name,
                artifact_backend=store.backend_name if source_type != "unknown" else "",
                artifact_key=source_key if source_type != "unknown" else "",
                artifact_status="present" if source_type != "unknown" else "missing",
            )
            run_source_id = self.repository.create_run_source(
                run_id=task.run_id,
                source_asset_id=source_asset_id,
                input_order=input_order,
                source_type=source_type,
            )
            self.repository.update_run_source_status(
                run_source_id,
                status="queued",
                current_phase="prepare_source",
                status_message="Queued for processing.",
            )
            queued_sources.append(
                {
                    "path": path,
                    "input_order": input_order,
                    "source_type": source_type,
                    "run_source_id": run_source_id,
                    "source_asset_id": source_asset_id,
                    "source_key": source_key,
                }
            )
        return queued_sources

    def _process_derived_image(
        self,
        *,
        task: RunTask,
        store: ArtifactStore,
        run_source_id: str,
        derived_image_id: str,
        image_path: Path,
        derived_artifact_key: str,
        image_index: int,
        controller: RunFailureController,
        source_index: int,
        source_name: str,
        run_logger,
        trigger: str = "initial",
        execution_mode: str = "normal",
        max_attempts: int = MAX_AUTO_ATTEMPTS_PER_DERIVED_IMAGE,
    ) -> Dict[str, Any]:
        self.repository.update_derived_image_status(
            derived_image_id,
            status="processing",
            status_message=f"Processing derived image {image_index + 1}",
            started=True,
        )
        self.repository.update_run_live_state(
            task.run_id,
            current_phase="process_image",
            current_run_source_id=run_source_id,
            current_source_name=source_name,
            status_message=f"Processing crop {image_index + 1} for {source_name}",
        )
        derived_logger = run_logger.bind(run_source_id=run_source_id, derived_image_id=derived_image_id, phase="process_image")
        derived_logger.info("derived_image_started", f"Processing derived image {image_index + 1} for {source_name}")
        with bind_log_context(run_source_id=run_source_id, derived_image_id=derived_image_id, phase="process_image"):
            current_trigger = trigger
            current_execution_mode = execution_mode
            retry_of_attempt_id = ""
            for _ in range(max_attempts):
                attempt = self.repository.create_derived_image_attempt(
                    derived_image_id=derived_image_id,
                    trigger=current_trigger,
                    execution_mode=current_execution_mode,
                    status="running",
                    config_snapshot_json=_json_dumps(task.config_snapshot),
                    retry_of_attempt_id=retry_of_attempt_id,
                )
                attempt_id = str(attempt["attempt_id"])
                attempt_no = int(attempt["attempt_no"])
                self.repository.update_derived_image_attempt(attempt_id, started=True)
                raw_key = ""
                try:
                    if current_execution_mode == "recovery":
                        payload, metrics = self._execute_image_pipeline_recovery_subprocess(str(image_path), task.config_snapshot)
                    else:
                        payload, metrics = self._execute_image_pipeline(
                            str(image_path),
                            task.config_snapshot,
                            execution_mode=current_execution_mode,
                        )
                    raw_key = f"raw/{task.run_id}/{run_source_id}/image_{image_index}_attempt_{attempt_no}.json"
                    persisted_payload = _rewrite_payload_image_refs(
                        payload,
                        store=store,
                        artifact_key=derived_artifact_key,
                        artifact_backend=store.backend_name,
                    )
                    _store_json(store, raw_key, persisted_payload)
                    if metrics.calls:
                        self.repository.add_llm_call_metrics(
                            run_id=task.run_id,
                            run_source_id=run_source_id,
                            derived_image_id=derived_image_id,
                            call_metrics=[call.to_record() for call in metrics.calls],
                        )
                    normalization = self._persist_reactions(
                        store=store,
                        run_id=task.run_id,
                        run_source_id=run_source_id,
                        derived_image_id=derived_image_id,
                        attempt_id=attempt_id,
                        payload=payload,
                    )
                    outcome_class = _classify_payload(payload)
                    metrics_summary = metrics.summary()
                    self.repository.finalize_derived_image_attempt(
                        attempt_id,
                        {
                            "status": "completed",
                            "raw_artifact_key": raw_key,
                            "error_summary": str(payload.get("error") or ""),
                            "prompt_tokens": metrics_summary["prompt_tokens"],
                            "completion_tokens": metrics_summary["completion_tokens"],
                            "total_tokens": metrics_summary["total_tokens"],
                            "estimated_cost_usd": metrics_summary["estimated_cost_usd"],
                            "usage_completeness": metrics_summary["usage_completeness"],
                        },
                    )
                    retry_redo, next_trigger = _should_retry_redo(
                        payload=payload,
                        normalization=normalization,
                        attempt_no=attempt_no,
                        current_execution_mode=current_execution_mode,
                    )
                    if retry_redo:
                        self.repository.update_derived_image_status(
                            derived_image_id,
                            status="processing",
                            status_message=f"Retrying derived image {image_index + 1} after redo.",
                            outcome_class=outcome_class,
                            raw_artifact_key=raw_key,
                            error_text=str(payload.get("error") or ""),
                            accepted_reaction_count=normalization.accepted_reaction_count,
                            rejected_reaction_count=normalization.rejected_reaction_count,
                            normalization_status=normalization.normalization_status,
                            normalization_summary=normalization.normalization_summary,
                            last_attempt_id=attempt_id,
                            last_retry_reason=next_trigger,
                            attempt_count=attempt_no,
                        )
                        derived_logger.warning(
                            "derived_image_retry_scheduled",
                            f"Retrying derived image {image_index + 1} for {source_name} after redo-only output",
                            attempt_no=attempt_no,
                            retry_trigger=next_trigger,
                        )
                        retry_of_attempt_id = attempt_id
                        current_trigger = next_trigger
                        current_execution_mode = _execution_mode_for_trigger(next_trigger, fallback=current_execution_mode)
                        time.sleep(_retry_backoff_seconds(attempt_no))
                        continue
                    self.repository.finalize_derived_image_summary(
                        derived_image_id,
                        {
                            "status": "completed" if outcome_class != "failed" else "failed",
                            "status_message": f"Derived image {image_index + 1} finished.",
                            "outcome_class": outcome_class,
                            "raw_artifact_key": raw_key,
                            "error_text": str(payload.get("error") or ""),
                            "reaction_count": normalization.accepted_reaction_count,
                            "accepted_reaction_count": normalization.accepted_reaction_count,
                            "rejected_reaction_count": normalization.rejected_reaction_count,
                            "normalization_status": normalization.normalization_status,
                            "normalization_summary": normalization.normalization_summary,
                            "redo_count": 1 if outcome_class == "needs_redo" else 0,
                            "prompt_tokens": metrics_summary["prompt_tokens"],
                            "completion_tokens": metrics_summary["completion_tokens"],
                            "total_tokens": metrics_summary["total_tokens"],
                            "estimated_cost_usd": metrics_summary["estimated_cost_usd"],
                            "attempt_count": attempt_no,
                            "last_attempt_id": attempt_id,
                            "last_retry_reason": current_trigger,
                        },
                    )
                    derived_logger.info(
                        "derived_image_completed",
                        f"Derived image {image_index + 1} finished for {source_name}",
                        outcome_class=outcome_class,
                        reaction_count=normalization.accepted_reaction_count,
                        rejected_reaction_count=normalization.rejected_reaction_count,
                        normalization_status=normalization.normalization_status,
                        attempt_no=attempt_no,
                        execution_mode=current_execution_mode,
                    )
                    return {
                        "derived_count": 1,
                        "reaction_count": normalization.accepted_reaction_count,
                        "failure_count": 1 if outcome_class == "failed" else 0,
                        "redo_count": 1 if outcome_class == "needs_redo" else 0,
                        "prompt_tokens": metrics_summary["prompt_tokens"],
                        "completion_tokens": metrics_summary["completion_tokens"],
                        "total_tokens": metrics_summary["total_tokens"],
                        "estimated_cost_usd": metrics_summary["estimated_cost_usd"],
                        "usage_completeness": metrics_summary["usage_completeness"],
                        "error_summary": str(payload.get("error") or ""),
                    }
                except Exception as exc:
                    failure = getattr(exc, "_provider_failure", None) or classify_provider_exception(exc)
                    diagnostics = _provider_diagnostics(exc)
                    collector = _exception_metrics(exc)
                    metrics_summary = collector.summary() if collector.calls else _empty_metrics_summary()
                    if collector.calls:
                        self.repository.add_llm_call_metrics(
                            run_id=task.run_id,
                            run_source_id=run_source_id,
                            derived_image_id=derived_image_id,
                            call_metrics=[call.to_record() for call in collector.calls],
                        )
                    error_summary = _diagnostic_error_summary(exc, diagnostics)
                    self.repository.finalize_derived_image_attempt(
                        attempt_id,
                        {
                            "status": "failed",
                            "failure_kind": failure.kind,
                            "error_summary": error_summary,
                            "raw_artifact_key": raw_key,
                            "prompt_tokens": metrics_summary["prompt_tokens"],
                            "completion_tokens": metrics_summary["completion_tokens"],
                            "total_tokens": metrics_summary["total_tokens"],
                            "estimated_cost_usd": metrics_summary["estimated_cost_usd"],
                            "usage_completeness": metrics_summary["usage_completeness"],
                        },
                    )
                    should_retry, next_trigger = _should_retry_failure(
                        failure,
                        exc,
                        attempt_no,
                        current_execution_mode=current_execution_mode,
                    )
                    if should_retry:
                        self.repository.update_derived_image_status(
                            derived_image_id,
                            status="processing",
                            status_message=f"Retrying derived image {image_index + 1} after failure.",
                            error_text=error_summary,
                            last_attempt_id=attempt_id,
                            last_retry_reason=next_trigger,
                            attempt_count=attempt_no,
                        )
                        derived_logger.warning(
                            "derived_image_retry_scheduled",
                            f"Retrying derived image {image_index + 1} for {source_name} after {exc.__class__.__name__}",
                            attempt_no=attempt_no,
                            retry_trigger=next_trigger,
                            failure_kind=failure.kind,
                        )
                        retry_of_attempt_id = attempt_id
                        current_trigger = next_trigger
                        current_execution_mode = _execution_mode_for_trigger(next_trigger, fallback=current_execution_mode)
                        time.sleep(_retry_backoff_seconds(attempt_no))
                        continue
                    aborted, abort_reason = controller.record(failure, source_index=source_index, source_name=source_name)
                    self.repository.finalize_derived_image_summary(
                        derived_image_id,
                        {
                            "status": "failed",
                            "status_message": f"Derived image {image_index + 1} failed.",
                            "outcome_class": "failed",
                            "raw_artifact_key": raw_key,
                            "error_text": error_summary,
                            "accepted_reaction_count": 0,
                            "rejected_reaction_count": 0,
                            "normalization_status": "redo_pending" if current_trigger == "auto_recovery_retry" else "none_found",
                            "normalization_summary": "",
                            "prompt_tokens": metrics_summary["prompt_tokens"],
                            "completion_tokens": metrics_summary["completion_tokens"],
                            "total_tokens": metrics_summary["total_tokens"],
                            "estimated_cost_usd": metrics_summary["estimated_cost_usd"],
                            "attempt_count": attempt_no,
                            "last_attempt_id": attempt_id,
                            "last_retry_reason": current_trigger,
                        },
                    )
                    self.repository.update_run_live_state(
                        task.run_id,
                        last_error_summary=error_summary,
                        systemic_failure_kind=failure.kind if failure.systemic else "",
                        systemic_failure_count=controller.state.systemic_failure_count if failure.systemic else 0,
                        status_message=f"Crop failure on {source_name}: {error_summary}",
                    )
                    derived_logger.error(
                        "derived_image_failed",
                        f"Derived image {image_index + 1} failed for {source_name}: {exc}",
                        failure_kind=failure.kind,
                        systemic=failure.systemic,
                        llm_stage=diagnostics.get("llm_stage", ""),
                        llm_phase=diagnostics.get("llm_phase", ""),
                        provider=diagnostics.get("provider", ""),
                        model=diagnostics.get("model", ""),
                        base_url=diagnostics.get("base_url", ""),
                        exception_class=diagnostics.get("exception_class", exc.__class__.__name__),
                        cause_class=diagnostics.get("cause_class", ""),
                        cause_message=diagnostics.get("cause_message", ""),
                        traceback=diagnostics.get("traceback", ""),
                    )
                    if aborted:
                        raise RunAbortedError(abort_reason, controller_state=controller.state) from exc
                    return {
                        "derived_count": 1,
                        "reaction_count": 0,
                        "failure_count": 1,
                        "redo_count": 0,
                        "prompt_tokens": metrics_summary["prompt_tokens"],
                        "completion_tokens": metrics_summary["completion_tokens"],
                        "total_tokens": metrics_summary["total_tokens"],
                        "estimated_cost_usd": metrics_summary["estimated_cost_usd"],
                        "usage_completeness": metrics_summary["usage_completeness"],
                        "error_summary": error_summary,
                    }
            return {
                "derived_count": 1,
                "reaction_count": 0,
                "failure_count": 1,
                "redo_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": None,
                "usage_completeness": "none",
                "error_summary": f"Retry cap exhausted for {source_name}",
            }

    def _execute_image_pipeline(
        self,
        image_path: str,
        config: Dict[str, Any],
        *,
        execution_mode: str = "normal",
    ) -> tuple[Dict[str, Any], RunMetricsCollector]:
        from main import ChemEagle, ChemEagle_OS

        _apply_runtime_env(config)
        os.environ["CHEMEAGLE_ALLOW_PARTIAL_PAYLOAD"] = "1"
        collector = RunMetricsCollector()
        try:
            with bind_metrics_collector(collector):
                mode = str(config.get("mode") or config.get("CHEMEAGLE_RUN_MODE") or "cloud")
                kwargs = {}
                if execution_mode == "no_agents":
                    kwargs = {"use_plan_observer": False, "use_action_observer": False}
                payload = ChemEagle(image_path, **kwargs) if mode == "cloud" else ChemEagle_OS(image_path, **kwargs)
        except Exception as exc:
            setattr(exc, "_metrics_collector", collector)
            raise
        return payload, collector

    def _execute_image_pipeline_recovery_subprocess(self, image_path: str, config: Dict[str, Any]) -> tuple[Dict[str, Any], RunMetricsCollector]:
        _apply_runtime_env(config)
        env = dict(os.environ)
        env["CHEMEAGLE_ALLOW_PARTIAL_PAYLOAD"] = "1"
        mode = str(config.get("mode") or config.get("CHEMEAGLE_RUN_MODE") or "cloud")
        script = (
            "import json, traceback\n"
            "from main import ChemEagle, ChemEagle_OS\n"
            f"mode = {mode!r}\n"
            f"image_path = {image_path!r}\n"
            "kwargs = {'use_plan_observer': False, 'use_action_observer': False}\n"
            "fn = ChemEagle if mode == 'cloud' else ChemEagle_OS\n"
            "try:\n"
            "    result = fn(image_path, **kwargs)\n"
            "    print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}, ensure_ascii=False))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parent),
            check=False,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"Recovery subprocess produced no output. stderr: {proc.stderr.strip()}")
        payload = json.loads(lines[-1])
        if not isinstance(payload, dict):
            raise RuntimeError("Recovery subprocess returned an invalid payload.")
        if payload.get("ok"):
            return payload["result"], RunMetricsCollector()
        partial_payload = {
            "partial": True,
            "error": payload.get("error") or "Recovery subprocess failed",
            "traceback": payload.get("traceback") or proc.stderr.strip(),
            "execution_logs": [],
            "plan": [],
        }
        return partial_payload, RunMetricsCollector()

    def _persist_reactions(
        self,
        *,
        store: ArtifactStore,
        run_id: str,
        run_source_id: str,
        derived_image_id: str,
        attempt_id: str,
        payload: Dict[str, Any],
        override_outcome: str = "",
    ) -> NormalizationSummary:
        normalization = _normalize_reaction_candidates(payload)
        self.repository.purge_canonical_reactions_for_derived_image(derived_image_id)
        outcome_class = override_outcome or _classify_payload(payload)
        for index, reaction in enumerate(normalization.canonical_reactions):
            render_key = ""
            try:
                render_key = f"renders/{_hash_text(json.dumps(reaction, sort_keys=True, ensure_ascii=False))}.png"
                if not store.exists(render_key):
                    store.put_bytes(render_key, render_reaction_png(reaction), "image/png")
            except Exception:
                render_key = ""
            reaction_uid = self.repository.create_reaction(
                run_id=run_id,
                run_source_id=run_source_id,
                derived_image_id=derived_image_id,
                attempt_id=attempt_id,
                reaction_id=str(reaction.get("reaction_id") or f"reaction-{index}"),
                reaction_fingerprint=_reaction_fingerprint(reaction),
                outcome_class=outcome_class,
                structure_quality=str(normalization.structure_qualities[index]),
                acceptance_reason=str(normalization.acceptance_reasons[index]),
                render_artifact_key=render_key,
                raw_reaction_json=_json_dumps(reaction),
            )
            self.repository.add_reaction_molecules(reaction_uid, normalization.canonical_molecules[index])
            self.repository.add_reaction_conditions(reaction_uid, normalization.canonical_conditions[index])
            self.repository.add_reaction_additional_info(reaction_uid, normalization.canonical_additional_info[index])
        return normalization


def _merge_usage_completeness(current: str, new: str) -> str:
    order = {"none": 0, "partial": 1, "complete": 2}
    if current == new:
        return current
    if current == "none":
        return new
    if new == "none":
        return current
    return "partial"


def _apply_runtime_env(config: Dict[str, Any]) -> None:
    for key, value in config.items():
        if isinstance(value, bool):
            os.environ[key.upper()] = "1" if value else "0"
        elif value is None or str(value) == "":
            os.environ.pop(key.upper(), None)
        else:
            os.environ[key.upper()] = str(value)


_SERVICE_CACHE: Dict[str, ReviewDatasetService] = {}


def get_review_service(db_path: str = "") -> ReviewDatasetService:
    target = str(Path(db_path or os.getenv("REVIEW_DB_PATH", "./data/review_dataset.sqlite3")).expanduser().resolve())
    cached = _SERVICE_CACHE.get(target)
    if cached is not None:
        return cached
    service = ReviewDatasetService(ReviewRepository(Path(target)))
    _SERVICE_CACHE[target] = service
    return service
