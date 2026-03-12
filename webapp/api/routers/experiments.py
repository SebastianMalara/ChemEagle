from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from ..deps import review_service_for
from ..runtime import (
    apply_runtime_env,
    collect_batch_runtime_diagnostics,
    normalize_runtime_values,
    parse_profile_configs,
    save_env_file,
    scan_supported_files,
)
from ..schemas import ExperimentSubmissionResponse, ListEnvelope, LiveExperimentRequest, SideloadExperimentRequest

router = APIRouter(tags=["experiments"])


@router.get("/experiments", response_model=ListEnvelope)
def list_experiments(review_db_path: str = "") -> ListEnvelope:
    items = review_service_for(review_db_path).list_experiments()
    return ListEnvelope(items=items, total=len(items))


@router.post("/experiments/live", response_model=ExperimentSubmissionResponse)
def submit_live_experiment(request: LiveExperimentRequest) -> ExperimentSubmissionResponse:
    env_path = Path(request.env_path).expanduser()
    values = normalize_runtime_values(request.values)
    apply_runtime_env(values)
    save_status = save_env_file(env_path, values) if request.persist_to_env else ""

    source_paths: List[str] = []
    source_paths.extend(path for path in request.uploaded_paths if path)
    source_paths.extend(path for path in scan_supported_files(request.batch_folder_path) if path not in source_paths)
    if not source_paths:
        return ExperimentSubmissionResponse(
            status_text="No PDF/image sources found for batch ingest.",
            diagnostics={},
        )

    profile_configs = parse_profile_configs(values, request.comparison_profiles)
    diagnostics = collect_batch_runtime_diagnostics(
        base_values=values,
        profile_configs=profile_configs,
        source_paths=source_paths,
        mode=values.get("CHEMEAGLE_RUN_MODE", "cloud") or "cloud",
    )
    if diagnostics["blocking_errors"]:
        status_text = f"Batch preflight failed with {len(diagnostics['blocking_errors'])} blocking issue(s)."
        if save_status:
            status_text = f"{save_status} {status_text}"
        return ExperimentSubmissionResponse(status_text=status_text, diagnostics=diagnostics)

    preflight_summary = f"Runtime provider preflight {diagnostics['runtime_provider_preflight']['status']}."
    for config in profile_configs:
        config["preflight_status"] = diagnostics["runtime_provider_preflight"]["status"]
        config["preflight_summary"] = preflight_summary

    review_db_path = values.get("REVIEW_DB_PATH", "")
    service = review_service_for(review_db_path)
    result = service.submit_live_experiment(
        experiment_name=request.experiment_name or "Live Experiment",
        notes=request.experiment_notes,
        source_paths=source_paths,
        profile_configs=profile_configs,
    )
    diagnostics["artifact_backend"] = values.get("ARTIFACT_BACKEND", "filesystem")
    diagnostics["review_db_path"] = review_db_path

    status_text = f"Queued {len(result.get('run_ids', []))} run(s) in experiment {result.get('experiment_id', '')}."
    if save_status:
        status_text = f"{save_status} {status_text}"
    first_run_id = result["run_ids"][0] if result.get("run_ids") else ""
    return ExperimentSubmissionResponse(
        status_text=status_text,
        result=result,
        diagnostics=diagnostics,
        experiment_id=result.get("experiment_id", ""),
        run_id=first_run_id,
        run_ids=result.get("run_ids", []),
    )


@router.post("/experiments/sideload", response_model=ExperimentSubmissionResponse)
def submit_sideload_experiment(request: SideloadExperimentRequest) -> ExperimentSubmissionResponse:
    env_path = Path(request.env_path).expanduser()
    values = normalize_runtime_values(request.values)
    apply_runtime_env(values)
    save_status = save_env_file(env_path, values) if request.persist_to_env else ""

    sideload_paths = [str(Path(path).expanduser().resolve()) for path in request.sideload_paths if path]
    if not sideload_paths:
        return ExperimentSubmissionResponse(status_text="No sideload JSON payloads were provided.")

    review_db_path = values.get("REVIEW_DB_PATH", "")
    service = review_service_for(review_db_path)
    result = service.submit_sideload_experiment(
        experiment_name=request.experiment_name or "Sideload Experiment",
        notes=request.experiment_notes,
        json_paths=sideload_paths,
        recovery_roots=request.recovery_roots,
        config_snapshot=dict(values),
    )
    diagnostics: Dict[str, Any] = {
        "review_db_path": review_db_path,
        "sideload_paths": sideload_paths,
        "recovery_roots": request.recovery_roots,
    }
    status_text = f"Queued {len(result.get('run_ids', []))} sideload run(s) in experiment {result.get('experiment_id', '')}."
    if save_status:
        status_text = f"{save_status} {status_text}"
    first_run_id = result["run_ids"][0] if result.get("run_ids") else ""
    return ExperimentSubmissionResponse(
        status_text=status_text,
        result=result,
        diagnostics=diagnostics,
        experiment_id=result.get("experiment_id", ""),
        run_id=first_run_id,
        run_ids=result.get("run_ids", []),
    )
