from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..deps import review_service_for
from ..schemas import ExportRunRequest, ListEnvelope, RunActionRequest, RunMonitorResponse

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=ListEnvelope)
def list_runs(review_db_path: str = "", experiment_id: str = "") -> ListEnvelope:
    items = review_service_for(review_db_path).list_runs(experiment_id=experiment_id)
    return ListEnvelope(items=items, total=len(items))


@router.get("/runs/{run_id}/monitor", response_model=RunMonitorResponse)
def get_run_monitor(
    run_id: str,
    review_db_path: str = "",
    tail_lines: int = 200,
    min_level: str = "INFO",
    raw: bool = False,
    include_suppressed: bool = False,
    selected_run_source_id: str = "",
) -> RunMonitorResponse:
    service = review_service_for(review_db_path)
    try:
        monitor = service.get_run_monitor(
            run_id,
            tail_lines=tail_lines,
            min_level=min_level,
            raw=raw,
            include_suppressed=include_suppressed,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_detail = None
    if selected_run_source_id:
        try:
            source_detail = service.get_run_source_monitor(selected_run_source_id)
        except KeyError:
            source_detail = None
    retry_candidates = service.list_retry_candidates(run_id)
    return RunMonitorResponse(
        run=monitor["run"],
        progress=monitor.get("progress", {}) or {},
        sources=monitor.get("sources", []),
        log_tail=monitor.get("log_tail", {}),
        log_download_ref=service.get_log_download_ref(run_id),
        aggregates=monitor.get("aggregates", {}),
        retry_candidates=retry_candidates,
        selected_source_detail=source_detail,
    )


@router.get("/runs/{run_id}/sources/{run_source_id}")
def get_run_source_detail(run_id: str, run_source_id: str, review_db_path: str = "") -> Dict[str, Any]:
    del run_id
    service = review_service_for(review_db_path)
    try:
        return service.get_run_source_monitor(run_source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/retry-candidates", response_model=ListEnvelope)
def get_retry_candidates(run_id: str, review_db_path: str = "") -> ListEnvelope:
    items = review_service_for(review_db_path).list_retry_candidates(run_id)
    return ListEnvelope(items=items, total=len(items))


@router.post("/runs/{run_id}/export")
def export_run(run_id: str, request: ExportRunRequest) -> Dict[str, Any]:
    if not run_id:
        raise HTTPException(status_code=400, detail="No run selected for export.")
    exported = review_service_for(request.review_db_path).export_run_to_parquet(run_id, request.output_dir)
    return {"status_text": f"Exported parquet files for run {run_id}.", "files": exported}


@router.post("/runs/{run_id}/actions/retry-failed")
def retry_failed(run_id: str, request: RunActionRequest) -> Dict[str, Any]:
    retried = review_service_for(request.review_db_path).retry_failed_derived_images(
        run_id,
        include_needs_redo=False,
        include_failed=True,
    )
    return {"status_text": f"Retried {len(retried)} failed derived image(s) for run {run_id}.", "retried": retried}


@router.post("/runs/{run_id}/actions/retry-redo")
def retry_redo(run_id: str, request: RunActionRequest) -> Dict[str, Any]:
    retried = review_service_for(request.review_db_path).retry_failed_derived_images(
        run_id,
        include_needs_redo=True,
        include_failed=False,
    )
    return {"status_text": f"Retried {len(retried)} redo candidate(s) for run {run_id}.", "retried": retried}


@router.post("/runs/{run_id}/actions/reprocess")
def reprocess_run(run_id: str, request: RunActionRequest) -> Dict[str, Any]:
    summary = review_service_for(request.review_db_path).reprocess_normalization_for_run(
        run_id,
        only_invalid_reactions=False,
    )
    return {
        "status_text": (
            f"Reprocessed normalization for run {run_id}: "
            f"{summary.get('derived_images', 0)} derived image(s), "
            f"{summary.get('accepted_reactions', 0)} accepted reaction(s)."
        ),
        "summary": summary,
    }


@router.post("/derived-images/{derived_image_id}/retry")
def retry_derived_image(derived_image_id: str, request: RunActionRequest) -> Dict[str, Any]:
    execution_mode = request.retry_mode or "normal"
    trigger = "manual_no_agents_retry" if execution_mode == "no_agents" else "manual_retry"
    status_text = review_service_for(request.review_db_path).retry_derived_image(
        derived_image_id,
        trigger=trigger,
        execution_mode=execution_mode,
    )
    return {"status_text": status_text}


@router.post("/derived-images/{derived_image_id}/reprocess")
def reprocess_derived_image(derived_image_id: str, request: RunActionRequest) -> Dict[str, Any]:
    summary = review_service_for(request.review_db_path).reprocess_normalization_for_derived_images(
        [derived_image_id],
        purge_existing=True,
    )
    return {
        "status_text": (
            f"Reprocessed normalization for {derived_image_id}: "
            f"{summary.get('accepted_reactions', 0)} accepted / {summary.get('rejected_reactions', 0)} rejected."
        ),
        "summary": summary,
    }
