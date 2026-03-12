from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from review_artifacts import create_artifact_store_from_config

from ..deps import review_service_for
from ..runtime import default_runtime_values, normalize_runtime_values
from ..schemas import ListEnvelope, ReviewUpdate

router = APIRouter(tags=["review"])


def _run_config_for_reaction(service: Any, detail: Dict[str, Any]) -> Dict[str, str]:
    run = service.repository.get_run(str(detail["run_id"]))
    snapshot = json.loads((run or {}).get("config_snapshot_json") or "{}")
    values = default_runtime_values()
    values.update(normalize_runtime_values(snapshot))
    return values


def _artifact_store(config: Dict[str, str], backend: str):
    merged = dict(config)
    if backend:
        merged["ARTIFACT_BACKEND"] = backend
    return create_artifact_store_from_config(merged)


def _artifact_response(config: Dict[str, str], backend: str, key: str, fallback_name: str = "") -> Response:
    if not key:
        raise HTTPException(status_code=404, detail="Artifact key is empty.")
    store = _artifact_store(config, backend)
    media_type = mimetypes.guess_type(fallback_name or key)[0] or "application/octet-stream"
    if backend == "filesystem":
        ref = store.get_download_ref(key)
        if Path(ref).exists():
            return FileResponse(ref, media_type=media_type)
    data = store.get_bytes(key)
    return Response(content=data, media_type=media_type)


@router.get("/review/reactions", response_model=ListEnvelope)
def list_reactions(
    review_db_path: str = "",
    experiment_id: str = "",
    run_id: str = "",
    review_status: str = "",
    outcome_class: str = "",
) -> ListEnvelope:
    items = review_service_for(review_db_path).list_reactions(
        experiment_id=experiment_id,
        run_id=run_id,
        review_status=review_status,
        outcome_class=outcome_class,
    )
    return ListEnvelope(items=items, total=len(items))


@router.get("/review/reactions/{reaction_uid}")
def get_reaction_detail(reaction_uid: str, request: Request, review_db_path: str = "") -> Dict[str, Any]:
    service = review_service_for(review_db_path)
    try:
        detail = service.get_reaction_detail(reaction_uid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    source_url = ""
    derived_image_url = ""
    render_image_url = ""
    if detail.get("source_artifact_key"):
        source_url = str(request.url_for("get_reaction_artifact", reaction_uid=reaction_uid, kind="source"))
    if detail.get("derived_artifact_key"):
        derived_image_url = str(request.url_for("get_reaction_artifact", reaction_uid=reaction_uid, kind="derived"))
    if detail.get("render_artifact_key"):
        render_image_url = str(request.url_for("get_reaction_artifact", reaction_uid=reaction_uid, kind="render"))

    payload = dict(detail)
    raw_reaction_json = payload.get("raw_reaction_json")
    if isinstance(raw_reaction_json, str):
        try:
            payload["raw_reaction_json"] = json.loads(raw_reaction_json)
        except json.JSONDecodeError:
            payload["raw_reaction_json"] = {"raw": raw_reaction_json}
    payload["source_url"] = source_url
    payload["derived_image_url"] = derived_image_url
    payload["render_image_url"] = render_image_url
    payload["source_artifact_url"] = source_url
    payload["derived_artifact_url"] = derived_image_url
    payload["render_artifact_url"] = render_image_url
    return payload


@router.put("/review/reactions/{reaction_uid}")
def update_reaction_detail(reaction_uid: str, request: ReviewUpdate) -> Dict[str, Any]:
    service = review_service_for(request.review_db_path)
    try:
        service.update_reaction_review(
            reaction_uid,
            review_status=request.review_status,
            review_notes=request.review_notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "status_text": f"Saved review state for {reaction_uid}.",
        "reaction_uid": reaction_uid,
        "review_status": request.review_status,
        "review_notes": request.review_notes,
    }


@router.get("/review/reactions/{reaction_uid}/artifacts/{kind}", name="get_reaction_artifact")
def get_reaction_artifact(reaction_uid: str, kind: str, review_db_path: str = "") -> Response:
    service = review_service_for(review_db_path)
    try:
        detail = service.get_reaction_detail(reaction_uid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    config = _run_config_for_reaction(service, detail)
    if kind == "source":
        return _artifact_response(
            config,
            str(detail.get("source_artifact_backend") or ""),
            str(detail.get("source_artifact_key") or ""),
            str(detail.get("original_filename") or ""),
        )
    if kind == "derived":
        return _artifact_response(
            config,
            str(detail.get("derived_backend") or ""),
            str(detail.get("derived_artifact_key") or ""),
            str(detail.get("derived_artifact_key") or "reaction.png"),
        )
    if kind == "render":
        return _artifact_response(
            config,
            str(detail.get("derived_backend") or ""),
            str(detail.get("render_artifact_key") or ""),
            str(detail.get("render_artifact_key") or "render.png"),
        )
    raise HTTPException(status_code=404, detail=f"Unknown artifact kind: {kind}")
