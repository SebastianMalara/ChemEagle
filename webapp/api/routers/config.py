from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from ..runtime import (
    ENV_FILE_DEFAULT,
    apply_runtime_env,
    collect_preflight_diagnostics,
    config_metadata,
    default_runtime_values,
    normalize_runtime_values,
    refresh_model_catalog,
    save_env_file,
)
from ..schemas import (
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ModelRefreshRequest,
    ModelRefreshResponse,
    PreflightRequest,
    PreflightResponse,
    RuntimeConfig,
)

router = APIRouter(tags=["config"])


@router.get("/config", response_model=RuntimeConfig)
def get_config(env_path: str = str(ENV_FILE_DEFAULT)) -> RuntimeConfig:
    resolved = Path(env_path).expanduser()
    values = default_runtime_values(resolved)
    return RuntimeConfig(env_path=str(resolved), values=values, metadata=config_metadata(values))


@router.put("/config", response_model=ConfigUpdateResponse)
def update_config(request: ConfigUpdateRequest) -> ConfigUpdateResponse:
    env_path = Path(request.env_path).expanduser()
    values = normalize_runtime_values(request.values)
    apply_runtime_env(values)
    save_status = save_env_file(env_path, values) if request.persist_to_env else ""
    return ConfigUpdateResponse(
        env_path=str(env_path),
        values=values,
        save_status=save_status,
        metadata=config_metadata(values),
    )


@router.post("/models/refresh", response_model=ModelRefreshResponse)
def refresh_models(request: ModelRefreshRequest) -> ModelRefreshResponse:
    values = normalize_runtime_values(request.values)
    result = refresh_model_catalog(request.scope, request.current_model, values)
    return ModelRefreshResponse(**result)


@router.post("/preflight/runtime", response_model=PreflightResponse)
def run_runtime_preflight(request: PreflightRequest) -> PreflightResponse:
    values = normalize_runtime_values(request.values)
    apply_runtime_env(values)
    diagnostics = collect_preflight_diagnostics(
        request.file_path,
        request.mode,
        values,
        include_pdf_section=request.include_pdf_section,
    )
    return PreflightResponse(diagnostics=diagnostics)
