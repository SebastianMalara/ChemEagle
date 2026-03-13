from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    env_path: str
    values: Dict[str, str]
    metadata: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    env_path: str
    values: Dict[str, Any]
    persist_to_env: bool = True


class ConfigUpdateResponse(BaseModel):
    env_path: str
    values: Dict[str, str]
    save_status: str = ""
    metadata: Dict[str, Any]


class ModelRefreshRequest(BaseModel):
    scope: Literal["main", "ocr"]
    current_model: str = ""
    values: Dict[str, Any]


class ModelRefreshResponse(BaseModel):
    scope: str
    selected_model: str
    models: List[str]
    status: str


class PreflightRequest(BaseModel):
    file_path: str = ""
    mode: str = "cloud"
    values: Dict[str, Any]
    include_pdf_section: bool = False


class PreflightResponse(BaseModel):
    diagnostics: Dict[str, Any]


class UploadResponse(BaseModel):
    stored_paths: List[str]


class LiveExperimentRequest(BaseModel):
    env_path: str = ".env.chemeagle"
    persist_to_env: bool = False
    values: Dict[str, Any]
    experiment_name: str = "ChemEagle Review Experiment"
    experiment_notes: str = ""
    batch_folder_path: str = ""
    uploaded_paths: List[str] = Field(default_factory=list)
    comparison_profiles: List[Dict[str, Any]] = Field(default_factory=lambda: [{"profile_label": "baseline"}])


class SideloadExperimentRequest(BaseModel):
    env_path: str = ".env.chemeagle"
    persist_to_env: bool = False
    values: Dict[str, Any]
    experiment_name: str = "ChemEagle Review Experiment"
    experiment_notes: str = ""
    sideload_paths: List[str] = Field(default_factory=list)
    recovery_roots: List[str] = Field(default_factory=list)


class ExperimentSubmissionResponse(BaseModel):
    status_text: str
    result: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    experiment_id: str = ""
    run_id: str = ""
    run_ids: List[str] = Field(default_factory=list)


class ExportRunRequest(BaseModel):
    output_dir: str
    review_db_path: str = ""


class RunActionRequest(BaseModel):
    review_db_path: str = ""
    retry_mode: Literal["normal", "no_agents", "recovery"] = "normal"


class ReviewUpdate(BaseModel):
    review_db_path: str = ""
    review_status: Literal["unchecked", "ok", "not_ok"]
    review_notes: str = ""


class ListEnvelope(BaseModel):
    items: List[Dict[str, Any]]
    total: int


class RunMonitorResponse(BaseModel):
    run: Dict[str, Any]
    progress: Dict[str, Any]
    sources: List[Dict[str, Any]]
    log_tail: Dict[str, Any]
    log_download_ref: str = ""
    aggregates: Dict[str, Any]
    retry_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    selected_source_detail: Optional[Dict[str, Any]] = None
