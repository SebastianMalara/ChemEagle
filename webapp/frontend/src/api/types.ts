export interface RuntimeConfig {
  CHEMEAGLE_RUN_MODE?: string;
  CHEMEAGLE_DEVICE?: string;
  LLM_PROVIDER?: string;
  LLM_MODEL?: string;
  API_KEY?: string;
  AZURE_ENDPOINT?: string;
  API_VERSION?: string;
  OPENAI_API_KEY?: string;
  OPENAI_BASE_URL?: string;
  ANTHROPIC_API_KEY?: string;
  VLLM_BASE_URL?: string;
  VLLM_API_KEY?: string;
  OCR_BACKEND?: string;
  OCR_LLM_INHERIT_MAIN?: string;
  OCR_LLM_PROVIDER?: string;
  OCR_LLM_MODEL?: string;
  OCR_API_KEY?: string;
  OCR_AZURE_ENDPOINT?: string;
  OCR_API_VERSION?: string;
  OCR_OPENAI_API_KEY?: string;
  OCR_OPENAI_BASE_URL?: string;
  OCR_ANTHROPIC_API_KEY?: string;
  OCR_VLLM_BASE_URL?: string;
  OCR_VLLM_API_KEY?: string;
  OCR_LANG?: string;
  OCR_CONFIG?: string;
  TESSERACT_CMD?: string;
  MOLECULE_SMILES_RESCUE?: string;
  MOLECULE_SMILES_RESCUE_CONFIDENCE?: string;
  PDF_MODEL_SIZE?: string;
  PDF_PERSIST_IMAGES?: string;
  PDF_PERSIST_DIR?: string;
  ARTIFACT_BACKEND?: string;
  ARTIFACT_FILESYSTEM_ROOT?: string;
  ARTIFACT_S3_ENDPOINT_URL?: string;
  ARTIFACT_S3_ACCESS_KEY_ID?: string;
  ARTIFACT_S3_SECRET_ACCESS_KEY?: string;
  ARTIFACT_S3_BUCKET?: string;
  ARTIFACT_S3_REGION?: string;
  ARTIFACT_S3_USE_SSL?: string;
  ARTIFACT_S3_KEY_PREFIX?: string;
  REVIEW_DB_PATH?: string;
  [key: string]: string | undefined;
}

export interface ConfigResponse {
  env_path: string;
  values: RuntimeConfig;
  metadata?: Record<string, unknown>;
  save_status?: string;
}

export interface SaveConfigRequest {
  env_path: string;
  values: RuntimeConfig;
  persist_to_env: boolean;
}

export interface RuntimeDiagnostics {
  mode?: string;
  device?: string;
  resolved_ocr_backend?: string;
  blocking_errors: string[];
  warnings: string[];
  [key: string]: unknown;
}

export interface ModelCatalogResponse {
  scope: "main" | "ocr";
  selected_model: string;
  models: string[];
  status: string;
}

export interface ListEnvelope<T> {
  items: T[];
  total: number;
}

export interface ExperimentSubmissionResult {
  status_text: string;
  experiment_id: string;
  run_id?: string;
  run_ids: string[];
  diagnostics?: RuntimeDiagnostics;
  raw?: Record<string, unknown>;
}

export interface ExperimentSummary {
  experiment_id: string;
  name: string;
  status: string;
  created_at?: string;
  run_count?: number;
}

export interface RunSummary {
  run_id: string;
  experiment_id: string;
  experiment_name?: string;
  profile_label?: string;
  ingest_mode?: string;
  status?: string;
  total_reactions?: number;
  total_failures?: number;
  total_redo?: number;
  estimated_cost_usd?: number | null;
  created_at?: string;
}

export interface RunSource {
  run_source_id: string;
  input_order?: number;
  original_filename?: string;
  source_type?: string;
  status?: string;
  current_phase?: string;
  completed_derived_images?: number;
  expected_derived_images?: number;
  reaction_count?: number;
  failed_derived_images?: number;
  error_summary?: string;
}

export interface DerivedImageAttempt {
  attempt_id: string;
  attempt_no?: number;
  trigger?: string;
  execution_mode?: string;
  status?: string;
  failure_kind?: string;
  error_summary?: string;
  raw_artifact_key?: string;
}

export interface DerivedImage {
  derived_image_id: string;
  image_index?: number;
  page_hint?: string;
  status?: string;
  outcome_class?: string;
  reaction_count?: number;
  accepted_reaction_count?: number;
  rejected_reaction_count?: number;
  attempt_count?: number;
  normalization_status?: string;
  error_text?: string;
  original_filename?: string;
  attempts?: DerivedImageAttempt[];
}

export interface RetryCandidate extends DerivedImage {}

export interface RunMonitor {
  run: RunSummary;
  progress: {
    progress_fraction?: number;
    progress_label?: string;
    current_phase_label?: string;
    current_source_label?: string;
    status_summary?: string;
    is_active?: boolean;
  };
  aggregates: Record<string, number>;
  sources: RunSource[];
  retry_candidates?: DerivedImage[];
  log_tail: {
    formatted?: string;
    raw?: string;
    events?: Array<Record<string, unknown>>;
  };
  log_download_ref?: string;
}

export interface RunSourceDetail extends RunSource {
  derived_images: DerivedImage[];
}

export interface ExportedRun {
  reactions?: string;
  reaction_molecules?: string;
  reaction_conditions?: string;
  reaction_additional_info?: string;
}

export interface ReactionSummary {
  reaction_uid: string;
  run_id: string;
  reaction_id: string;
  review_status: string;
  outcome_class?: string;
  structure_quality?: string;
  profile_label?: string;
  original_filename?: string;
  page_hint?: string;
  estimated_cost_usd?: number | null;
}

export interface ReactionMolecule {
  side?: string;
  ordinal?: number;
  label?: string;
  smiles?: string;
  validation_kind?: string;
  structure_quality?: string;
}

export interface ReactionCondition {
  condition_type?: string;
  value_text?: string;
}

export interface ReactionInfo {
  info_type?: string;
  value_text?: string;
}

export interface ReactionDetail extends ReactionSummary {
  reaction_fingerprint?: string;
  acceptance_reason?: string;
  review_notes?: string;
  raw_reaction_json?: unknown;
  molecules?: ReactionMolecule[];
  conditions?: ReactionCondition[];
  additional_info?: ReactionInfo[];
  image_index?: number;
  source_artifact_url?: string;
  derived_artifact_url?: string;
  render_artifact_url?: string;
  source_url?: string;
  derived_image_url?: string;
  render_image_url?: string;
}

export interface ReviewUpdate {
  reaction_uid: string;
  review_status: string;
  review_notes: string;
}
