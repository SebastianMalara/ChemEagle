import type {
  ConfigResponse,
  ExperimentSummary,
  ExperimentSubmissionResult,
  ReactionDetail,
  ReactionSummary,
  RunMonitor,
  RunSourceDetail,
  RunSummary,
} from "./types";

export const mockConfig: ConfigResponse = {
  env_path: ".env.chemeagle",
  values: {
    CHEMEAGLE_RUN_MODE: "cloud",
    CHEMEAGLE_DEVICE: "auto",
    LLM_PROVIDER: "openai",
    LLM_MODEL: "gpt-5-mini",
    OPENAI_API_KEY: "",
    OCR_BACKEND: "llm_vision",
    OCR_LLM_INHERIT_MAIN: "1",
    PDF_MODEL_SIZE: "large",
    ARTIFACT_BACKEND: "filesystem",
    ARTIFACT_FILESYSTEM_ROOT: "./data/artifacts",
    REVIEW_DB_PATH: "./data/review_dataset.sqlite3",
  },
};

export const mockExperiments: ExperimentSummary[] = [
  {
    experiment_id: "exp-demo-1",
    name: "Paperbreaker Review Experiment",
    status: "running",
    run_count: 2,
    created_at: "2026-03-12T10:00:00Z",
  },
  {
    experiment_id: "exp-demo-2",
    name: "Paper batch archive",
    status: "completed",
    run_count: 1,
    created_at: "2026-03-11T09:30:00Z",
  },
];

export const mockRuns: RunSummary[] = [
  {
    run_id: "run-demo-1",
    experiment_id: "exp-demo-1",
    experiment_name: "Paperbreaker Review Experiment",
    profile_label: "baseline",
    ingest_mode: "live_batch",
    status: "running",
    total_reactions: 48,
    total_failures: 4,
    total_redo: 3,
    estimated_cost_usd: 2.83,
    created_at: "2026-03-12T10:05:00Z",
  },
  {
    run_id: "run-demo-2",
    experiment_id: "exp-demo-1",
    experiment_name: "Paperbreaker Review Experiment",
    profile_label: "no_agents_retry",
    ingest_mode: "live_batch",
    status: "queued",
    total_reactions: 0,
    total_failures: 0,
    total_redo: 0,
    estimated_cost_usd: 0,
    created_at: "2026-03-12T10:07:00Z",
  },
];

export const mockMonitor: RunMonitor = {
  run: mockRuns[0],
  progress: {
    progress_fraction: 0.63,
    progress_label: "12 / 19 sources processed",
    current_phase_label: "Normalize and persist",
    current_source_label: "scheme_014.pdf",
    status_summary: "Run is active. salvageable=2 rejects=3 planner/tool empties=1 local crashes=1 provider=0",
    is_active: true,
  },
  aggregates: {
    salvageable_from_raw: 2,
    normalization_rejects: 3,
    planner_tool_empties: 1,
    local_runtime_crashes: 1,
    systemic_provider_failures: 0,
  },
  sources: [
    {
      run_source_id: "source-1",
      input_order: 0,
      original_filename: "scheme_014.pdf",
      status: "running",
      current_phase: "persist",
      completed_derived_images: 3,
      expected_derived_images: 5,
      reaction_count: 7,
      failed_derived_images: 1,
    },
    {
      run_source_id: "source-2",
      input_order: 1,
      original_filename: "scheme_015.pdf",
      status: "completed",
      current_phase: "done",
      completed_derived_images: 4,
      expected_derived_images: 4,
      reaction_count: 9,
      failed_derived_images: 0,
    },
  ],
  log_tail: {
    formatted: "[INFO] Extracting scheme_014.pdf\n[WARN] Normalization rejected 1 reaction\n[INFO] Persisted 7 reactions",
    raw: "[INFO] Extracting scheme_014.pdf\n[WARN] Normalization rejected 1 reaction\n[INFO] Persisted 7 reactions",
    events: [],
  },
};

export const mockSourceDetail: RunSourceDetail = {
  ...mockMonitor.sources[0],
  derived_images: [
    {
      derived_image_id: "derived-1",
      image_index: 0,
      page_hint: "page 3",
      status: "completed",
      outcome_class: "succeeded",
      reaction_count: 4,
      accepted_reaction_count: 3,
      rejected_reaction_count: 1,
      attempt_count: 2,
      normalization_status: "partial",
      original_filename: "scheme_014.pdf",
      attempts: [
        {
          attempt_id: "attempt-1",
          attempt_no: 1,
          trigger: "initial",
          execution_mode: "normal",
          status: "completed",
          failure_kind: "",
          error_summary: "",
        },
        {
          attempt_id: "attempt-2",
          attempt_no: 2,
          trigger: "manual_reprocess",
          execution_mode: "recovery",
          status: "completed",
          failure_kind: "",
          error_summary: "",
        },
      ],
    },
    {
      derived_image_id: "derived-2",
      image_index: 1,
      page_hint: "page 4",
      status: "failed",
      outcome_class: "failed",
      reaction_count: 0,
      accepted_reaction_count: 0,
      rejected_reaction_count: 0,
      attempt_count: 1,
      normalization_status: "redo_pending",
      original_filename: "scheme_014.pdf",
      error_text: "Tool call returned empty payload",
      attempts: [
        {
          attempt_id: "attempt-3",
          attempt_no: 1,
          trigger: "initial",
          execution_mode: "normal",
          status: "failed",
          failure_kind: "tool_call_empty",
          error_summary: "Tool call returned empty payload",
        },
      ],
    },
  ],
};

export const mockReactions: ReactionSummary[] = [
  {
    reaction_uid: "reaction-1",
    run_id: "run-demo-1",
    reaction_id: "0_1",
    review_status: "unchecked",
    outcome_class: "succeeded",
    structure_quality: "clean",
    profile_label: "baseline",
    original_filename: "scheme_014.pdf",
    page_hint: "page 3",
    estimated_cost_usd: 0.09,
  },
  {
    reaction_uid: "reaction-2",
    run_id: "run-demo-1",
    reaction_id: "0_2",
    review_status: "ok",
    outcome_class: "succeeded",
    structure_quality: "partial",
    profile_label: "baseline",
    original_filename: "scheme_014.pdf",
    page_hint: "page 3",
    estimated_cost_usd: 0.07,
  },
  {
    reaction_uid: "reaction-3",
    run_id: "run-demo-1",
    reaction_id: "1_1",
    review_status: "not_ok",
    outcome_class: "needs_redo",
    structure_quality: "rejected_missing_smiles",
    profile_label: "baseline",
    original_filename: "scheme_015.pdf",
    page_hint: "page 7",
    estimated_cost_usd: 0.11,
  },
];

export const mockReactionDetail: ReactionDetail = {
  ...mockReactions[0],
  reaction_fingerprint: "0d3f54f1e52f",
  acceptance_reason: "Accepted after canonical normalization",
  review_notes: "",
  image_index: 0,
  raw_reaction_json: {
    reaction_id: "0_1",
    reactants: [{ label: "A", smiles: "CCOC(=O)N" }],
    products: [{ label: "P", smiles: "CCOC(=O)Nc1ccccc1" }],
  },
  molecules: [
    {
      side: "reactant",
      ordinal: 0,
      label: "A",
      smiles: "CCOC(=O)N",
      validation_kind: "rdkit_valid",
    },
    {
      side: "product",
      ordinal: 1,
      label: "P",
      smiles: "CCOC(=O)Nc1ccccc1",
      validation_kind: "rdkit_valid",
    },
  ],
  conditions: [
    { condition_type: "temperature", value_text: "80 C" },
    { condition_type: "time", value_text: "16 h" },
  ],
  additional_info: [
    { info_type: "yield", value_text: "78%" },
    { info_type: "catalyst", value_text: "Pd(PPh3)4" },
  ],
  source_artifact_url:
    "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=1800&q=80",
  derived_artifact_url:
    "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=1200&q=80",
  render_artifact_url:
    "https://images.unsplash.com/photo-1518152006812-edab29b069ac?auto=format&fit=crop&w=1200&q=80",
};

export const mockSubmission: ExperimentSubmissionResult = {
  status_text: "Queued 2 run(s) in experiment exp-demo-1.",
  experiment_id: "exp-demo-1",
  run_ids: ["run-demo-1", "run-demo-2"],
  diagnostics: {
    blocking_errors: [],
    warnings: ["runtime_provider_preflight: using demo mock data"],
  },
};
