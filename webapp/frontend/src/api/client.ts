import {
  mockConfig,
  mockExperiments,
  mockMonitor,
  mockReactionDetail,
  mockReactions,
  mockRuns,
  mockSourceDetail,
  mockSubmission,
} from "./mockData";
import type {
  ConfigResponse,
  ExportedRun,
  ExperimentSubmissionResult,
  ExperimentSummary,
  ListEnvelope,
  ModelCatalogResponse,
  ReactionDetail,
  ReactionSummary,
  ReviewUpdate,
  RunMonitor,
  RunSourceDetail,
  RunSummary,
  RuntimeDiagnostics,
  RuntimeConfig,
  SaveConfigRequest,
} from "./types";

const useMocks = import.meta.env.VITE_USE_MOCKS === "1";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

async function requestJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  if (useMocks) {
    throw new Error("mock");
  }

  const response = await fetch(input, init);
  return parseJson<T>(response);
}

async function withMockFallback<T>(action: () => Promise<T>, fallback: () => T): Promise<T> {
  try {
    return await action();
  } catch (error) {
    if (useMocks || error instanceof TypeError || (error as Error).message === "mock") {
      return fallback();
    }
    throw error;
  }
}

function unwrapItems<T>(payload: T[] | ListEnvelope<T>): T[] {
  return Array.isArray(payload) ? payload : payload.items;
}

export const api = {
  getConfig() {
    return withMockFallback(
      () => requestJson<ConfigResponse>("/api/config"),
      () => mockConfig,
    );
  },

  saveConfig(payload: SaveConfigRequest) {
    return withMockFallback(
      () =>
        requestJson<ConfigResponse>("/api/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
      () => ({
        env_path: payload.env_path,
        values: payload.values,
        save_status: payload.persist_to_env ? `Saved ${payload.env_path}.` : "",
      }),
    );
  },

  refreshModels(scope: "main" | "ocr", values: RuntimeConfig) {
    const currentModel = scope === "main" ? values.LLM_MODEL || "" : values.OCR_LLM_MODEL || "";
    return withMockFallback(
      () =>
        requestJson<ModelCatalogResponse>("/api/models/refresh", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scope, current_model: currentModel, values }),
        }),
      () => ({
        scope,
        selected_model: scope === "main" ? values.LLM_MODEL || "gpt-5-mini" : values.OCR_LLM_MODEL || "gpt-5-mini",
        models: ["gpt-5-mini", "gpt-5.1", "claude-sonnet-4"],
        status: `Loaded ${scope} catalog from mock response.`,
      }),
    );
  },

  runPreflight(values: RuntimeConfig) {
    return withMockFallback(
      () =>
        requestJson<{ diagnostics: RuntimeDiagnostics }>("/api/preflight/runtime", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            values,
            mode: values.CHEMEAGLE_RUN_MODE || "cloud",
            include_pdf_section: Boolean(values.PDF_MODEL_SIZE),
          }),
        }).then((payload) => payload.diagnostics),
      () => ({
        mode: values.CHEMEAGLE_RUN_MODE,
        device: values.CHEMEAGLE_DEVICE,
        resolved_ocr_backend: values.OCR_BACKEND || "llm_vision",
        blocking_errors: [],
        warnings: ["Mock diagnostics only. Backend preflight is not connected."],
      }),
    );
  },

  uploadFiles(files: FileList | File[], prefix: string) {
    return withMockFallback(
      async () => {
        const formData = new FormData();
        Array.from(files).forEach((file) => {
          formData.append("files", file);
        });
        const response = await requestJson<{ stored_paths: string[] }>(
          `/api/uploads?prefix=${encodeURIComponent(prefix)}`,
          {
            method: "POST",
            body: formData,
          },
        );
        return response.stored_paths;
      },
      () => Array.from(files).map((file) => `/mock/${file.name}`),
    );
  },

  submitLiveExperiment(payload: Record<string, unknown>) {
    return withMockFallback(
      () =>
        requestJson<ExperimentSubmissionResult>("/api/experiments/live", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).then((response) => ({
          ...response,
          run_ids: response.run_ids || [],
        })),
      () => mockSubmission,
    );
  },

  submitSideloadExperiment(payload: Record<string, unknown>) {
    return withMockFallback(
      () =>
        requestJson<ExperimentSubmissionResult>("/api/experiments/sideload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).then((response) => ({
          ...response,
          run_ids: response.run_ids || [],
        })),
      () => mockSubmission,
    );
  },

  listExperiments() {
    return withMockFallback(
      () => requestJson<ListEnvelope<ExperimentSummary>>("/api/experiments").then(unwrapItems),
      () => mockExperiments,
    );
  },

  listRuns(experimentId = "") {
    const url = experimentId ? `/api/runs?experiment_id=${encodeURIComponent(experimentId)}` : "/api/runs";
    return withMockFallback(
      () => requestJson<ListEnvelope<RunSummary>>(url).then(unwrapItems),
      () => mockRuns.filter((run) => !experimentId || run.experiment_id === experimentId),
    );
  },

  getRunMonitor(runId: string) {
    return withMockFallback(
      () => requestJson<RunMonitor>(`/api/runs/${encodeURIComponent(runId)}/monitor`),
      () => ({ ...mockMonitor, run: { ...mockMonitor.run, run_id: runId || mockMonitor.run.run_id } }),
    );
  },

  getRunSource(runId: string, runSourceId: string) {
    return withMockFallback(
      () =>
        requestJson<RunSourceDetail>(
          `/api/runs/${encodeURIComponent(runId)}/sources/${encodeURIComponent(runSourceId)}`,
        ),
      () => mockSourceDetail,
    );
  },

  listRetryCandidates(runId: string) {
    return withMockFallback(
      () =>
        requestJson<ListEnvelope<RunSourceDetail["derived_images"][number]>>(
          `/api/runs/${encodeURIComponent(runId)}/retry-candidates`,
        ).then(unwrapItems),
      () => mockSourceDetail.derived_images,
    );
  },

  retryFailed(runId: string) {
    return withMockFallback(
      () =>
        requestJson<{ status_text?: string; message?: string }>(
          `/api/runs/${encodeURIComponent(runId)}/actions/retry-failed`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          },
        ).then((payload) => ({ message: payload.status_text || payload.message || "" })),
      () => ({ message: `Retried failed images for ${runId} in mock mode.` }),
    );
  },

  retryRedo(runId: string) {
    return withMockFallback(
      () =>
        requestJson<{ status_text?: string; message?: string }>(
          `/api/runs/${encodeURIComponent(runId)}/actions/retry-redo`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          },
        ).then((payload) => ({ message: payload.status_text || payload.message || "" })),
      () => ({ message: `Retried redo images for ${runId} in mock mode.` }),
    );
  },

  reprocessRun(runId: string) {
    return withMockFallback(
      () =>
        requestJson<{ status_text?: string; message?: string }>(
          `/api/runs/${encodeURIComponent(runId)}/actions/reprocess`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          },
        ).then((payload) => ({ message: payload.status_text || payload.message || "" })),
      () => ({ message: `Reprocessed normalization for ${runId} in mock mode.` }),
    );
  },

  retryDerivedImage(derivedImageId: string, executionMode: string) {
    return withMockFallback(
      () =>
        requestJson<{ status_text?: string; message?: string }>(
          `/api/derived-images/${encodeURIComponent(derivedImageId)}/retry`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ retry_mode: executionMode }),
          },
        ).then((payload) => ({ message: payload.status_text || payload.message || "" })),
      () => ({ message: `Retried ${derivedImageId} with ${executionMode} in mock mode.` }),
    );
  },

  reprocessDerivedImage(derivedImageId: string) {
    return withMockFallback(
      () =>
        requestJson<{ status_text?: string; message?: string }>(
          `/api/derived-images/${encodeURIComponent(derivedImageId)}/reprocess`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          },
        ).then((payload) => ({ message: payload.status_text || payload.message || "" })),
      () => ({ message: `Reprocessed ${derivedImageId} in mock mode.` }),
    );
  },

  exportRun(runId: string, outputDir: string) {
    return withMockFallback(
      () =>
        requestJson<{ files: ExportedRun }>(`/api/runs/${encodeURIComponent(runId)}/export`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ output_dir: outputDir }),
        }).then((payload) => payload.files),
      () => ({
        reactions: `${outputDir}/${runId}_reactions.parquet`,
        reaction_molecules: `${outputDir}/${runId}_reaction_molecules.parquet`,
      }),
    );
  },

  listReviewReactions(runId = "", reviewStatus = "", outcomeClass = "") {
    const params = new URLSearchParams();
    if (runId) params.set("run_id", runId);
    if (reviewStatus && reviewStatus !== "all") params.set("review_status", reviewStatus);
    if (outcomeClass && outcomeClass !== "all") params.set("outcome_class", outcomeClass);
    const url = `/api/review/reactions${params.toString() ? `?${params.toString()}` : ""}`;
    return withMockFallback(
      () => requestJson<ListEnvelope<ReactionSummary>>(url).then(unwrapItems),
      () =>
        mockReactions.filter((reaction) => {
          if (runId && reaction.run_id !== runId) return false;
          if (reviewStatus && reviewStatus !== "all" && reaction.review_status !== reviewStatus) return false;
          if (outcomeClass && outcomeClass !== "all" && reaction.outcome_class !== outcomeClass) return false;
          return true;
        }),
    );
  },

  getReactionDetail(reactionUid: string) {
    return withMockFallback(
      () =>
        requestJson<ReactionDetail>(`/api/review/reactions/${encodeURIComponent(reactionUid)}`).then((detail) => ({
          ...detail,
          source_artifact_url: detail.source_artifact_url || detail.source_url,
          derived_artifact_url: detail.derived_artifact_url || detail.derived_image_url,
          render_artifact_url: detail.render_artifact_url || detail.render_image_url,
        })),
      () => ({
        ...mockReactionDetail,
        reaction_uid: reactionUid || mockReactionDetail.reaction_uid,
      }),
    );
  },

  updateReview(reactionUid: string, payload: ReviewUpdate) {
    return withMockFallback(
      () =>
        requestJson<{ reaction_uid: string; review_status: string; review_notes: string }>(
          `/api/review/reactions/${encodeURIComponent(reactionUid)}`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              review_status: payload.review_status,
              review_notes: payload.review_notes,
            }),
          },
        ).then((saved) => ({
          reaction_uid: saved.reaction_uid,
          review_status: saved.review_status,
          review_notes: saved.review_notes,
        })),
      () => payload,
    );
  },

  artifactUrl(reactionUid: string, kind: "source" | "derived" | "render") {
    return `/api/review/reactions/${encodeURIComponent(reactionUid)}/artifacts/${kind}`;
  },
};
