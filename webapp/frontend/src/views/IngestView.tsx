import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ExperimentSubmissionResult, RuntimeConfig } from "../api/types";

export function IngestView() {
  const [experimentName, setExperimentName] = useState("Paperbreaker Review Experiment");
  const [notes, setNotes] = useState("");
  const [batchFolderPath, setBatchFolderPath] = useState("");
  const [comparisonProfiles, setComparisonProfiles] = useState('[\n  {"profile_label": "baseline"}\n]');
  const [recoveryRoots, setRecoveryRoots] = useState("");
  const [envPath, setEnvPath] = useState(".env.chemeagle");
  const [config, setConfig] = useState<RuntimeConfig>({});
  const [liveFiles, setLiveFiles] = useState<FileList | null>(null);
  const [sideloadFiles, setSideloadFiles] = useState<FileList | null>(null);
  const [result, setResult] = useState<ExperimentSubmissionResult | null>(null);
  const [status, setStatus] = useState("Ready to queue a batch.");
  const [lastMode, setLastMode] = useState<"live" | "sideload" | "">("");

  useEffect(() => {
    void api
      .getConfig()
      .then((response) => {
        setEnvPath(response.env_path);
        setConfig(response.values);
      })
      .catch((error: Error) => {
        setStatus(error.message);
      });
  }, []);

  async function handleLiveSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    let parsedProfiles: Array<Record<string, unknown>> = [{ profile_label: "baseline" }];
    try {
      parsedProfiles = JSON.parse(comparisonProfiles) as Array<Record<string, unknown>>;
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Comparison profiles JSON is invalid.");
      return;
    }
    const uploadedPaths = liveFiles ? await api.uploadFiles(liveFiles, "live") : [];
    const submission = await api.submitLiveExperiment({
      env_path: envPath,
      persist_to_env: false,
      values: config,
      experiment_name: experimentName,
      experiment_notes: notes,
      batch_folder_path: batchFolderPath,
      uploaded_paths: uploadedPaths,
      comparison_profiles: parsedProfiles,
    });
    setResult(submission);
    setStatus(submission.status_text);
    setLastMode("live");
  }

  async function handleSideloadSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const sideloadPaths = sideloadFiles ? await api.uploadFiles(sideloadFiles, "sideload") : [];
    const submission = await api.submitSideloadExperiment({
      env_path: envPath,
      persist_to_env: false,
      values: config,
      experiment_name: experimentName,
      experiment_notes: notes,
      sideload_paths: sideloadPaths,
      recovery_roots: recoveryRoots.split(/\r?\n/).map((item) => item.trim()).filter(Boolean),
    });
    setResult(submission);
    setStatus(submission.status_text);
    setLastMode("sideload");
  }

  const queuedRuns = result?.run_ids.length || (result?.run_id ? 1 : 0);
  const queuedLabel = queuedRuns ? `${queuedRuns} run${queuedRuns === 1 ? "" : "s"} queued` : "Waiting";
  const handoffLabel = lastMode === "sideload" ? "Historical batch queued" : "Live batch queued";

  return (
    <section className="page">
      <header className="page-hero">
        <div className="hero-copy">
          <p className="eyebrow">Ingest</p>
          <h2>Queue batches</h2>
          <p className="hero-text">Upload sources or sideload a previous run.</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span className="stat-label">Status</span>
            <strong className="stat-value stat-value--small">{status}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Run ids</span>
            <strong className="stat-value">{result?.run_ids.length || 0}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Experiment</span>
            <strong className="stat-value stat-value--small">{result?.experiment_id || "—"}</strong>
          </div>
        </div>
      </header>

      <div className="page-grid">
        <form className="panel panel-stack batch-pane batch-form-pane" onSubmit={handleLiveSubmit}>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Live batch</p>
              <h3>Queue live experiment</h3>
            </div>
          </div>
          <div className="batch-pane-scroll">
            <div className="form-grid">
              <label className="field field-span-2">
                <span>Experiment name</span>
                <input value={experimentName} onChange={(event) => setExperimentName(event.target.value)} />
              </label>
              <label className="field field-span-2">
                <span>Experiment notes</span>
                <textarea value={notes} onChange={(event) => setNotes(event.target.value)} />
              </label>
              <label className="field field-span-2">
                <span>Batch folder path</span>
                <input
                  value={batchFolderPath}
                  onChange={(event) => setBatchFolderPath(event.target.value)}
                  placeholder="/srv/chemeagle/papers"
                />
              </label>
              <label className="field field-span-2">
                <span>Upload source files</span>
                <input type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp" onChange={(event) => setLiveFiles(event.target.files)} />
              </label>
              <label className="field field-span-2">
                <span>Comparison profiles JSON</span>
                <textarea
                  className="code-textarea"
                  value={comparisonProfiles}
                  onChange={(event) => setComparisonProfiles(event.target.value)}
                />
              </label>
            </div>
          </div>
          <button className="button button-primary" type="submit">
            Queue live run
          </button>
        </form>

        <form className="panel panel-stack batch-pane batch-form-pane" onSubmit={handleSideloadSubmit}>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Sideload</p>
              <h3>Import historical JSON</h3>
            </div>
          </div>
          <div className="batch-pane-scroll">
            <div className="form-grid">
              <label className="field field-span-2">
                <span>Upload sideload JSON files</span>
                <input type="file" multiple accept=".json" onChange={(event) => setSideloadFiles(event.target.files)} />
              </label>
              <label className="field field-span-2">
                <span>Recovery roots</span>
                <textarea
                  placeholder="/srv/chemeagle/recovery-a&#10;/srv/chemeagle/recovery-b"
                  value={recoveryRoots}
                  onChange={(event) => setRecoveryRoots(event.target.value)}
                />
              </label>
            </div>
          </div>
          <button className="button" type="submit">
            Queue sideload run
          </button>
        </form>
      </div>
      {result ? (
        <div className="detail-card submission-card">
          <div className="detail-card-header">
            <div>
              <p className="panel-kicker">Queued</p>
              <h4>{handoffLabel}</h4>
            </div>
            <Link className="button button-primary" to="/runs">
              Open runs monitor
            </Link>
          </div>
          <div className="meta-grid">
            <div>
              <dt>Experiment</dt>
              <dd>{result.experiment_id || "—"}</dd>
            </div>
            <div>
              <dt>Queued runs</dt>
              <dd>{queuedLabel}</dd>
            </div>
            <div>
              <dt>Next step</dt>
              <dd>Monitor progress and logs from Runs.</dd>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
