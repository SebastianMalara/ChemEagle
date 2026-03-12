import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAsyncResource, usePollingResource } from "../api/hooks";
import type { DerivedImage, ExperimentSummary, RunSourceDetail, RunSummary } from "../api/types";

function statusTone(status?: string) {
  if (status === "completed") return "chip-ok";
  if (status === "failed" || status === "interrupted") return "chip-warn";
  return "";
}

export function RunsView() {
  const experimentsResource = useAsyncResource<ExperimentSummary[]>(
    () => api.listExperiments(),
    [],
  );
  const [selectedExperiment, setSelectedExperiment] = useState("");
  const runsResource = useAsyncResource<RunSummary[]>(
    () => api.listRuns(selectedExperiment),
    [selectedExperiment],
  );
  const [selectedRun, setSelectedRun] = useState("");
  const monitorResource = usePollingResource(
    () => (selectedRun ? api.getRunMonitor(selectedRun) : Promise.resolve(null)),
    [selectedRun],
    2500,
  );
  const [selectedSource, setSelectedSource] = useState("");
  const [selectedDerived, setSelectedDerived] = useState("");
  const [sourceDetail, setSourceDetail] = useState<RunSourceDetail | null>(null);
  const [exportDir, setExportDir] = useState("./data/exports");
  const [actionStatus, setActionStatus] = useState("");

  useEffect(() => {
    if (runsResource.data?.length && !selectedRun) {
      setSelectedRun(runsResource.data[0].run_id);
    }
  }, [runsResource.data, selectedRun]);

  useEffect(() => {
    const sources = monitorResource.data?.sources || [];
    if (sources.length && !selectedSource) {
      setSelectedSource(sources[0].run_source_id);
    }
  }, [monitorResource.data, selectedSource]);

  useEffect(() => {
    if (!selectedRun || !selectedSource) {
      setSourceDetail(null);
      return;
    }

    void api
      .getRunSource(selectedRun, selectedSource)
      .then((detail) => {
        setSourceDetail(detail);
        if (!selectedDerived && detail.derived_images.length) {
          setSelectedDerived(detail.derived_images[0].derived_image_id);
        }
      })
      .catch((error: Error) => {
        setActionStatus(error.message);
      });
  }, [selectedRun, selectedSource, selectedDerived]);

  const selectedDerivedImage =
    sourceDetail?.derived_images.find((item) => item.derived_image_id === selectedDerived) ||
    sourceDetail?.derived_images[0] ||
    null;

  async function runAction(action: () => Promise<{ message?: string }>, fallback: string) {
    const response = await action();
    setActionStatus(response.message || fallback);
    await monitorResource.reload();
  }

  return (
    <section className="page">
      <header className="page-hero">
        <div className="hero-copy">
          <p className="eyebrow">Runs</p>
          <h2>Polling monitor for experiments, sources, derived images, retries, and exports.</h2>
          <p className="hero-text">
            This keeps the current timer-driven behavior but turns it into a
            cleaner operational view instead of a callback dump.
          </p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span className="stat-label">Experiments</span>
            <strong className="stat-value">{experimentsResource.data?.length || 0}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Runs</span>
            <strong className="stat-value">{runsResource.data?.length || 0}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Selection</span>
            <strong className="stat-value stat-value--small">{selectedRun || "—"}</strong>
          </div>
        </div>
      </header>

      <div className="panel page-stack">
        <div className="filters filters-wide">
          <label className="field">
            <span>Experiment</span>
            <select value={selectedExperiment} onChange={(event) => setSelectedExperiment(event.target.value)}>
              <option value="">All experiments</option>
              {experimentsResource.data?.map((experiment) => (
                <option key={experiment.experiment_id} value={experiment.experiment_id}>
                  {experiment.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Run</span>
            <select value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>
              <option value="">Select a run</option>
              {runsResource.data?.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.profile_label} · {run.run_id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Export directory</span>
            <input value={exportDir} onChange={(event) => setExportDir(event.target.value)} />
          </label>
        </div>

        <div className="detail-card">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Progress</p>
              <h3>Live monitor</h3>
            </div>
            <p className="table-count">{monitorResource.data?.progress.status_summary || actionStatus}</p>
          </div>
          <div className="progress-shell">
            <div
              className="progress-bar"
              style={{
                width: `${Math.round(((monitorResource.data?.progress.progress_fraction || 0) * 100))}%`,
              }}
            />
          </div>
          <div className="meta-grid">
            <div>
              <dt>Label</dt>
              <dd>{monitorResource.data?.progress.progress_label || "Waiting for selection"}</dd>
            </div>
            <div>
              <dt>Phase</dt>
              <dd>{monitorResource.data?.progress.current_phase_label || "—"}</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>{monitorResource.data?.progress.current_source_label || "—"}</dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd>{monitorResource.data?.run.status || "—"}</dd>
            </div>
          </div>
          <div className="chip-row">
            {Object.entries(monitorResource.data?.aggregates || {}).map(([key, value]) => (
              <span className="chip" key={key}>
                {key.replaceAll("_", " ")} {value}
              </span>
            ))}
          </div>
        </div>

        <div className="page-grid">
          <section className="panel panel-stack">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Run sources</p>
                <h3>Source queue</h3>
              </div>
            </div>
            <div className="table-wrap">
              <table className="reaction-table">
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>Status</th>
                    <th>Phase</th>
                    <th>Derived</th>
                  </tr>
                </thead>
                <tbody>
                  {monitorResource.data?.sources.map((source) => (
                    <tr
                      className={source.run_source_id === selectedSource ? "is-active-row" : ""}
                      key={source.run_source_id}
                      onClick={() => setSelectedSource(source.run_source_id)}
                    >
                      <td>{source.original_filename}</td>
                      <td>{source.status}</td>
                      <td>{source.current_phase}</td>
                      <td>
                        {source.completed_derived_images}/{source.expected_derived_images}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <label className="field">
              <span>Derived image</span>
              <select value={selectedDerived} onChange={(event) => setSelectedDerived(event.target.value)}>
                <option value="">Select a derived image</option>
                {sourceDetail?.derived_images.map((item) => (
                  <option key={item.derived_image_id} value={item.derived_image_id}>
                    {item.page_hint || "image"} · {item.derived_image_id}
                  </option>
                ))}
              </select>
            </label>

            <div className="action-row">
              <button
                className="button"
                disabled={!selectedRun}
                onClick={() => void runAction(() => api.retryFailed(selectedRun), "Retried failed images.")}
              >
                Retry failed
              </button>
              <button
                className="button"
                disabled={!selectedRun}
                onClick={() => void runAction(() => api.retryRedo(selectedRun), "Retried redo images.")}
              >
                Retry redo
              </button>
              <button
                className="button"
                disabled={!selectedRun}
                onClick={() => void runAction(() => api.reprocessRun(selectedRun), "Reprocessed run normalization.")}
              >
                Reprocess run
              </button>
            </div>
          </section>

          <section className="panel panel-stack">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Derived detail</p>
                <h3>Attempts and recovery</h3>
              </div>
            </div>
            {selectedDerivedImage ? (
              <>
                <div className="meta-grid">
                  <div>
                    <dt>Outcome</dt>
                    <dd>{selectedDerivedImage.outcome_class || "—"}</dd>
                  </div>
                  <div>
                    <dt>Normalization</dt>
                    <dd>{selectedDerivedImage.normalization_status || "—"}</dd>
                  </div>
                  <div>
                    <dt>Accepted</dt>
                    <dd>{selectedDerivedImage.accepted_reaction_count || 0}</dd>
                  </div>
                  <div>
                    <dt>Rejected</dt>
                    <dd>{selectedDerivedImage.rejected_reaction_count || 0}</dd>
                  </div>
                </div>
                <div className="action-row">
                  {["normal", "no_agents", "recovery"].map((mode) => (
                    <button
                      className="button"
                      key={mode}
                      onClick={() =>
                        void runAction(
                          () => api.retryDerivedImage(selectedDerivedImage.derived_image_id, mode),
                          `Retried ${selectedDerivedImage.derived_image_id} with ${mode}.`,
                        )
                      }
                    >
                      Retry {mode}
                    </button>
                  ))}
                  <button
                    className="button button-primary"
                    onClick={() =>
                      void runAction(
                        () => api.reprocessDerivedImage(selectedDerivedImage.derived_image_id),
                        `Reprocessed ${selectedDerivedImage.derived_image_id}.`,
                      )
                    }
                  >
                    Reprocess derived
                  </button>
                </div>
                <div className="table-wrap">
                  <table className="detail-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Trigger</th>
                        <th>Mode</th>
                        <th>Status</th>
                        <th>Failure</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(selectedDerivedImage.attempts || []).map((attempt) => (
                        <tr key={attempt.attempt_id}>
                          <td>{attempt.attempt_no}</td>
                          <td>{attempt.trigger}</td>
                          <td>{attempt.execution_mode}</td>
                          <td>
                            <span className={`chip ${statusTone(attempt.status)}`}>{attempt.status}</span>
                          </td>
                          <td>{attempt.failure_kind || attempt.error_summary || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p className="panel-kicker">Awaiting selection</p>
                <h3>No derived image selected</h3>
                <p>Pick a source and derived image to inspect attempts or trigger recovery actions.</p>
              </div>
            )}
          </section>
        </div>

        <div className="page-grid">
          <section className="panel panel-stack">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Logs</p>
                <h3>Run tail</h3>
              </div>
            </div>
            <pre className="json-block log-block">
              {monitorResource.data?.log_tail.formatted || "No log tail available."}
            </pre>
          </section>
          <section className="panel panel-stack">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Export</p>
                <h3>Parquet output</h3>
              </div>
            </div>
            <p className="hero-text">
              Exports stay explicit. The frontend only requests them; the backend
              remains responsible for writing parquet files on the server.
            </p>
            <button
              className="button button-primary"
              disabled={!selectedRun}
              onClick={() =>
                void api.exportRun(selectedRun, exportDir).then((response) => {
                  setActionStatus(JSON.stringify(response, null, 2));
                })
              }
            >
              Export selected run
            </button>
            <pre className="json-block">{actionStatus}</pre>
          </section>
        </div>
      </div>
    </section>
  );
}
