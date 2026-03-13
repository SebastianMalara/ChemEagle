import { ChangeEvent, useEffect, useState } from "react";
import { api } from "../api/client";
import type { RuntimeConfig, RuntimeDiagnostics } from "../api/types";

const modelScopes = [
  { id: "main", label: "Main model catalog" },
  { id: "ocr", label: "OCR model catalog" },
] as const;

function updateConfig(
  current: RuntimeConfig,
  event: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>,
) {
  const { name, value } = event.target;
  return { ...current, [name]: value };
}

export function SettingsView() {
  const [envPath, setEnvPath] = useState(".env.chemeagle");
  const [config, setConfig] = useState<RuntimeConfig>({});
  const [status, setStatus] = useState("Loading runtime config...");
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics | null>(null);
  const [catalogStatus, setCatalogStatus] = useState<Record<string, string>>({});

  useEffect(() => {
    async function load() {
      const response = await api.getConfig();
      setEnvPath(response.env_path);
      setConfig(response.values);
      setStatus("Loaded runtime config from the backend.");
    }

    void load().catch((error: Error) => {
      setStatus(error.message);
    });
  }, []);

  async function handleSave(persistToEnv: boolean) {
    const response = await api.saveConfig({
      env_path: envPath,
      values: config,
      persist_to_env: persistToEnv,
    });
    setEnvPath(response.env_path);
    setConfig(response.values);
    setStatus(
      persistToEnv
        ? `Saved configuration and persisted it to ${response.env_path}.`
        : "Applied configuration without writing to the env file.",
    );
  }

  async function handleRefresh(scope: "main" | "ocr") {
    const response = await api.refreshModels(scope, config);
    setCatalogStatus((current) => ({ ...current, [scope]: response.status }));
    if (scope === "main") {
      setConfig((current) => ({
        ...current,
        LLM_MODEL: response.selected_model,
      }));
      return;
    }
    setConfig((current) => ({
      ...current,
      OCR_LLM_MODEL: response.selected_model,
    }));
  }

  async function handlePreflight() {
    const response = await api.runPreflight(config);
    setDiagnostics(response);
    setStatus(
      response.blocking_errors.length
        ? `Preflight found ${response.blocking_errors.length} blocking issue(s).`
        : "Preflight passed without blocking issues.",
    );
  }

  return (
    <section className="page">
      <header className="page-hero">
        <div className="hero-copy">
          <p className="eyebrow">Settings</p>
          <h2>Runtime control</h2>
          <p className="hero-text">Profiles, OCR, storage, and preflight.</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span className="stat-label">Env file</span>
            <strong className="stat-value stat-value--small">{envPath}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Mode</span>
            <strong className="stat-value">{config.CHEMEAGLE_RUN_MODE || "cloud"}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Backend</span>
            <strong className="stat-value">{config.ARTIFACT_BACKEND || "filesystem"}</strong>
          </div>
        </div>
      </header>

      <div className="page-grid">
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Editable config</p>
              <h3>Runtime form</h3>
            </div>
            <p className="table-count">{status}</p>
          </div>

          <div className="form-grid">
            <label className="field field-span-2">
              <span>Env file path</span>
              <input value={envPath} onChange={(event) => setEnvPath(event.target.value)} />
            </label>
            <label className="field">
              <span>Run mode</span>
              <select
                name="CHEMEAGLE_RUN_MODE"
                value={config.CHEMEAGLE_RUN_MODE || "cloud"}
                onChange={(event) => setConfig(updateConfig(config, event))}
              >
                <option value="cloud">cloud</option>
                <option value="local_os">local_os</option>
              </select>
            </label>
            <label className="field">
              <span>Device</span>
              <select
                name="CHEMEAGLE_DEVICE"
                value={config.CHEMEAGLE_DEVICE || "auto"}
                onChange={(event) => setConfig(updateConfig(config, event))}
              >
                <option value="auto">auto</option>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
                <option value="metal">metal</option>
              </select>
            </label>
            <label className="field">
              <span>LLM provider</span>
              <input
                name="LLM_PROVIDER"
                value={config.LLM_PROVIDER || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
            <label className="field">
              <span>LLM model</span>
              <input
                name="LLM_MODEL"
                value={config.LLM_MODEL || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
            <label className="field field-span-2">
              <span>OpenAI API key</span>
              <input
                type="password"
                name="OPENAI_API_KEY"
                value={config.OPENAI_API_KEY || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
            <label className="field">
              <span>OCR backend</span>
              <select
                name="OCR_BACKEND"
                value={config.OCR_BACKEND || "auto"}
                onChange={(event) => setConfig(updateConfig(config, event))}
              >
                <option value="auto">auto</option>
                <option value="llm_vision">llm_vision</option>
                <option value="easyocr">easyocr</option>
                <option value="tesseract">tesseract</option>
              </select>
            </label>
            <label className="field">
              <span>OCR model</span>
              <input
                name="OCR_LLM_MODEL"
                value={config.OCR_LLM_MODEL || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
            <label className="field">
              <span>PDF model size</span>
              <select
                name="PDF_MODEL_SIZE"
                value={config.PDF_MODEL_SIZE || "large"}
                onChange={(event) => setConfig(updateConfig(config, event))}
              >
                <option value="base">base</option>
                <option value="large">large</option>
              </select>
            </label>
            <label className="field">
              <span>Artifact backend</span>
              <select
                name="ARTIFACT_BACKEND"
                value={config.ARTIFACT_BACKEND || "filesystem"}
                onChange={(event) => setConfig(updateConfig(config, event))}
              >
                <option value="filesystem">filesystem</option>
                <option value="minio">minio</option>
              </select>
            </label>
            <label className="field field-span-2">
              <span>Artifact root</span>
              <input
                name="ARTIFACT_FILESYSTEM_ROOT"
                value={config.ARTIFACT_FILESYSTEM_ROOT || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
            <label className="field field-span-2">
              <span>Review DB path</span>
              <input
                name="REVIEW_DB_PATH"
                value={config.REVIEW_DB_PATH || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
            <label className="field field-span-2">
              <span>OCR config</span>
              <textarea
                name="OCR_CONFIG"
                value={config.OCR_CONFIG || ""}
                onChange={(event) => setConfig(updateConfig(config, event))}
              />
            </label>
          </div>

          <div className="action-row">
            <button className="button" onClick={() => void handleSave(false)}>
              Apply only
            </button>
            <button className="button button-primary" onClick={() => void handleSave(true)}>
              Save to env
            </button>
            <button className="button" onClick={() => void handlePreflight()}>
              Run preflight
            </button>
          </div>
        </section>

        <section className="panel panel-stack">
          <div>
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Catalog refresh</p>
                <h3>Provider catalogs</h3>
              </div>
            </div>
            <div className="card-grid">
              {modelScopes.map((scope) => (
                <article className="detail-card" key={scope.id}>
                  <h4>{scope.label}</h4>
                  <p>{catalogStatus[scope.id] || "Manual entry is enabled. Refresh to fetch a catalog."}</p>
                  <button className="button" onClick={() => void handleRefresh(scope.id)}>
                    Refresh {scope.id}
                  </button>
                </article>
              ))}
            </div>
          </div>

          <div>
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Diagnostics</p>
                <h3>Runtime preflight</h3>
              </div>
            </div>
            {diagnostics ? (
              <div className="diagnostic-grid">
                <div className="detail-card">
                  <h4>Blocking</h4>
                  <ul className="chip-list">
                    {diagnostics.blocking_errors.length ? (
                      diagnostics.blocking_errors.map((item) => (
                        <li className="chip chip-warn" key={item}>
                          {item}
                        </li>
                      ))
                    ) : (
                      <li className="chip chip-ok">No blocking errors</li>
                    )}
                  </ul>
                </div>
                <div className="detail-card">
                  <h4>Warnings</h4>
                  <ul className="chip-list">
                    {diagnostics.warnings.length ? (
                      diagnostics.warnings.map((item) => (
                        <li className="chip" key={item}>
                          {item}
                        </li>
                      ))
                    ) : (
                      <li className="chip chip-ok">No warnings</li>
                    )}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <p className="panel-kicker">Waiting</p>
                <h3>No diagnostics yet</h3>
                <p>Run preflight to inspect model, OCR, runtime, and PDF readiness.</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </section>
  );
}
