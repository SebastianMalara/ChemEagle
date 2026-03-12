import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAsyncResource } from "../api/hooks";
import type { ReactionDetail, ReactionSummary, ReviewUpdate, RunSummary } from "../api/types";

type PreviewKind = "image" | "pdf";

interface PreviewState {
  kind: PreviewKind;
  title: string;
  url: string;
}

function safeText(value: unknown, fallback = "—") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function formatSearchBlob(reaction: ReactionSummary) {
  return [
    reaction.reaction_uid,
    reaction.reaction_id,
    reaction.original_filename,
    reaction.profile_label,
    reaction.run_id,
    reaction.review_status,
    reaction.structure_quality,
  ]
    .join(" ")
    .toLowerCase();
}

function inferPreviewKind(reference: string) {
  return reference.toLowerCase().includes(".pdf") ? "pdf" : "image";
}

function sourcePreviewUrl(detail: ReactionDetail | null) {
  if (!detail?.source_artifact_url) {
    return "";
  }
  const kind = inferPreviewKind(`${detail.original_filename || ""} ${detail.source_artifact_url}`);
  if (kind !== "pdf") {
    return detail.source_artifact_url;
  }

  const pageMatch = String(detail.page_hint || "").match(/(\d+)/);
  const fragments = ["toolbar=0", "navpanes=0", "view=FitH"];
  if (pageMatch) {
    fragments.unshift(`page=${pageMatch[1]}`);
  }
  return `${detail.source_artifact_url}#${fragments.join("&")}`;
}

function renderPreviewBody(url: string, kind: PreviewKind, title: string, className: string) {
  if (!url) {
    return <div className="image-empty">Preview unavailable</div>;
  }
  if (kind === "pdf") {
    return <iframe className={className} src={url} title={title} />;
  }
  return <img alt={title} className={className} src={url} />;
}

export function ReviewView() {
  const runsResource = useAsyncResource<RunSummary[]>(() => api.listRuns(), []);
  const [runFilter, setRunFilter] = useState("");
  const [reviewStatusFilter, setReviewStatusFilter] = useState("all");
  const [outcomeFilter, setOutcomeFilter] = useState("all");
  const [searchText, setSearchText] = useState("");
  const reactionsResource = useAsyncResource<ReactionSummary[]>(
    () => api.listReviewReactions(runFilter, reviewStatusFilter, outcomeFilter),
    [runFilter, reviewStatusFilter, outcomeFilter],
  );
  const [selectedReactionUid, setSelectedReactionUid] = useState("");
  const [detail, setDetail] = useState<ReactionDetail | null>(null);
  const [reviewStatus, setReviewStatus] = useState("unchecked");
  const [reviewNotes, setReviewNotes] = useState("");
  const [saveStatus, setSaveStatus] = useState("");
  const [preview, setPreview] = useState<PreviewState | null>(null);

  useEffect(() => {
    if (!reactionsResource.data?.length) {
      setSelectedReactionUid("");
      return;
    }
    if (!selectedReactionUid) {
      setSelectedReactionUid(reactionsResource.data[0].reaction_uid);
    }
  }, [reactionsResource.data, selectedReactionUid]);

  useEffect(() => {
    if (!selectedReactionUid) {
      setDetail(null);
      return;
    }

    void api
      .getReactionDetail(selectedReactionUid)
      .then((nextDetail) => {
        setDetail({
          ...nextDetail,
          source_artifact_url:
            nextDetail.source_artifact_url || api.artifactUrl(nextDetail.reaction_uid, "source"),
          derived_artifact_url:
            nextDetail.derived_artifact_url || api.artifactUrl(nextDetail.reaction_uid, "derived"),
          render_artifact_url:
            nextDetail.render_artifact_url || api.artifactUrl(nextDetail.reaction_uid, "render"),
        });
        setReviewStatus(nextDetail.review_status || "unchecked");
        setReviewNotes(nextDetail.review_notes || "");
      })
      .catch((error: Error) => {
        setSaveStatus(error.message);
      });
  }, [selectedReactionUid]);

  useEffect(() => {
    if (!preview) {
      return undefined;
    }

    function handleKeydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setPreview(null);
      }
    }

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [preview]);

  const needle = searchText.trim().toLowerCase();
  const visibleReactions = (reactionsResource.data || []).filter((reaction) => {
    if (!needle) {
      return true;
    }
    return formatSearchBlob(reaction).includes(needle);
  });

  const sourceCount = new Set(visibleReactions.map((reaction) => reaction.original_filename).filter(Boolean)).size;
  const sourceUrl = sourcePreviewUrl(detail);
  const sourceKind = inferPreviewKind(`${detail?.original_filename || ""} ${detail?.source_artifact_url || ""}`);

  async function saveReview() {
    if (!detail) {
      return;
    }
    const payload: ReviewUpdate = {
      reaction_uid: detail.reaction_uid,
      review_status: reviewStatus,
      review_notes: reviewNotes,
    };
    const saved = await api.updateReview(detail.reaction_uid, payload);
    setReviewStatus(saved.review_status);
    setReviewNotes(saved.review_notes);
    setSaveStatus(`Saved review state for ${saved.reaction_uid}.`);
    await reactionsResource.reload();
  }

  return (
    <section className="page">
      <header className="hero review-hero">
        <div className="hero-copy">
          <p className="eyebrow">Review</p>
          <h2>Review console</h2>
          <p className="hero-text">Scan the queue, verify structure, save the call.</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span className="stat-label">Reactions</span>
            <strong className="stat-value">{visibleReactions.length}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Runs</span>
            <strong className="stat-value">{new Set(visibleReactions.map((row) => row.run_id)).size}</strong>
          </div>
          <div className="stat-card">
            <span className="stat-label">Sources</span>
            <strong className="stat-value">{sourceCount}</strong>
          </div>
        </div>
      </header>

      <main className="layout review-layout">
        <section className="panel list-panel review-queue-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Queue</p>
              <h3>Reaction queue</h3>
            </div>
            <p className="table-count">{visibleReactions.length} visible</p>
          </div>

          <div className="filters review-filters">
            <label className="field">
              <span>Run</span>
              <select value={runFilter} onChange={(event) => setRunFilter(event.target.value)}>
                <option value="">All runs</option>
                {runsResource.data?.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.profile_label} · {run.run_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Status</span>
              <select value={reviewStatusFilter} onChange={(event) => setReviewStatusFilter(event.target.value)}>
                <option value="all">all</option>
                <option value="unchecked">unchecked</option>
                <option value="ok">ok</option>
                <option value="not_ok">not_ok</option>
              </select>
            </label>
            <label className="field">
              <span>Outcome</span>
              <select value={outcomeFilter} onChange={(event) => setOutcomeFilter(event.target.value)}>
                <option value="all">all</option>
                <option value="succeeded">succeeded</option>
                <option value="empty">empty</option>
                <option value="failed">failed</option>
                <option value="needs_redo">needs_redo</option>
                <option value="imported_without_artifact">imported_without_artifact</option>
              </select>
            </label>
            <label className="field field-search">
              <span>Search</span>
              <input
                type="search"
                placeholder="reaction id, source, label, smiles, notes"
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
              />
            </label>
          </div>

          <div className="queue-scroll">
            <div className="queue-list">
              {visibleReactions.map((reaction) => (
                <button
                  className={`queue-card${reaction.reaction_uid === selectedReactionUid ? " is-active" : ""}`}
                  key={reaction.reaction_uid}
                  onClick={() => setSelectedReactionUid(reaction.reaction_uid)}
                  type="button"
                >
                  <div className="queue-card-header">
                    <strong className="queue-card-title">{safeText(reaction.reaction_id)}</strong>
                    <span className="chip queue-chip">{safeText(reaction.review_status)}</span>
                  </div>
                  <p className="queue-card-source">{safeText(reaction.original_filename)}</p>
                  <p className="queue-card-subtle">{safeText(reaction.page_hint, "page/image unavailable")}</p>
                  <div className="queue-card-footer">
                    <span>{safeText(reaction.profile_label, "profile unavailable")}</span>
                    <span>{safeText(reaction.structure_quality)}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="panel detail-panel review-detail-panel">
          {!detail ? (
            <div className="empty-state">
              <p className="panel-kicker">Detail</p>
              <h3>Select a reaction</h3>
              <p>Pick a row to inspect images, molecules, and review state.</p>
            </div>
          ) : (
            <div className="detail-content">
              <div className="panel-header detail-header">
                <div>
                  <p className="panel-kicker">Reaction detail</p>
                  <h3>{safeText(detail.reaction_id)}</h3>
                </div>
                <div className="action-row">
                  {detail.source_artifact_url ? (
                    <a
                      className="source-link"
                      href={detail.source_artifact_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open source
                    </a>
                  ) : null}
                  {sourceUrl ? (
                    <button
                      className="button"
                      onClick={() => setPreview({ kind: sourceKind, title: "Source asset", url: sourceUrl })}
                      type="button"
                    >
                      Enlarge source
                    </button>
                  ) : null}
                </div>
              </div>

              <div className="detail-grid review-detail-grid">
                <section className="detail-card detail-card-wide preview-card preview-card-source">
                  <div className="detail-card-header">
                    <div>
                      <h4>Source frame</h4>
                      <p className="table-count">
                        {safeText(detail.original_filename)} · {safeText(detail.page_hint, "page unknown")}
                      </p>
                    </div>
                    {sourceUrl ? (
                      <button
                        className="button"
                        onClick={() => setPreview({ kind: sourceKind, title: "Source asset", url: sourceUrl })}
                        type="button"
                      >
                        Expand
                      </button>
                    ) : null}
                  </div>
                  <div className="image-frame image-frame-source">
                    {renderPreviewBody(sourceUrl, sourceKind, "Source asset", "artifact-preview")}
                  </div>
                </section>

                <div className="preview-grid detail-card-wide">
                  <section className="detail-card preview-card">
                    <div className="detail-card-header">
                      <h4>Original crop</h4>
                      {detail.derived_artifact_url ? (
                        <button
                          className="button"
                          onClick={() =>
                            setPreview({
                              kind: "image",
                              title: "Original crop",
                              url: detail.derived_artifact_url || "",
                            })
                          }
                          type="button"
                        >
                          Expand
                        </button>
                      ) : null}
                    </div>
                    <div className="image-frame image-frame-support">
                      {renderPreviewBody(
                        detail.derived_artifact_url || "",
                        "image",
                        "Derived crop",
                        "artifact-preview",
                      )}
                    </div>
                  </section>

                  <section className="detail-card preview-card">
                    <div className="detail-card-header">
                      <h4>RDKit render</h4>
                      {detail.render_artifact_url ? (
                        <button
                          className="button"
                          onClick={() =>
                            setPreview({
                              kind: "image",
                              title: "RDKit render",
                              url: detail.render_artifact_url || "",
                            })
                          }
                          type="button"
                        >
                          Expand
                        </button>
                      ) : null}
                    </div>
                    <div className="image-frame image-frame-support">
                      {renderPreviewBody(
                        detail.render_artifact_url || "",
                        "image",
                        "RDKit render",
                        "artifact-preview",
                      )}
                    </div>
                  </section>
                </div>

                <section className="detail-card detail-card-wide">
                  <h4>Metadata</h4>
                  <dl className="meta-grid">
                    <div>
                      <dt>Reaction UID</dt>
                      <dd>{detail.reaction_uid}</dd>
                    </div>
                    <div>
                      <dt>Run</dt>
                      <dd>{detail.run_id}</dd>
                    </div>
                    <div>
                      <dt>Source</dt>
                      <dd>{safeText(detail.original_filename)}</dd>
                    </div>
                    <div>
                      <dt>Outcome</dt>
                      <dd>{safeText(detail.outcome_class)}</dd>
                    </div>
                    <div>
                      <dt>Quality</dt>
                      <dd>{safeText(detail.structure_quality)}</dd>
                    </div>
                    <div>
                      <dt>Acceptance</dt>
                      <dd>{safeText(detail.acceptance_reason)}</dd>
                    </div>
                  </dl>
                </section>

                <section className="detail-card detail-card-wide">
                  <h4>Molecules</h4>
                  <div className="table-wrap compact-wrap">
                    <table className="detail-table">
                      <thead>
                        <tr>
                          <th>Side</th>
                          <th>#</th>
                          <th>Label</th>
                          <th>SMILES</th>
                          <th>Validation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(detail.molecules || []).map((molecule, index) => (
                          <tr key={`${molecule.side}-${index}`}>
                            <td>{safeText(molecule.side)}</td>
                            <td>{safeText(molecule.ordinal)}</td>
                            <td>{safeText(molecule.label)}</td>
                            <td>{safeText(molecule.smiles)}</td>
                            <td>{safeText(molecule.validation_kind || molecule.structure_quality)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="detail-card">
                  <h4>Conditions</h4>
                  <ul className="chip-list">
                    {(detail.conditions || []).map((condition, index) => (
                      <li className="chip" key={`${condition.condition_type}-${index}`}>
                        {safeText(condition.condition_type)}: {safeText(condition.value_text)}
                      </li>
                    ))}
                  </ul>
                </section>

                <section className="detail-card">
                  <h4>Additional info</h4>
                  <ul className="chip-list">
                    {(detail.additional_info || []).map((info, index) => (
                      <li className="chip" key={`${info.info_type}-${index}`}>
                        {safeText(info.info_type)}: {safeText(info.value_text)}
                      </li>
                    ))}
                  </ul>
                </section>

                <section className="detail-card detail-card-wide">
                  <h4>Review decision</h4>
                  <div className="action-row">
                    {["unchecked", "ok", "not_ok"].map((option) => (
                      <button
                        className={`button${reviewStatus === option ? " button-primary" : ""}`}
                        key={option}
                        onClick={() => setReviewStatus(option)}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                  <label className="field field-span-2">
                    <span>Review notes</span>
                    <textarea value={reviewNotes} onChange={(event) => setReviewNotes(event.target.value)} />
                  </label>
                  <div className="action-row">
                    <button className="button button-primary" onClick={() => void saveReview()}>
                      Save review state
                    </button>
                    <p className="table-count">{saveStatus}</p>
                  </div>
                </section>

                <details className="detail-card detail-card-wide detail-disclosure">
                  <summary>Reaction JSON</summary>
                  <pre className="json-block">{JSON.stringify(detail.raw_reaction_json, null, 2)}</pre>
                </details>
              </div>
            </div>
          )}
        </section>
      </main>

      {preview ? (
        <div className="lightbox-backdrop" onClick={() => setPreview(null)} role="presentation">
          <div className="lightbox-panel" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
            <div className="lightbox-header">
              <div>
                <p className="panel-kicker">Preview</p>
                <h3>{preview.title}</h3>
              </div>
              <button className="button" onClick={() => setPreview(null)} type="button">
                Close
              </button>
            </div>
            <div className="lightbox-body">
              {renderPreviewBody(preview.url, preview.kind, preview.title, "artifact-preview artifact-preview-lightbox")}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
