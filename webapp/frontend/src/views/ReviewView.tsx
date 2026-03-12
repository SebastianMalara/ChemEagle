import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAsyncResource } from "../api/hooks";
import type { ReactionDetail, ReactionSummary, ReviewUpdate, RunSummary } from "../api/types";

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

  const needle = searchText.trim().toLowerCase();
  const visibleReactions = (reactionsResource.data || []).filter((reaction) => {
    if (!needle) {
      return true;
    }
    return formatSearchBlob(reaction).includes(needle);
  });

  const sourceCount = new Set(visibleReactions.map((reaction) => reaction.original_filename).filter(Boolean)).size;

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
          <h2>Reaction review without setup friction.</h2>
          <p className="hero-text">
            This is the visual anchor for the new app: dense enough for batch QA,
            shaped like the deliverables bundle, but connected to live review mutations.
          </p>
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

      <main className="layout">
        <section className="panel list-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Filter</p>
              <h3>Reaction table</h3>
            </div>
            <p className="table-count">{visibleReactions.length} visible</p>
          </div>

          <div className="filters">
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

          <div className="table-wrap">
            <table className="reaction-table">
              <thead>
                <tr>
                  <th>Reaction</th>
                  <th>Source</th>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Quality</th>
                </tr>
              </thead>
              <tbody>
                {visibleReactions.map((reaction) => (
                  <tr
                    className={`reaction-row${reaction.reaction_uid === selectedReactionUid ? " is-active-row" : ""}`}
                    key={reaction.reaction_uid}
                    onClick={() => setSelectedReactionUid(reaction.reaction_uid)}
                  >
                    <td>
                      <div className="reaction-primary">
                        <strong>{safeText(reaction.reaction_id)}</strong>
                        <span className="row-subtle">{reaction.reaction_uid}</span>
                      </div>
                    </td>
                    <td>
                      <strong>{safeText(reaction.original_filename)}</strong>
                      <div className="row-subtle">{safeText(reaction.page_hint, "page/image unavailable")}</div>
                    </td>
                    <td>
                      <strong>{safeText(reaction.profile_label)}</strong>
                      <div className="row-subtle">{reaction.run_id}</div>
                    </td>
                    <td>
                      <span className="chip">{safeText(reaction.review_status)}</span>
                    </td>
                    <td>
                      <span className="chip">{safeText(reaction.structure_quality)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel detail-panel">
          {!detail ? (
            <div className="empty-state">
              <p className="panel-kicker">Detail</p>
              <h3>Select a reaction</h3>
              <p>Choose a row on the left to inspect metadata, images, molecules, conditions, and raw JSON.</p>
            </div>
          ) : (
            <div className="detail-content">
              <div className="panel-header detail-header">
                <div>
                  <p className="panel-kicker">Reaction detail</p>
                  <h3>{safeText(detail.reaction_id)}</h3>
                </div>
                <a
                  className="source-link"
                  href={detail.source_artifact_url || "#"}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open source asset
                </a>
              </div>

              <div className="detail-grid">
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

                <section className="detail-card">
                  <h4>Original crop</h4>
                  <div className="image-frame">
                    {detail.derived_artifact_url ? (
                      <img alt="Derived crop" src={detail.derived_artifact_url} />
                    ) : (
                      <div className="image-empty">No crop available</div>
                    )}
                  </div>
                </section>

                <section className="detail-card">
                  <h4>RDKit render</h4>
                  <div className="image-frame">
                    {detail.render_artifact_url ? (
                      <img alt="RDKit render" src={detail.render_artifact_url} />
                    ) : (
                      <div className="image-empty">No render available</div>
                    )}
                  </div>
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

                <section className="detail-card detail-card-wide">
                  <h4>Raw reaction JSON</h4>
                  <pre className="json-block">{JSON.stringify(detail.raw_reaction_json, null, 2)}</pre>
                </section>
              </div>
            </div>
          )}
        </section>
      </main>
    </section>
  );
}
