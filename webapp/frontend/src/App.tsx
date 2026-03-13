import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { IngestView } from "./views/IngestView";
import { ReviewView } from "./views/ReviewView";
import { RunsView } from "./views/RunsView";
import { SettingsView } from "./views/SettingsView";

const themeOptions = [
  {
    id: "paper",
    label: "Field Notes",
    description: "Warm paper and oxide accents.",
    swatches: ["#f5f1e8", "#ae4c28", "#316f43"],
  },
  {
    id: "cobalt",
    label: "Cobalt Lab",
    description: "Cool brass, slate, and glass.",
    swatches: ["#edf3f7", "#225b72", "#c17b2c"],
  },
  {
    id: "ember",
    label: "Ember Archive",
    description: "Burnt sienna with archival reds.",
    swatches: ["#fbf1e6", "#9f3a24", "#764b1a"],
  },
  {
    id: "night",
    label: "Night Shift",
    description: "Charcoal console without neon sludge.",
    swatches: ["#12171d", "#7fd1c8", "#f0b35c"],
  },
  {
    id: "dracula",
    label: "Dracula",
    description: "The real purple-and-cyan night palette.",
    swatches: ["#282a36", "#bd93f9", "#8be9fd"],
  },
] as const;

const navItems = [
  { to: "/review", label: "Review", shortLabel: "Rv" },
  { to: "/runs", label: "Runs", shortLabel: "Rn" },
  { to: "/ingest", label: "Ingest", shortLabel: "Ig" },
  { to: "/settings", label: "Settings", shortLabel: "St" },
];

function readInitialTheme() {
  const saved = safeStorageGet("paperbreaker-theme");
  if (saved && themeOptions.some((theme) => theme.id === saved)) {
    return saved;
  }
  return "paper";
}

function readInitialRailCollapsed() {
  return safeStorageGet("paperbreaker-rail-collapsed") === "1";
}

function safeStorageGet(key: string) {
  if (typeof window === "undefined") {
    return null;
  }
  const storage = window.localStorage;
  if (!storage || typeof storage.getItem !== "function") {
    return null;
  }
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function safeStorageSet(key: string, value: string) {
  if (typeof window === "undefined") {
    return;
  }
  const storage = window.localStorage;
  if (!storage || typeof storage.setItem !== "function") {
    return;
  }
  try {
    storage.setItem(key, value);
  } catch {
    // Ignore storage failures in constrained or test environments.
  }
}

export default function App() {
  const [theme, setTheme] = useState<string>(readInitialTheme);
  const [railCollapsed, setRailCollapsed] = useState<boolean>(readInitialRailCollapsed);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    safeStorageSet("paperbreaker-theme", theme);
  }, [theme]);

  useEffect(() => {
    safeStorageSet("paperbreaker-rail-collapsed", railCollapsed ? "1" : "0");
  }, [railCollapsed]);

  return (
    <div className={`app-shell${railCollapsed ? " is-rail-collapsed" : ""}`}>
      <aside className="rail">
        <div className="rail-toolbar">
          <div className="rail-toolbar-brand">
            <span aria-label="Paperbreaker chick logo" className="rail-mark" role="img">
              🐣
            </span>
            {!railCollapsed ? <span className="rail-toolbar-wordmark">Paperbreaker</span> : null}
          </div>
          <button
            aria-label={railCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="rail-toggle"
            onClick={() => setRailCollapsed((current) => !current)}
            type="button"
          >
            {railCollapsed ? "›" : "‹"}
          </button>
        </div>
        <div className="rail-brand">
          <p className="eyebrow">{railCollapsed ? "PB" : "Reaction ops"}</p>
          {!railCollapsed ? (
            <>
              <h1>Reaction console.</h1>
              <p className="brand-copy">Break papers into reaction data.</p>
            </>
          ) : (
            <p className="brand-copy brand-copy-compact">PB</p>
          )}
        </div>
        <nav className="rail-nav" aria-label="Primary">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              aria-label={item.label}
              className={({ isActive }) =>
                `rail-link${isActive ? " is-active" : ""}`
              }
              title={railCollapsed ? item.label : undefined}
              to={item.to}
            >
              {railCollapsed ? item.shortLabel : item.label}
            </NavLink>
          ))}
        </nav>
        {!railCollapsed ? (
          <>
            <section className="theme-panel" aria-label="Theme selector">
              <div className="panel-header panel-header-stack">
                <div>
                  <p className="panel-kicker">Theme</p>
                  <h2>Theme presets</h2>
                </div>
                <p className="table-count">Local only</p>
              </div>
              <div className="theme-grid">
                {themeOptions.map((option) => (
                  <button
                    key={option.id}
                    className={`theme-card${theme === option.id ? " is-active" : ""}`}
                    onClick={() => setTheme(option.id)}
                    type="button"
                  >
                    <span className="theme-swatches" aria-hidden="true">
                      {option.swatches.map((swatch) => (
                        <span
                          key={swatch}
                          className="theme-swatch"
                          style={{ background: swatch }}
                        />
                      ))}
                    </span>
                    <span className="theme-label">{option.label}</span>
                    <span className="theme-description">{option.description}</span>
                  </button>
                ))}
              </div>
            </section>
            <div className="rail-note">
              <p className="panel-kicker">Signal</p>
              <div className="rail-note-strip">
                <span className="rail-signal">FastAPI</span>
                <span className="rail-signal">Polling</span>
                <span className="rail-signal">Artifacts</span>
              </div>
              <p className="rail-note-copy">Built for batch QA, not slide decks.</p>
            </div>
          </>
        ) : null}
      </aside>
      <main className="content-shell">
        <Routes>
          <Route path="/" element={<Navigate to="/review" replace />} />
          <Route path="/settings" element={<SettingsView />} />
          <Route path="/ingest" element={<IngestView />} />
          <Route path="/runs" element={<RunsView />} />
          <Route path="/review" element={<ReviewView />} />
        </Routes>
      </main>
    </div>
  );
}
