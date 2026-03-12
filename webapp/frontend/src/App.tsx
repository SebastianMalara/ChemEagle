import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { IngestView } from "./views/IngestView";
import { ReviewView } from "./views/ReviewView";
import { RunsView } from "./views/RunsView";
import { SettingsView } from "./views/SettingsView";

const navItems = [
  { to: "/review", label: "Review" },
  { to: "/runs", label: "Runs" },
  { to: "/ingest", label: "Ingest" },
  { to: "/settings", label: "Settings" },
];

export default function App() {
  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="rail-brand">
          <p className="eyebrow">ChemEagle Web</p>
          <h1>Editorial lab console for batch extraction and review.</h1>
        </div>
        <nav className="rail-nav" aria-label="Primary">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              className={({ isActive }) =>
                `rail-link${isActive ? " is-active" : ""}`
              }
              to={item.to}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="rail-note">
          <p className="panel-kicker">Default stack</p>
          <p>
            FastAPI serves the API. This SPA stays deliberately dense and
            inspectable, with review leading the visual language.
          </p>
        </div>
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
