/*
 * App shell: sidebar nav + the five screens. No router dependency — a union type
 * and useState, because this is five screens and a router is a decision
 * (CLAUDE.md: "a dependency that renders UI is a decision, and decisions become ADRs").
 * The banner states the model version and what its numbers are measured on, from the
 * backend (GET /demo/version): the trained heads are the served path since Phase 14.
 */
import { useEffect, useState } from "react";
import { getVersion } from "./services/demoApi";
import Evidence from "./pages/Evidence";
import Inspector from "./pages/Inspector";
import Scanner from "./pages/Scanner";
import Triage from "./pages/Triage";
import Trust from "./pages/Trust";
import "./styles/tokens.css";
import "./styles/app.css";

type Screen = "triage" | "inspector" | "evidence" | "scanner" | "trust";
const NAV: [Screen, string, string][] = [
  ["triage", "Triage Board", "which proteins need a human"],
  ["inspector", "Inspector", "seven outputs, one sequence"],
  ["evidence", "Evidence Ledger", "model vs. homology"],
  ["scanner", "Mutation Scanner", "counterfactuals"],
  ["trust", "Trust Console", "calibration + abstention"],
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("triage");
  const [seed, setSeed] = useState<{ sequence: string; group: string } | null>(null);
  const [version, setVersion] = useState<Record<string, any> | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    getVersion()
      .then(setVersion)
      .catch(() => {
        setDown(true);
      });
  }, []);

  const openIn = (next: Screen) => (sequence: string, group: string) => {
    setSeed({ sequence, group });
    setScreen(next);
  };

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          ProtLoc Fusion<span>trained heads · frozen ESM-2</span>
        </div>
        <nav className="nav">
          {NAV.map(([id, label, hint]) => (
            <button
              key={id}
              onClick={() => {
                setScreen(id);
              }}
              aria-current={screen === id}
            >
              {label}
              <small>{hint}</small>
            </button>
          ))}
        </nav>
        <div className="prov" style={{ marginTop: "auto", flexDirection: "column", gap: 2 }}>
          <span>model {version?.model_version ?? "—"}</span>
          <span>config {version?.config_hash ?? "—"}</span>
          <span>rules {version?.rules_version ?? "—"}</span>
          <span>{version?.threshold_policy ?? "—"} thresholds</span>
        </div>
      </aside>

      <main className="main">
        {down && (
          <div className="banner">
            Backend unreachable on /demo. Start it from <span className="mono">backend/</span> with{" "}
            <span className="mono">uvicorn protloc.api.app_v2:app --port 8010</span>; it refuses to
            start, and says why, if the model registry does not verify.
          </div>
        )}
        {!down && version && (
          <div className="banner">
            {version.banner}
            {Array.isArray(version.limits) && (
              <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                {(version.limits as string[]).map((l) => (
                  <li key={l}>{l}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {screen === "triage" && <Triage onOpen={openIn("inspector")} />}
        {screen === "inspector" && <Inspector seed={seed} onScan={openIn("scanner")} />}
        {screen === "evidence" && <Evidence />}
        {screen === "scanner" && <Scanner seed={seed} />}
        {screen === "trust" && <Trust />}
      </main>
    </div>
  );
}
