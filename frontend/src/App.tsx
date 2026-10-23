/*
 * App shell: sidebar nav + the screens. No router dependency — a union type
 * and useState, because this is five screens and a router is a decision
 * (CLAUDE.md: "a dependency that renders UI is a decision, and decisions become ADRs").
 * The banner states the model version and what its numbers are measured on, from the
 * backend (GET /demo/version): the trained heads are the served path since Phase 14.
 */
import { useEffect, useState } from "react";
import { getVersion } from "./services/demoApi";
import Compare from "./pages/Compare";
import Evidence from "./pages/Evidence";
import Gallery from "./pages/Gallery";
import Home from "./pages/Home";
import Inspector from "./pages/Inspector";
import Scanner from "./pages/Scanner";
import Triage from "./pages/Triage";
import Trust from "./pages/Trust";
import "./styles/tokens.css";
import "./styles/app.css";

type Theme = "system" | "light" | "dark";
const NEXT_THEME: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

type Screen =
  "home" | "gallery" | "triage" | "inspector" | "compare" | "evidence" | "scanner" | "trust";
interface Seed {
  sequence: string;
  group: string;
  id?: string;
  autorun?: boolean;
}
const NAV: [Screen, string, string][] = [
  ["home", "Home", "what it answers, and its limits"],
  ["gallery", "Gallery", "cached examples, in-sample"],
  ["triage", "Triage Board", "which proteins need a human"],
  ["inspector", "Inspector", "seven outputs, one sequence"],
  ["compare", "Compare", "two sequences side by side"],
  ["evidence", "Evidence Ledger", "model vs. homology"],
  ["scanner", "Mutation Scanner", "counterfactuals"],
  ["trust", "Trust Console", "calibration + abstention"],
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [seed, setSeed] = useState<Seed | null>(null);
  const [version, setVersion] = useState<Record<string, any> | null>(null);
  const [down, setDown] = useState(false);
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    getVersion()
      .then(setVersion)
      .catch(() => {
        setDown(true);
      });
  }, []);

  // "system" removes the attribute so the prefers-color-scheme rule in tokens.css decides.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);

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
        <button
          className="btn"
          style={{ marginTop: "auto" }}
          aria-label={`Theme: ${theme}. Switch to ${NEXT_THEME[theme]}.`}
          onClick={() => {
            setTheme(NEXT_THEME[theme]);
          }}
        >
          Theme: {theme}
        </button>
        <div
          className="prov"
          style={{ marginTop: "var(--space-3)", flexDirection: "column", gap: 2 }}
        >
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

        {screen === "home" && (
          <Home
            version={version}
            down={down}
            onStart={() => {
              setScreen("gallery");
            }}
            onOpenRecent={(e) => {
              setSeed({
                sequence: e.sequence,
                group: e.group,
                id: e.id,
                autorun: e.kind === "predict",
              });
              setScreen(e.kind === "scan" ? "scanner" : "inspector");
            }}
          />
        )}
        {screen === "gallery" && (
          <Gallery
            onOpen={(e) => {
              setSeed({ sequence: e.sequence, group: e.group, id: e.id, autorun: true });
              setScreen("inspector");
            }}
          />
        )}
        {screen === "triage" && <Triage onOpen={openIn("inspector")} />}
        {screen === "inspector" && <Inspector seed={seed} onScan={openIn("scanner")} />}
        {screen === "compare" && <Compare />}
        {screen === "evidence" && <Evidence />}
        {screen === "scanner" && <Scanner seed={seed} />}
        {screen === "trust" && <Trust />}
      </main>
    </div>
  );
}
