/*
 * App shell (design D2, ADR-0020): a collapsible grouped sidebar, a top bar with the command
 * palette, backend status and theme, and the screens. Routing is the URL hash (services/router):
 * a refresh keeps the screen and `#/predict?acc=<accession>` reopens a cached sequence.
 * The model version and what its numbers are measured on come from GET /demo/version and live
 * in the status popover; when the backend is unreachable every screen says so above its content.
 */
import { useCallback, useEffect, useRef, useState, type ReactElement } from "react";
import { CircleAlert } from "lucide-react";
import { getCatalogEntry, getVersion } from "./services/demoApi";
import { useRoute, type Screen } from "./services/router";
import { CommandPalette, type OpenedEntry } from "./components/CommandPalette";
import { NEXT_THEME, START_COMMAND, navFor, type Theme } from "./components/nav";
import { Sidebar } from "./components/Sidebar";
import { CopyButton, type BackendStatus } from "./components/StatusPill";
import { TopBar } from "./components/TopBar";
import About from "./pages/About";
import Compare from "./pages/Compare";
import Evidence from "./pages/Evidence";
import Gallery from "./pages/Gallery";
import Home from "./pages/Home";
import Inspector from "./pages/Inspector";
import Scanner from "./pages/Scanner";
import Space from "./pages/Space";
import Triage from "./pages/Triage";
import Trust from "./pages/Trust";
import "./styles/tokens.css";
import "./styles/app.css";
import "./styles/shell.css";
import "./styles/pages.css";

interface Seed {
  sequence: string;
  group: string;
  id?: string;
  autorun?: boolean;
}

const COLLAPSED_KEY = "protloc.sidebar.collapsed";
/** While the backend is unreachable, check again this often. */
const RECHECK_MS = 15_000;

function readCollapsed(): boolean {
  try {
    return window.localStorage.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

function writeCollapsed(value: boolean): void {
  try {
    window.localStorage.setItem(COLLAPSED_KEY, value ? "1" : "0");
  } catch {
    // Storage can be refused (private window, blocked site data); the toggle still works.
  }
}

export default function App(): ReactElement {
  const [route, navigate] = useRoute();
  const [seed, setSeed] = useState<Seed | null>(null);
  const [batchSeed, setBatchSeed] = useState<{ fasta: string; group: string } | null>(null);
  const [version, setVersion] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [theme, setTheme] = useState<Theme>("system");
  const [collapsed, setCollapsed] = useState<boolean>(readCollapsed);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const mainRef = useRef<HTMLElement>(null);
  const firstRender = useRef(true);

  const checkBackend = useCallback(() => {
    setStatus("checking");
    getVersion()
      .then((v) => {
        setVersion(v);
        setStatus("online");
      })
      .catch(() => {
        setStatus("offline");
      });
  }, []);

  useEffect(() => {
    checkBackend();
  }, [checkBackend]);

  useEffect(() => {
    if (status !== "offline") return undefined;
    const timer = setInterval(checkBackend, RECHECK_MS);
    return (): void => {
      clearInterval(timer);
    };
  }, [status, checkBackend]);

  // "system" removes the attribute so the prefers-color-scheme rule in tokens.css decides.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const onKey = (ev: KeyboardEvent): void => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k") {
        ev.preventDefault();
        setPaletteOpen((o) => !o);
      } else if (ev.key === "Escape") {
        setDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return (): void => {
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  const { item, group } = navFor(route.screen);

  // Title, scroll and focus follow the route, so a keyboard or screen-reader user lands on
  // the new screen rather than on the link they pressed. Not on first load.
  useEffect(() => {
    document.title = `${item.label} · ProtLoc Fusion`;
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    window.scrollTo({ top: 0 });
    mainRef.current?.focus({ preventScroll: true });
  }, [item.label]);

  // A linked accession (#/predict?acc=…) is looked up once and run, unless it is already seeded.
  const acc = route.screen === "predict" ? route.params.acc : undefined;
  useEffect(() => {
    if (!acc || seed?.id === acc) return;
    let live = true;
    getCatalogEntry(acc)
      .then((e) => {
        if (live)
          setSeed({ sequence: e.sequence, group: e.organism_group, id: e.id, autorun: true });
      })
      .catch(() => {
        // The Predict screen stays empty; the palette and Catalog report lookup failures.
      });
    return (): void => {
      live = false;
    };
  }, [acc, seed?.id]);

  const go = (screen: Screen): void => {
    setDrawerOpen(false);
    navigate(screen);
  };

  const openEntry = (e: OpenedEntry): void => {
    setSeed({ sequence: e.sequence, group: e.group, id: e.id, autorun: true });
    navigate("predict", { acc: e.id });
  };

  // A cached sequence by accession: the route effect looks it up and runs it.
  const openAccession = (accession: string): void => {
    navigate("predict", { acc: accession });
  };

  const openIn = (next: Screen) => (sequence: string, grp: string) => {
    setSeed({ sequence, group: grp });
    navigate(next);
  };

  const toggleCollapsed = (): void => {
    setCollapsed((c) => {
      writeCollapsed(!c);
      return !c;
    });
  };

  return (
    <div className="app" data-collapsed={collapsed}>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Sidebar
        current={route.screen}
        collapsed={collapsed}
        open={drawerOpen}
        onNavigate={go}
        onToggle={toggleCollapsed}
      />

      {drawerOpen && (
        <div
          className="drawer-scrim"
          aria-hidden="true"
          onClick={() => {
            setDrawerOpen(false);
          }}
        />
      )}

      <div className="workspace">
        <TopBar
          group={group}
          title={item.label}
          status={status}
          version={version}
          theme={theme}
          onRetry={checkBackend}
          onTheme={() => {
            setTheme(NEXT_THEME[theme]);
          }}
          onSearch={() => {
            setPaletteOpen(true);
          }}
          onMenu={() => {
            setDrawerOpen(true);
          }}
        />

        <main className="content" id="main" ref={mainRef} tabIndex={-1}>
          {status === "offline" && (
            <div className="callout warn" role="status">
              <CircleAlert size={16} aria-hidden="true" />
              <div className="callout-body">
                <strong>Backend unreachable.</strong> Start it from <code>backend/</code>; this page
                checks again every {RECHECK_MS / 1000} s.
                <div className="code-line">
                  <code>{START_COMMAND}</code>
                  <CopyButton text={START_COMMAND} label="Copy start command" />
                </div>
              </div>
            </div>
          )}

          {route.screen === "home" && (
            <Home
              down={status === "offline"}
              onRun={(sequence, grp, id) => {
                setSeed({ sequence, group: grp, autorun: true, ...(id ? { id } : {}) });
                navigate("predict");
              }}
              onBatch={(fasta, grp) => {
                setBatchSeed({ fasta, group: grp });
                navigate("batch");
              }}
              onAbout={() => {
                navigate("about");
              }}
              onOpenRecent={(e) => {
                setSeed({
                  sequence: e.sequence,
                  group: e.group,
                  id: e.id,
                  autorun: e.kind === "predict",
                });
                navigate(e.kind === "scan" ? "scanner" : "predict");
              }}
            />
          )}
          {route.screen === "catalog" && <Gallery onOpen={openEntry} />}
          {route.screen === "batch" && <Triage seed={batchSeed} onOpen={openIn("predict")} />}
          {route.screen === "predict" && (
            <Inspector seed={seed} onScan={openIn("scanner")} onOpenAccession={openAccession} />
          )}
          {route.screen === "space" && <Space onOpenAccession={openAccession} />}
          {route.screen === "compare" && <Compare />}
          {route.screen === "evidence" && <Evidence />}
          {route.screen === "scanner" && <Scanner seed={seed} />}
          {route.screen === "trust" && <Trust />}
          {route.screen === "about" && <About version={version} down={status === "offline"} />}
        </main>
      </div>

      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        onNavigate={go}
        onOpenEntry={openEntry}
      />
    </div>
  );
}
