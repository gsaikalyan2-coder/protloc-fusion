/*
 * The top bar: where you are, the palette trigger, backend status and the theme switch.
 */
import type { ReactElement } from "react";
import { ChevronRight, Menu, Monitor, Moon, Search, Sun } from "lucide-react";
import { NEXT_THEME, type Theme } from "./nav";
import { StatusPill, type BackendStatus } from "./StatusPill";

const THEME_ICON = { system: Monitor, light: Sun, dark: Moon } as const;

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent);

export function TopBar({
  group,
  title,
  status,
  version,
  theme,
  onRetry,
  onTheme,
  onSearch,
  onMenu,
}: {
  group: string | null;
  title: string;
  status: BackendStatus;
  version: Record<string, unknown> | null;
  theme: Theme;
  onRetry: () => void;
  onTheme: () => void;
  onSearch: () => void;
  /** Opens the navigation drawer; the button is shown on narrow screens only. */
  onMenu: () => void;
}): ReactElement {
  const ThemeIcon = THEME_ICON[theme];
  return (
    <header className="topbar">
      <button
        type="button"
        className="icon-btn menu-btn"
        onClick={onMenu}
        aria-label="Open navigation"
      >
        <Menu size={18} strokeWidth={1.75} aria-hidden="true" />
      </button>
      <div className="crumbs">
        {group && (
          <>
            <span className="crumb-group">{group}</span>
            <ChevronRight size={14} aria-hidden="true" className="crumb-sep" />
          </>
        )}
        <span className="crumb-page">{title}</span>
      </div>

      <div className="topbar-actions">
        <button type="button" className="search-trigger" onClick={onSearch}>
          <Search size={14} aria-hidden="true" />
          <span className="search-text">Search or jump to…</span>
          <kbd aria-label={IS_MAC ? "Command K" : "Control K"}>{IS_MAC ? "⌘K" : "Ctrl K"}</kbd>
        </button>
        <StatusPill status={status} version={version} onRetry={onRetry} />
        <button
          type="button"
          className="icon-btn"
          onClick={onTheme}
          aria-label={`Theme: ${theme}. Switch to ${NEXT_THEME[theme]}.`}
          title={`Theme: ${theme}`}
        >
          <ThemeIcon size={16} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
