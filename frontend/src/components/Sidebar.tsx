/*
 * The sidebar: brand, three navigation groups and a collapse control. Collapsed, it keeps the
 * icons and moves each label into the button's accessible name and a native tooltip.
 */
import { useEffect, useRef, type ReactElement } from "react";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import type { Screen } from "../services/router";
import { NAV_GROUPS } from "./nav";

/** A cell outline with a nucleus: the product mark. Achromatic, drawn in currentColor. */
function Mark(): ReactElement {
  return (
    <svg className="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="2.5" y="2.5" width="19" height="19" rx="6" fill="none" stroke="currentColor" />
      <circle cx="14" cy="10" r="4" fill="currentColor" />
      <circle cx="8" cy="16.5" r="1.25" fill="currentColor" />
    </svg>
  );
}

export function Sidebar({
  current,
  collapsed,
  open = false,
  onNavigate,
  onToggle,
}: {
  current: Screen;
  collapsed: boolean;
  /** Narrow screens: the sidebar is a drawer, open or closed. */
  open?: boolean;
  onNavigate: (screen: Screen) => void;
  onToggle: () => void;
}): ReactElement {
  const ToggleIcon = collapsed ? PanelLeftOpen : PanelLeftClose;
  const ref = useRef<HTMLElement>(null);

  // When the drawer opens on a narrow screen, move focus into it.
  useEffect(() => {
    if (open) ref.current?.querySelector<HTMLElement>(".nav-item[aria-current='page']")?.focus();
  }, [open]);
  return (
    <aside className="sidebar" aria-label="Primary" data-open={open} ref={ref}>
      <a
        className="brand"
        href="#/"
        onClick={(ev) => {
          ev.preventDefault();
          onNavigate("home");
        }}
      >
        <Mark />
        <span className="brand-text">
          ProtLoc Fusion
          <small>trained heads · frozen ESM-2</small>
        </span>
      </a>

      <nav className="nav-groups">
        {NAV_GROUPS.map((g) => (
          <div className="nav-group" key={g.label ?? "top"}>
            {g.label && <div className="nav-label">{g.label}</div>}
            <ul>
              {g.items.map(({ screen, label, icon: Icon }) => (
                <li key={screen}>
                  <a
                    className="nav-item"
                    href={screen === "home" ? "#/" : `#/${screen}`}
                    aria-current={current === screen ? "page" : undefined}
                    aria-label={label}
                    title={collapsed ? label : undefined}
                    onClick={(ev) => {
                      ev.preventDefault();
                      onNavigate(screen);
                    }}
                  >
                    <Icon size={16} strokeWidth={1.75} aria-hidden="true" />
                    <span className="nav-text">{label}</span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <button
        type="button"
        className="nav-item sidebar-toggle"
        onClick={onToggle}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        aria-expanded={!collapsed}
        title={collapsed ? "Expand sidebar" : undefined}
      >
        <ToggleIcon size={16} strokeWidth={1.75} aria-hidden="true" />
        <span className="nav-text">Collapse</span>
      </button>
    </aside>
  );
}
