/*
 * The navigation model: three groups, one icon per screen. The sidebar, the top-bar breadcrumb
 * and the command palette all read this list, so a screen is named in one place.
 */
import {
  Columns2,
  Dna,
  House,
  Info,
  Layers,
  Library,
  Microscope,
  Orbit,
  Scale,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react";
import type { Screen } from "../services/router";

export interface NavItem {
  screen: Screen;
  label: string;
  /** One line for the command palette: what the screen answers. */
  hint: string;
  icon: LucideIcon;
}

export interface NavGroup {
  /** `null` for the ungrouped Home entry at the top. */
  label: string | null;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    label: null,
    items: [{ screen: "home", label: "Home", hint: "run a prediction", icon: House }],
  },
  {
    label: "Analyze",
    items: [
      {
        screen: "predict",
        label: "Predict",
        hint: "seven outputs, one sequence",
        icon: Microscope,
      },
      { screen: "batch", label: "Batch", hint: "which proteins need a human", icon: Layers },
      { screen: "compare", label: "Compare", hint: "two sequences side by side", icon: Columns2 },
    ],
  },
  {
    label: "Explore",
    items: [
      { screen: "catalog", label: "Catalog", hint: "all cached sequences", icon: Library },
      { screen: "evidence", label: "Evidence", hint: "model vs. homology", icon: Scale },
      { screen: "scanner", label: "Mutation scanner", hint: "counterfactuals", icon: Dna },
      { screen: "space", label: "Protein space", hint: "the cached sequences in 3D", icon: Orbit },
    ],
  },
  {
    label: "Model",
    items: [
      { screen: "trust", label: "Trust", hint: "calibration and abstention", icon: ShieldCheck },
      { screen: "about", label: "About & limits", hint: "what it claims, and privacy", icon: Info },
    ],
  },
];

export const NAV_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);

/** The item and its group label for a screen. Every screen has exactly one item. */
export function navFor(screen: Screen): { item: NavItem; group: string | null } {
  for (const g of NAV_GROUPS) {
    const item = g.items.find((i) => i.screen === screen);
    if (item) return { item, group: g.label };
  }
  throw new Error(`no navigation entry for screen "${screen}"`);
}

/** How to start the backend, shown wherever the frontend reports it unreachable. */
export const START_COMMAND = "uvicorn protloc.api.app_v2:app --port 8010";

export type Theme = "system" | "light" | "dark";
export const NEXT_THEME: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

export type PaletteOption =
  | { kind: "screen"; key: string; item: NavItem; group: string }
  | { kind: "lookup"; key: string; query: string };

/** A UniProt accession (P12345, A0A0B4J2F0) or entry name (INS_HUMAN); plain words are not. */
const ACCESSION_LIKE =
  /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}|[A-Z0-9]{1,10}_[A-Z0-9]{1,5})$/i;

export function paletteOptions(query: string): PaletteOption[] {
  const q = query.trim().toLowerCase();
  const screens: PaletteOption[] = NAV_GROUPS.flatMap((g) =>
    g.items
      .filter(
        (i) =>
          !q ||
          i.label.toLowerCase().includes(q) ||
          i.hint.toLowerCase().includes(q) ||
          (g.label ?? "").toLowerCase().includes(q),
      )
      .map((item): PaletteOption => ({
        kind: "screen",
        key: `screen:${item.screen}`,
        item,
        group: g.label ?? "Go to",
      })),
  );
  const raw = query.trim();
  const lookup: PaletteOption[] = ACCESSION_LIKE.test(raw)
    ? [{ kind: "lookup", key: `lookup:${raw}`, query: raw }]
    : [];
  return [...lookup, ...screens];
}
