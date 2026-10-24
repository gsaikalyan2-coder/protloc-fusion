/*
 * Hash routing without a router dependency (ADR-0020 admits none). A screen and a few query
 * parameters live in `location.hash`, so a refresh keeps the screen, the back button works and a
 * catalog result can be linked: `#/predict?acc=P12345`. Pasted sequences stay out of the URL.
 */
import { useCallback, useEffect, useState } from "react";

export const SCREENS = [
  "home",
  "predict",
  "batch",
  "compare",
  "catalog",
  "evidence",
  "scanner",
  "space",
  "trust",
  "about",
] as const;
export type Screen = (typeof SCREENS)[number];
export type RouteParams = Record<string, string>;

export interface Route {
  screen: Screen;
  params: RouteParams;
}

function isScreen(name: string): name is Screen {
  return (SCREENS as readonly string[]).includes(name);
}

/** Read a route from a hash such as `#/predict?acc=P12345`. Anything unknown is Home. */
export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const cut = raw.indexOf("?");
  const path = cut === -1 ? raw : raw.slice(0, cut);
  const query = cut === -1 ? "" : raw.slice(cut + 1);
  const name = path.split("/")[0] ?? "";
  const params: RouteParams = {};
  new URLSearchParams(query).forEach((value, key) => {
    params[key] = value;
  });
  return { screen: isScreen(name) ? name : "home", params };
}

/** The hash for a screen; Home is `#/`. Parameters are written in a stable order. */
export function formatHash(screen: Screen, params: RouteParams = {}): string {
  const query = new URLSearchParams(
    Object.keys(params)
      .sort()
      .map((k): [string, string] => [k, params[k] ?? ""]),
  ).toString();
  const path = screen === "home" ? "#/" : `#/${screen}`;
  return query ? `${path}?${query}` : path;
}

export type Navigate = (screen: Screen, params?: RouteParams) => void;

/** The current route, and a function that changes it through the browser history. */
export function useRoute(): [Route, Navigate] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));

  useEffect(() => {
    const onChange = (): void => {
      setRoute(parseHash(window.location.hash));
    };
    window.addEventListener("hashchange", onChange);
    return (): void => {
      window.removeEventListener("hashchange", onChange);
    };
  }, []);

  const navigate = useCallback<Navigate>((screen, params = {}) => {
    const next = formatHash(screen, params);
    if (window.location.hash === next) {
      setRoute(parseHash(next));
    } else {
      window.location.hash = next;
    }
  }, []);

  return [route, navigate];
}
