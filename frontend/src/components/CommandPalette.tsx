/*
 * ⌘K / Ctrl+K: jump to a screen, or open a cached sequence by accession or entry name. One
 * input, one list, arrow keys and Enter; the list is a listbox driven by aria-activedescendant,
 * so focus never leaves the input. The lookup is GET /demo/catalog/{key}; a miss is shown
 * inline with the backend's own reason.
 */
import type { KeyboardEvent, ReactElement } from "react";
import { useEffect, useMemo, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { ArrowRight, FileSearch, Search } from "lucide-react";
import { getCatalogEntry } from "../services/demoApi";
import { describeError } from "../services/problems";
import type { Screen } from "../services/router";
import { paletteOptions, type PaletteOption } from "./nav";

export interface OpenedEntry {
  id: string;
  sequence: string;
  group: string;
}

export function CommandPalette({
  open,
  onOpenChange,
  onNavigate,
  onOpenEntry,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (screen: Screen) => void;
  onOpenEntry: (e: OpenedEntry) => void;
}): ReactElement {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const options = useMemo(() => paletteOptions(query), [query]);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setError(null);
      setPending(false);
    }
  }, [open]);

  useEffect(() => {
    setActive(0);
    setError(null);
  }, [query]);

  const choose = (o: PaletteOption | undefined): void => {
    if (!o || pending) return;
    if (o.kind === "screen") {
      onNavigate(o.item.screen);
      onOpenChange(false);
      return;
    }
    setPending(true);
    getCatalogEntry(o.query)
      .then((e) => {
        onOpenEntry({ id: e.id, sequence: e.sequence, group: e.organism_group });
        onOpenChange(false);
      })
      .catch((e: unknown) => {
        setError(describeError(e));
      })
      .finally(() => {
        setPending(false);
      });
  };

  const onKeyDown = (ev: KeyboardEvent<HTMLInputElement>): void => {
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      setActive((a) => (options.length ? (a + 1) % options.length : 0));
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      setActive((a) => (options.length ? (a - 1 + options.length) % options.length : 0));
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      choose(options[active]);
    }
  };

  const activeId = options[active] ? `palette-${options[active].key}` : undefined;
  let lastGroup = "";

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="scrim" />
        <Dialog.Content className="palette" aria-describedby={undefined}>
          <Dialog.Title className="sr-only">Search or jump to</Dialog.Title>
          <div className="palette-input">
            <Search size={16} aria-hidden="true" />
            <input
              autoFocus
              value={query}
              onChange={(ev) => {
                setQuery(ev.target.value);
              }}
              onKeyDown={onKeyDown}
              placeholder="Jump to a screen, or type an accession…"
              role="combobox"
              aria-expanded="true"
              aria-controls="palette-list"
              aria-activedescendant={activeId}
              aria-autocomplete="list"
              spellCheck={false}
            />
          </div>

          <ul className="palette-list" id="palette-list" role="listbox" aria-label="Results">
            {options.map((o, i) => {
              const group = o.kind === "lookup" ? "Open sequence" : o.group;
              const header = group !== lastGroup ? group : null;
              lastGroup = group;
              const Icon = o.kind === "lookup" ? FileSearch : o.item.icon;
              return (
                <li key={o.key} role="presentation">
                  {header && (
                    <div className="palette-group" role="presentation">
                      {header}
                    </div>
                  )}
                  <div
                    id={`palette-${o.key}`}
                    role="option"
                    aria-selected={i === active}
                    className="palette-option"
                    onMouseMove={() => {
                      setActive(i);
                    }}
                    onClick={() => {
                      choose(o);
                    }}
                  >
                    <Icon size={16} strokeWidth={1.75} aria-hidden="true" />
                    {o.kind === "lookup" ? (
                      <span>
                        Open <span className="mono">{o.query}</span> in Predict
                        {pending && <span className="muted"> · looking up…</span>}
                      </span>
                    ) : (
                      <span>
                        {o.item.label} <span className="muted">· {o.item.hint}</span>
                      </span>
                    )}
                    <ArrowRight className="palette-go" size={14} aria-hidden="true" />
                  </div>
                </li>
              );
            })}
            {options.length === 0 && <li className="palette-empty">No screen matches.</li>}
          </ul>

          {error && (
            <p className="palette-error" role="alert">
              {error}
            </p>
          )}

          <div className="palette-foot" aria-hidden="true">
            <span>
              <kbd>↑</kbd>
              <kbd>↓</kbd> move
            </span>
            <span>
              <kbd>↵</kbd> open
            </span>
            <span>
              <kbd>esc</kbd> close
            </span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
