/*
 * Backend status in the top bar. The pill says one word; its popover carries what the old
 * permanent banner said — the model version, what the numbers are measured on, the served
 * limits — and, when the backend is down, the command that starts it.
 */
import type { ReactElement } from "react";
import { useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { Check, Copy, RefreshCw } from "lucide-react";
import { START_COMMAND } from "./nav";

export type BackendStatus = "checking" | "online" | "offline";

const WORD: Record<BackendStatus, string> = {
  checking: "Connecting",
  online: "Online",
  offline: "Offline",
};

/** Copies text, and says so for two seconds. Clipboard access can be refused; that is shown. */
export function CopyButton({ text, label }: { text: string; label: string }): ReactElement {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const copy = (): void => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setState("copied");
      })
      .catch(() => {
        setState("failed");
      })
      .finally(() => {
        setTimeout(() => {
          setState("idle");
        }, 2000);
      });
  };
  return (
    <button type="button" className="icon-btn" onClick={copy} aria-label={label} title={label}>
      {state === "copied" ? (
        <Check size={14} aria-hidden="true" />
      ) : (
        <Copy size={14} aria-hidden="true" />
      )}
      <span className="sr-only" aria-live="polite">
        {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : ""}
      </span>
    </button>
  );
}

function asText(v: unknown): string {
  return typeof v === "string" || typeof v === "number" ? String(v) : "—";
}

export function StatusPill({
  status,
  version,
  onRetry,
}: {
  status: BackendStatus;
  version: Record<string, unknown> | null;
  onRetry: () => void;
}): ReactElement {
  const limits = Array.isArray(version?.limits) ? (version.limits as unknown[]) : [];
  const rows: [string, unknown][] = [
    ["model", version?.model_version],
    ["config", version?.config_hash],
    ["rules", version?.rules_version],
    ["thresholds", version?.threshold_policy],
  ];

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          className="status-pill"
          data-status={status}
          aria-label={`Backend ${WORD[status].toLowerCase()}. Show details.`}
        >
          <span className="status-dot" aria-hidden="true" />
          {WORD[status]}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="popover status-popover" align="end" sideOffset={8}>
          <div className="popover-head">
            <span className="status-dot" data-status={status} aria-hidden="true" />
            <strong>Backend {WORD[status].toLowerCase()}</strong>
            <button
              type="button"
              className="icon-btn"
              onClick={onRetry}
              aria-label="Check again"
              title="Check again"
              disabled={status === "checking"}
            >
              <RefreshCw size={14} aria-hidden="true" />
            </button>
          </div>

          {status === "offline" ? (
            <>
              <p className="popover-text">
                Start it from <code>backend/</code>. It refuses to start, and says why, if the model
                registry does not verify.
              </p>
              <div className="code-line">
                <code>{START_COMMAND}</code>
                <CopyButton text={START_COMMAND} label="Copy start command" />
              </div>
            </>
          ) : (
            <>
              <dl className="kv">
                {rows.map(([k, v]) => (
                  <div key={k}>
                    <dt>{k}</dt>
                    <dd className="mono">
                      {/* The composite model version is four bundle versions joined by "+". */}
                      {asText(v)
                        .split("+")
                        .map((part, i) => (
                          <span key={part} className="kv-part">
                            {i > 0 && "+"}
                            {part}
                          </span>
                        ))}
                    </dd>
                  </div>
                ))}
              </dl>
              {typeof version?.banner === "string" && (
                <p className="popover-text">{version.banner}</p>
              )}
              {limits.length > 0 && (
                <ul className="popover-list">
                  {limits.map((l) => (
                    <li key={asText(l)}>{asText(l)}</li>
                  ))}
                </ul>
              )}
            </>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
