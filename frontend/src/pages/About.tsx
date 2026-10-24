/*
 * About & limits — what each output is, what the tool does not claim, and what happens to a
 * submitted sequence. Every limit the backend enforces is read from GET /demo/version, so this
 * page cannot drift from the server.
 */
import type { ReactElement } from "react";
import { Card } from "../components/ui";
import { OUTPUTS } from "./content";

interface GroupInfo {
  label: string;
  served: boolean;
}

export default function About({
  version,
  down,
}: {
  /** GET /demo/version, as loaded by the shell. */
  version: Record<string, unknown> | null;
  down: boolean;
}): ReactElement {
  const groups = Object.values((version?.organism_groups ?? {}) as Record<string, GroupInfo>);
  const served = groups.filter((g) => g.served).map((g) => g.label);
  const refused = groups.filter((g) => !g.served).map((g) => g.label);
  const limits = Array.isArray(version?.limits) ? (version.limits as string[]) : [];

  return (
    <>
      <div className="head">
        <div>
          <h1>About &amp; limits</h1>
          <p className="sub">
            What each output is, which engine produces it, and what this tool does not claim.
          </p>
        </div>
      </div>

      <Card title="What it answers">
        <dl className="spec-list">
          {OUTPUTS.map((o) => (
            <div key={o.name}>
              <dt>
                {o.name} <span className={`badge ${o.engine}`}>{o.source}</span>
              </dt>
              <dd>{o.says}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <div className="grid two">
        <Card title="Limits, stated plainly">
          {down && (
            <p className="hint">The backend is unreachable, so live limits cannot be shown.</p>
          )}
          {typeof version?.banner === "string" && <p className="sub">{version.banner}</p>}
          <ul className="plain">
            {limits.map((l) => (
              <li key={l}>{l}</li>
            ))}
            <li>
              Organism groups served: {served.length > 0 ? served.join(", ") : "—"}.{" "}
              {refused.length > 0 && <>Refused with an error, not guessed: {refused.join(", ")}.</>}
            </li>
            <li>
              The example sequences are real Swiss-Prot entries the heads were trained or tuned on,
              so their results are in-sample and flatter the model. In the full catalog, partitions
              1–4 are in-sample and partition 5 is the held-out test set.
            </li>
            <li>
              Metrics are for the subset the heads were evaluated on and are not comparable to
              DeepLoc or to the Phase 8 full-corpus numbers.
            </li>
          </ul>
        </Card>

        <Card title="Privacy">
          <ul className="plain">
            <li>
              A result lives at a URL that contains a random job id. The id is hard to guess but the
              URL is not access-controlled: anyone who has the link can read the result. Do not
              submit sequences you need to keep confidential.
            </li>
            <li>
              Results expire after a short retention window. An expired result answers 410 Gone and
              the page says so; it is never shown as an empty result.
            </li>
            <li>
              The Recent jobs list lives only in this browser&apos;s local storage. It keeps job
              ids, labels and, for single sequences, the sequence itself; Clear removes it.
            </li>
            <li>
              There are no accounts and no irreversible or financial actions anywhere in this tool.
            </li>
          </ul>
        </Card>
      </div>
    </>
  );
}
