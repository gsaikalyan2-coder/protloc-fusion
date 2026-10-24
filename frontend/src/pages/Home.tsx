/*
 * Home — what the tool answers, what it does not, and who can read a result.
 * Every limit that the backend enforces (cached-sequence count, served groups) is read from
 * GET /demo/version, so this page cannot drift from the server. Nothing here is a control
 * that does nothing: the one button opens the Inspector.
 */
import type { ReactElement } from "react";
import { RecentJobs } from "../components/RecentJobs";
import { Card } from "../components/ui";
import type { HistoryEntry } from "../services/history";

interface GroupInfo {
  label: string;
  served: boolean;
}

/** The parts of GET /demo/version this page reads. */
interface VersionInfo {
  organism_groups?: Record<string, GroupInfo>;
  limits?: string[];
}

const OUTPUTS: { name: string; source: string; says: string }[] = [
  {
    name: "Subcellular location (multi-label)",
    source: "trained head · calibrated per label",
    says: "Which compartments the head scores above each label's own threshold. If none clears its threshold the page says so instead of showing the top guess.",
  },
  {
    name: "Membrane type",
    source: "motif engine · not a trained model",
    says: "A rule-based call from hydropathy and motif features. It is labelled as such wherever it appears.",
  },
  {
    name: "Signal peptide and cleavage site",
    source: "trained CRF head",
    says: "Per-residue posterior for an N-terminal signal region and a predicted cleavage position. Transit peptides are not predicted.",
  },
  {
    name: "GO molecular function",
    source: "trained head · calibrated per term",
    says: "A small vocabulary of frequent terms only. Terms from homologous hits are shown separately and never merged with these.",
  },
  {
    name: "Homology evidence",
    source: "MMseqs2 search against Swiss-Prot",
    says: "Nearest Swiss-Prot hits and the GO terms they carry. Evidence is shown beside the model and never feeds a score.",
  },
  {
    name: "Per-residue explanation",
    source: "computed from the trained localization head",
    says: "Shows which residues the head's output depends on (occlusion, pooling attention, signal posterior). It is faithfulness to the head, not a claim about biology.",
  },
];

export default function Home({
  version,
  down,
  onStart,
  onOpenRecent,
}: {
  version: VersionInfo | null;
  down: boolean;
  onStart: () => void;
  onOpenRecent?: (e: {
    id: string;
    sequence: string;
    group: string;
    kind: HistoryEntry["kind"];
  }) => void;
}): ReactElement {
  const groups = version?.organism_groups ?? {};
  const served = Object.values(groups)
    .filter((g) => g.served)
    .map((g) => g.label);
  const refused = Object.values(groups)
    .filter((g) => !g.served)
    .map((g) => g.label);
  const limits = version?.limits ?? [];

  return (
    <div>
      <div className="head">
        <div>
          <h1>ProtLoc Fusion</h1>
          <p className="sub">
            One protein sequence and one organism-group choice in; location, membrane type, sorting
            signal, GO function, homology evidence and a per-residue explanation out, from one
            shared protein-language-model representation.
          </p>
        </div>
        <button className="btn primary" onClick={onStart}>
          Try an example
        </button>
      </div>

      {onOpenRecent && <RecentJobs onOpen={onOpenRecent} />}

      <Card title="What it answers">
        <ul className="plain">
          {OUTPUTS.map((o) => (
            <li key={o.name}>
              <strong>{o.name}</strong> <span className="muted">· {o.source}</span>
              <div className="sub">{o.says}</div>
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Limits, stated plainly">
        {down && <p className="sub">The backend is unreachable, so live limits cannot be shown.</p>}
        <ul className="plain">
          {limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
          <li>
            Organism groups served: {served.length > 0 ? served.join(", ") : "—"}.{" "}
            {refused.length > 0 && <>Refused with an error, not guessed: {refused.join(", ")}.</>}
          </li>
          <li>
            Gallery sequences are real Swiss-Prot entries that the heads were trained or tuned on,
            so their results are in-sample and flatter the model.
          </li>
          <li>
            Metrics are for the subset the heads were evaluated on and are not comparable to DeepLoc
            or to the Phase 8 full-corpus numbers.
          </li>
        </ul>
      </Card>

      <Card title="Privacy">
        <ul className="plain">
          <li>
            A result lives at a URL that contains a random job id. The id is hard to guess but the
            URL is not access-controlled: anyone who has the link can read the result. Do not submit
            sequences you need to keep confidential.
          </li>
          <li>
            Results expire after a short retention window. An expired result answers 410 Gone and
            the page says so; it is never shown as an empty result.
          </li>
          <li>
            The Recent jobs list lives only in this browser&apos;s local storage. It keeps job ids,
            labels and, for single sequences, the sequence itself; Clear removes it.
          </li>
          <li>
            There are no accounts and no irreversible or financial actions anywhere in this tool.
          </li>
        </ul>
      </Card>
    </div>
  );
}
