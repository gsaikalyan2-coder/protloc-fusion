/*
 * Home — the launchpad (design D4). The product's promise is the first thing on the page: paste
 * one sequence, pick the organism group, run. Several FASTA records go to Batch instead. Below it,
 * the six outputs in one line each (full statements on About & limits) and the recent jobs.
 */
import { useMemo, useState, type FormEvent, type ReactElement } from "react";
import { ArrowRight, Layers } from "lucide-react";
import { RecentJobs } from "../components/RecentJobs";
import { ExamplePicker, GroupSegmented, useExamples } from "../components/ui";
import { parseInput } from "../services/fasta";
import type { HistoryEntry } from "../services/history";
import { OUTPUTS } from "./content";

export default function Home({
  down,
  onRun,
  onBatch,
  onAbout,
  onOpenRecent,
}: {
  down: boolean;
  onRun: (sequence: string, group: string, id: string | undefined) => void;
  onBatch: (fasta: string, group: string) => void;
  onAbout: () => void;
  onOpenRecent?: (e: {
    id: string;
    sequence: string;
    group: string;
    kind: HistoryEntry["kind"];
  }) => void;
}): ReactElement {
  const { examples } = useExamples();
  const [text, setText] = useState("");
  const [group, setGroup] = useState("eukaryote");
  const [picked, setPicked] = useState("");
  const parsed = useMemo(() => parseInput(text), [text]);
  const many = parsed.records > 1;

  const submit = (ev: FormEvent): void => {
    ev.preventDefault();
    if (many) onBatch(text.trim(), group);
    else if (parsed.sequence) onRun(parsed.sequence, group, picked || parsed.id);
  };

  return (
    <>
      <section className="hero">
        <p className="overline">Protein annotation · frozen ESM-2 150M · trained heads</p>
        <h1>One sequence in. Seven answers out.</h1>
        <p className="sub">
          Location, membrane type, sorting signal and cleavage site, GO function, homology evidence
          and a per-residue explanation, from one shared protein-language-model representation.
          Every answer says which engine produced it.
        </p>
      </section>

      <form className="launch card" onSubmit={submit} aria-label="Run a prediction">
        <label htmlFor="launch-input" className="launch-label">
          Protein sequence
          <span className="note">
            {parsed.records === 0
              ? "Paste residues or FASTA. Several records run as a batch."
              : many
                ? `${String(parsed.records)} FASTA records · runs as a batch`
                : `${String(parsed.sequence.length)} residues`}
          </span>
        </label>
        <textarea
          id="launch-input"
          value={text}
          placeholder={
            ">sp|P12345|EXAMPLE\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ…"
          }
          spellCheck={false}
          onChange={(ev) => {
            setText(ev.target.value);
            setPicked("");
          }}
        />
        <div className="launch-controls">
          <GroupSegmented value={group} onChange={setGroup} />
          <div className="row tight">
            <ExamplePicker
              examples={examples}
              value={picked}
              onPick={(e) => {
                setText(e.sequence);
                setGroup(e.organism_group);
                setPicked(e.id);
              }}
            />
            <button
              className="btn primary lg"
              type="submit"
              disabled={down || parsed.records === 0}
            >
              {many ? (
                <>
                  <Layers size={16} aria-hidden="true" /> Run batch
                </>
              ) : (
                <>
                  Predict <ArrowRight size={16} aria-hidden="true" />
                </>
              )}
            </button>
          </div>
        </div>
        {down && <p className="hint">Start the backend to run a prediction.</p>}
      </form>

      <section aria-labelledby="outputs-title" className="section">
        <div className="section-head">
          <h2 id="outputs-title">What comes back</h2>
          <button type="button" className="btn ghost sm" onClick={onAbout}>
            Limits &amp; privacy <ArrowRight size={14} aria-hidden="true" />
          </button>
        </div>
        <ul className="output-grid">
          {OUTPUTS.map((o) => (
            <li key={o.name} className="output-tile">
              <span className={`engine-dot ${o.engine}`} aria-hidden="true" />
              <strong>{o.name}</strong>
              <span className="hint">{o.short}</span>
              <span className="note">{o.source}</span>
            </li>
          ))}
        </ul>
      </section>

      {onOpenRecent && <RecentJobs onOpen={onOpenRecent} />}
    </>
  );
}
