/*
 * Predict (design D5; was SCREEN 2, the Evidence-linked Inspector)
 * One sequence, the annotation outputs on one surface: calibrated multi-label localization
 * with each label's own threshold marker (Phase 10), membrane type as a chip modifier, the
 * signal peptide and its cleavage coordinate (Phase 11 CRF), GO function (Phase 12 head),
 * per-residue explanation, consistency flags, and the homology verdict. Every card says which
 * engine produced it: model or motif. POST /demo/predict, then poll (ADR-0001).
 *
 * Layout: input, then a result header and a five-cell summary that answers the headline
 * questions, the per-residue track at full width, the detail cards, and the secondary material
 * (structure, consistency, engines, export) in tabs.
 */
import { useEffect, useState, type ReactElement, type ReactNode } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { ArrowRight, Microscope, Printer } from "lucide-react";
import { predict, useJob, useSpace, type AttributionTrack, type Result } from "../services/demoApi";
import { Neighbours, ProteinSpace } from "../components/ProteinSpace";
import { SequenceTrack } from "../components/SequenceTrack";
import { DownloadButtons } from "../components/DownloadButtons";
import { PrintReport } from "../components/PrintReport";
import { SourceSummary } from "../components/SourceSummary";
import { StructurePanel } from "../components/StructurePanel";
import {
  Card,
  Chip,
  ConfidenceRow,
  EmptyState,
  Flags,
  JobState,
  ProblemNotice,
  ProvenanceBadge,
  ProvenanceBar,
  Segmented,
  SequenceForm,
  Skeleton,
  useExamples,
} from "../components/ui";

type Mode = "attribution" | "attention" | "signal" | "hydropathy";

/** The track a mode draws, so its claim can be printed beside it. Hydropathy has none. */
function trackFor(mode: Mode, r: Result): AttributionTrack | undefined {
  if (mode === "attribution") return r.attribution_tracks[r.attribution_method];
  if (mode === "attention") return r.attribution_tracks.pooling_attention;
  if (mode === "signal") return r.signal_track;
  return undefined;
}

const pretty = (s: string): string => s.replaceAll("_", " ");

function Stat({
  label,
  children,
  detail,
}: {
  label: string;
  children: ReactNode;
  detail?: ReactNode;
}): ReactElement {
  return (
    <div className="stat">
      <div className="overline">{label}</div>
      <div className="stat-value">{children}</div>
      {detail && <div className="note">{detail}</div>}
    </div>
  );
}

/** The headline answers, one cell each, before any detail. */
function Summary({ r }: { r: Result }): ReactElement {
  const top = r.localization.find((l) => l.code === r.top_location);
  const predicted = r.localization.filter((l) => l.predicted);
  const membrane = Object.entries(r.membrane_type).sort((a, b) => b[1] - a[1])[0];
  const go = r.go_function.filter((g) => g.predicted).sort((a, b) => b.probability - a.probability);
  const sig = r.sorting_signal;
  const ev = r.evidence_agreement;
  return (
    <div className="summary">
      <Stat
        label="Location"
        detail={
          r.no_confident_call
            ? `highest ${r.top_location} ${top ? top.probability.toFixed(2) : ""}, not asserted`
            : predicted.length > 1
              ? `${String(predicted.length)} labels clear their thresholds`
              : `p ${top ? top.probability.toFixed(2) : "—"}`
        }
      >
        {r.no_confident_call ? (
          <span className="stat-uncertain">No confident call</span>
        ) : (
          <span className="row tight">
            {predicted.map((l) => (
              <Chip key={l.code} code={l.code} name={l.name} />
            ))}
            <span>{predicted.map((l) => l.name).join(", ")}</span>
          </span>
        )}
      </Stat>
      <Stat label="Membrane" detail={membrane ? `p ${membrane[1].toFixed(2)} · motif engine` : ""}>
        {membrane ? pretty(membrane[0]) : "—"}
      </Stat>
      <Stat
        label="Sorting signal"
        detail={
          sig.cleavage_site !== null
            ? `cleaves after residue ${String(sig.cleavage_site)}`
            : `p ${sig.probability.toFixed(2)}`
        }
      >
        {pretty(sig.type)}
      </Stat>
      <Stat
        label="GO function"
        detail={
          go[0]
            ? `${go[0].go_id} · p ${go[0].probability.toFixed(2)}`
            : "no term reaches its threshold"
        }
      >
        {go[0]?.name ?? "—"}
      </Stat>
      <Stat
        label="Homology"
        detail={
          ev
            ? `${ev.top_hit_accession} · ${(ev.identity * 100).toFixed(1)}% identity`
            : r.evidence_status.available
              ? "no close hit"
              : "evidence unavailable"
        }
      >
        {ev ? (ev.agrees ? "Agrees with the model" : "Differs from the model") : "—"}
      </Stat>
    </div>
  );
}

/** Where this sequence sits among the cached ones, with its nearest neighbours. */
function SpaceCard({
  result,
  onOpen,
}: {
  result: Result;
  onOpen?: ((accession: string) => void) | undefined;
}): ReactElement {
  const state = useSpace();
  return (
    <Card title="Protein space" aside={<span className="badge">orientation only</span>}>
      {state.status === "loading" && <Skeleton lines={4} label="Loading the protein-space map" />}
      {state.status === "error" && <ProblemNotice problem={state.problem} />}
      {state.status === "ready" && (
        <div className="space-grid">
          <ProteinSpace space={state.data} focusSha={result.sequence_sha256} onOpen={onOpen} />
          <Neighbours space={state.data} focusSha={result.sequence_sha256} onOpen={onOpen} />
        </div>
      )}
    </Card>
  );
}

export default function Inspector({
  seed,
  onScan,
  onOpenAccession,
}: {
  seed: { sequence: string; group: string; id?: string; autorun?: boolean } | null;
  onScan: (seq: string, group: string) => void;
  /** Open another cached sequence in Predict, by accession. */
  onOpenAccession?: (accession: string) => void;
}) {
  const { examples } = useExamples();
  const [seq, setSeq] = useState(seed?.sequence ?? "");
  const [group, setGroup] = useState(seed?.group ?? "eukaryote");
  const [mode, setMode] = useState<Mode>("attribution");
  const [cursor, setCursor] = useState(0);
  const { run, retry, job, done, error, accepted, busy } = useJob();

  useEffect(() => {
    if (seed) {
      setSeq(seed.sequence);
      setGroup(seed.group);
      if (seed.autorun) void run(...predict(seed.sequence, seed.group, seed.id));
    }
  }, [seed, run]);

  useEffect(() => {
    const ex = examples[0];
    if (!seq && ex) {
      setSeq(ex.sequence);
      setGroup(ex.organism_group);
    }
  }, [examples, seq]);

  const r = done?.results?.[0] ?? null;
  const track = r ? trackFor(mode, r) : undefined;
  const inSample =
    r !== null &&
    examples.some((e) => e.sequence_sha256 === r.sequence_sha256 && e.in_training_set === true);

  return (
    <>
      <div className="head">
        <div>
          <h1>Predict</h1>
          <p className="sub">
            Annotations for one sequence, each labelled with the engine that produced it.
          </p>
        </div>
      </div>

      <SequenceForm
        id="predict-input"
        value={seq}
        onChange={setSeq}
        group={group}
        onGroup={setGroup}
        examples={examples}
        onPick={(e) => {
          setSeq(e.sequence);
          setGroup(e.organism_group);
        }}
        action="Predict"
        busyAction="Running…"
        busy={busy}
        onSubmit={() => void run(...predict(seq, group))}
        compact={r !== null}
      />

      <JobState job={job} error={error} accepted={accepted} onRetry={retry} />
      {busy && !r && (
        <Card>
          <Skeleton lines={5} label="Waiting for the prediction" />
        </Card>
      )}
      {!busy && !r && !error && (
        <Card>
          <EmptyState icon={Microscope} title="No result yet">
            Paste a sequence or load an example, choose the organism group, and press Predict.
          </EmptyState>
        </Card>
      )}

      {r && (
        <>
          <section className="result-head" aria-label="Result">
            <div>
              <div className="overline">Result</div>
              <h2 className="result-id mono">{r.id}</h2>
              <div className="row tight note">
                <span>{r.length} residues</span>
                <span aria-hidden="true">·</span>
                <span>{pretty(r.organism_group)}</span>
                <span aria-hidden="true">·</span>
                <span className="mono">sha256 {r.sequence_sha256.slice(0, 12)}…</span>
                {r.result_cache === "hit" && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span>served from the result cache</span>
                  </>
                )}
              </div>
            </div>
            <div className="row tight">
              {inSample && <span className="badge insample">in-sample</span>}
              {r.encoder === "live" && <span className="badge">encoded live</span>}
              <button
                className="btn"
                onClick={() => {
                  window.print();
                }}
              >
                <Printer size={14} aria-hidden="true" /> Print report
              </button>
              <button
                className="btn"
                onClick={() => {
                  onScan(r.sequence, r.organism_group);
                }}
              >
                Send to scanner <ArrowRight size={14} aria-hidden="true" />
              </button>
            </div>
          </section>

          {inSample && (
            <div className="flag warning">
              <strong>In-sample sequence.</strong> The heads were fitted on this sequence, so this
              result is not evidence of accuracy.
            </div>
          )}
          {r.outside_evaluated_subset === true && (
            <div className="flag uncertain" role="status">
              <strong>Encoded live, outside the evaluated subset.</strong> This sequence is not one
              of the 500 cached sequences, so frozen ESM-2 ran at request time (ADR-0019). The heads
              and calibrators are the same, but every reported metric was measured on the cached
              subset: treat the probabilities as indicative.
            </div>
          )}
          {!r.evidence_status.available && (
            <div className="flag uncertain" role="status">
              <strong>Homology evidence is unavailable.</strong>{" "}
              {r.evidence_status.unavailable_reason ?? "No reason was reported"}. The model results
              on this page are unaffected, and nothing was substituted for the missing evidence.
            </div>
          )}

          <Summary r={r} />

          <div className="explain-grid">
            <Card
              title="Per-residue explanation"
              aside={
                <Segmented
                  label="Per-residue explanation method"
                  value={mode}
                  onChange={setMode}
                  options={[
                    { value: "attribution", label: pretty(r.attribution_method) },
                    { value: "attention", label: "attention" },
                    { value: "signal", label: "signal peptide" },
                    { value: "hydropathy", label: "hydropathy" },
                  ]}
                />
              }
            >
              <SequenceTrack result={r} mode={mode} onCursor={setCursor} />
              <div className="track-notes">
                <p className="note">
                  {mode === "hydropathy"
                    ? "Kyte-Doolittle hydropathy from the motif engine, not a trained-model output."
                    : (track?.claim ?? "")}
                </p>
                {track && (
                  <p className="note mono">
                    {r.attribution_source} · method {track.method} · target {track.target} ·{" "}
                    {track.signed ? "signed" : "unsigned"} · {String(track.forward_passes)} forward
                    passes
                  </p>
                )}
                <p className="note">
                  Comparable within this sequence and label only. It describes the trained head, not
                  causal importance and not a binding site.
                </p>
              </div>
            </Card>
            <StructurePanel result={r} mode={mode} cursor={cursor} />
          </div>

          <div className="grid two">
            <Card
              title="Localization"
              aside={<ProvenanceBadge src={r.task_sources.localization} />}
            >
              {r.localization.map((row) => (
                <ConfidenceRow key={row.code} row={row} />
              ))}
              <p className="note mono">
                {r.threshold_policy} · calibrator {r.calibration_version ?? "none"} · dashed mark =
                that label&apos;s threshold
              </p>
              {r.no_confident_call && (
                <div className="flag uncertain">
                  <strong>No confident call.</strong> No compartment clears its own threshold. The
                  highest score, {r.top_location} at{" "}
                  {r.localization.find((l) => l.code === r.top_location)?.probability.toFixed(2)},
                  is shown for transparency and is not asserted.
                </div>
              )}
            </Card>

            <div className="stack">
              <Card
                title="Membrane type"
                aside={<ProvenanceBadge src={r.task_sources.membrane_type} />}
              >
                <ul className="kv-rows">
                  {Object.entries(r.membrane_type)
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => (
                      <li key={k}>
                        <span className="row tight">
                          <Chip code={r.top_location} membrane={k} />
                          {pretty(k)}
                        </span>
                        <span className="num">{v.toFixed(2)}</span>
                      </li>
                    ))}
                </ul>
              </Card>
              <Card
                title="Sorting signal"
                aside={<ProvenanceBadge src={r.task_sources.signal_peptide} />}
              >
                <ul className="kv-rows">
                  <li>
                    <span>{pretty(r.sorting_signal.type)}</span>
                    <span className="num">{r.sorting_signal.probability.toFixed(2)}</span>
                  </li>
                  {r.sorting_signal.region && (
                    <li>
                      <span>region</span>
                      <span className="num">
                        {r.sorting_signal.region[0] + 1}–{r.sorting_signal.region[1]}
                      </span>
                    </li>
                  )}
                  {r.sorting_signal.cleavage_site !== null && (
                    <li>
                      <span>cleavage after</span>
                      <span className="num">{r.sorting_signal.cleavage_site}</span>
                    </li>
                  )}
                </ul>
                <p className="note">
                  candidates:{" "}
                  {Object.entries(r.sorting_signal.candidates)
                    .map(([k, v]) => `${pretty(k)} ${v.toFixed(2)}`)
                    .join(" · ")}
                </p>
              </Card>
              <Card
                title="GO molecular function"
                aside={<ProvenanceBadge src={r.task_sources.go_function} />}
              >
                {r.go_function.filter((g) => g.predicted).length === 0 ? (
                  <p className="hint">No term reaches its threshold.</p>
                ) : (
                  <ul className="kv-rows">
                    {r.go_function
                      .filter((g) => g.predicted)
                      .map((g) => (
                        <li key={g.go_id}>
                          <span>
                            {g.name} <span className="note mono">{g.go_id}</span>
                          </span>
                          <span className="num">{g.probability.toFixed(2)}</span>
                        </li>
                      ))}
                  </ul>
                )}
                <p className="note">
                  Predicted by the trained head, calibrated, propagated to ancestors; threshold{" "}
                  {r.go_function[0]?.threshold.toFixed(2)}. Terms carried by homologs are on the
                  Evidence screen, not here.
                </p>
              </Card>
            </div>
          </div>

          <SpaceCard result={r} onOpen={onOpenAccession} />

          <Tabs.Root className="tabs" defaultValue="consistency">
            <Tabs.List className="tab-list" aria-label="More about this result">
              <Tabs.Trigger className="tab" value="consistency">
                Consistency
                {r.consistency_flags.length > 0 && (
                  <span className="tab-count">{r.consistency_flags.length}</span>
                )}
              </Tabs.Trigger>
              <Tabs.Trigger className="tab" value="engines">
                Engines
              </Tabs.Trigger>
              <Tabs.Trigger className="tab" value="export">
                Export
              </Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content className="tab-panel" value="consistency">
              <Card title="Consistency check">
                <Flags flags={r.consistency_flags} />
                {r.evidence_agreement && (
                  <p className="hint">
                    Closest Swiss-Prot hit{" "}
                    <span className="mono">{r.evidence_agreement.top_hit_accession}</span> at{" "}
                    <span className="mono">
                      {(r.evidence_agreement.identity * 100).toFixed(1)}%
                    </span>{" "}
                    identity — its GO terms{" "}
                    {r.evidence_agreement.agrees
                      ? "agree with the model's."
                      : "differ from the model's (see the Evidence screen)."}
                  </p>
                )}
              </Card>
            </Tabs.Content>
            <Tabs.Content className="tab-panel" value="engines">
              <SourceSummary result={r} />
            </Tabs.Content>
            <Tabs.Content className="tab-panel" value="export">
              <DownloadButtons result={r} jobId={done?.job_id ?? "unknown"} />
              <p className="note">
                Print report gives a paper layout of this result with its provenance and limits;
                choose &quot;Save as PDF&quot; in the print dialog for a file.
              </p>
            </Tabs.Content>
          </Tabs.Root>

          <ProvenanceBar p={r.provenance} />
        </>
      )}
      {r && (
        <PrintReport
          result={r}
          jobId={done?.job_id ?? "unknown"}
          inSample={inSample}
          printedAt={new Date()}
        />
      )}
    </>
  );
}
