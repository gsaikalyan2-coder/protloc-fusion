/* StructurePanel - loads the AlphaFold backbone for a result and says plainly when it cannot. */
import type { ReactElement } from "react";
import { useStructure, type Result } from "../services/demoApi";
import type { Mode } from "./SequenceTrack";
import { StructureViewer } from "./StructureViewer";
import { Box } from "lucide-react";
import { Card, EmptyState, ProblemNotice, Skeleton } from "./ui";

export function StructurePanel({
  result,
  mode,
  cursor,
}: {
  result: Result;
  mode: Mode;
  cursor: number;
}): ReactElement {
  const state = useStructure(result.sequence_sha256);
  const structure = state.status === "ready" ? state.data : null;
  const aligned = structure !== null && structure.ca.length === result.sequence.length;

  return (
    <Card
      title="3D backbone"
      aside={<span className="badge">AlphaFold DB · orientation only</span>}
    >
      {state.status === "loading" && (
        <>
          <p className="hint" role="status">
            Loading the AlphaFold backbone…
          </p>
          <Skeleton lines={6} label="Loading" />
        </>
      )}
      {state.status === "missing" && (
        <EmptyState icon={Box} title="Nothing to draw">
          {state.message} The prediction is unaffected; the 3D view is orientation only.
        </EmptyState>
      )}
      {state.status === "error" && <ProblemNotice problem={state.problem} />}
      {structure && !aligned && (
        <p className="hint" role="status">
          The stored backbone has {String(structure.ca.length)} residues but this sequence has{" "}
          {String(result.sequence.length)}, so nothing is drawn.
        </p>
      )}
      {structure && aligned && (
        <>
          <StructureViewer result={result} mode={mode} structure={structure} cursor={cursor} />
          <p className="note">
            {structure.note} Source: {structure.source}, {structure.accession}, model version{" "}
            {String(structure.model_version ?? "not reported")}; {structure.license}. Colouring by
            an explanation track puts that track on the AlphaFold backbone for orientation; the
            model did not use the structure.
          </p>
        </>
      )}
    </Card>
  );
}
