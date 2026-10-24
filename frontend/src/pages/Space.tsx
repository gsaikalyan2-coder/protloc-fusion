/*
 * Protein space — every cached sequence as a point in 3D, placed by the first three principal
 * components of its mean-pooled ESM-2 vector and coloured by its Swiss-Prot location. Click a
 * point to open that sequence in Predict. Orientation only: built offline by
 * scripts/build_space.py from the embedding cache, served by GET /demo/space, never a score.
 */
import type { ReactElement } from "react";
import { ProteinSpace } from "../components/ProteinSpace";
import { Card, ProblemNotice, Skeleton } from "../components/ui";
import { useSpace } from "../services/demoApi";

export default function Space({
  onOpenAccession,
}: {
  onOpenAccession: (accession: string) => void;
}): ReactElement {
  const state = useSpace();
  return (
    <>
      <div className="head">
        <div>
          <h1>Protein space</h1>
          <p className="sub">
            The cached sequences in the representation every head reads. Proteins that sit close
            together have similar ESM-2 vectors; colour is where Swiss-Prot says they are. Rotate
            it, filter by compartment, and click a point to open that sequence in Predict.
          </p>
        </div>
      </div>
      <Card>
        {state.status === "loading" && <Skeleton lines={8} label="Loading the protein-space map" />}
        {state.status === "error" && <ProblemNotice problem={state.problem} />}
        {state.status === "ready" && (
          <>
            <ProteinSpace space={state.data} onOpen={onOpenAccession} height={600} />
            <p className="note">
              {state.data.note} {state.data.scope_label}. Embedding{" "}
              {state.data.embedding.checkpoint} at {state.data.embedding.revision.slice(0, 7)},{" "}
              {state.data.embedding.pooling} pooling.
            </p>
          </>
        )}
      </Card>
    </>
  );
}
