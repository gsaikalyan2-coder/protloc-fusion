/*
 * What each output is and which engine produces it. Read by Home (one line each) and About
 * (the full statement), so the two can never describe an output differently.
 */
export interface OutputSpec {
  name: string;
  /** "model": a trained head; "motif": the rule engine; "evidence": independent search. */
  engine: "model" | "motif" | "evidence";
  source: string;
  short: string;
  says: string;
}

export const OUTPUTS: OutputSpec[] = [
  {
    name: "Subcellular location",
    engine: "model",
    source: "trained head · calibrated per label",
    short: "Multi-label, each label against its own threshold.",
    says: "Which compartments the head scores above each label's own threshold. If none clears its threshold the page says so instead of showing the top guess.",
  },
  {
    name: "Membrane type",
    engine: "motif",
    source: "motif engine · not a trained model",
    short: "Rule-based call from hydropathy and motifs.",
    says: "A rule-based call from hydropathy and motif features. It is labelled as such wherever it appears.",
  },
  {
    name: "Signal peptide and cleavage site",
    engine: "model",
    source: "trained CRF head",
    short: "Per-residue posterior and a cleavage position.",
    says: "Per-residue posterior for an N-terminal signal region and a predicted cleavage position. Transit peptides are not predicted.",
  },
  {
    name: "GO molecular function",
    engine: "model",
    source: "trained head · calibrated per term",
    short: "A small vocabulary of frequent terms.",
    says: "A small vocabulary of frequent terms only. Terms from homologous hits are shown separately and never merged with these.",
  },
  {
    name: "Homology evidence",
    engine: "evidence",
    source: "MMseqs2 search against Swiss-Prot",
    short: "Nearest Swiss-Prot hits, beside the model, never in it.",
    says: "Nearest Swiss-Prot hits and the GO terms they carry. Evidence is shown beside the model and never feeds a score.",
  },
  {
    name: "Per-residue explanation",
    engine: "model",
    source: "computed from the trained localization head",
    short: "Which residues the head's output depends on.",
    says: "Shows which residues the head's output depends on (occlusion, pooling attention, signal posterior). It is faithfulness to the head, not a claim about biology.",
  },
];
