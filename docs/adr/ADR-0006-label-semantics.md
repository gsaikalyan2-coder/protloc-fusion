# ADR-0006 — Label semantics: four states, per-label evidence, and where a negative comes from

- **Status:** **Accepted** (2026-10-02; proposed 2026-09-15). Saikalyan reviewed §3.3
  against the real per-task counts of the Swiss-Prot 2026_03 corpus (`docs/phases/05.md`).
- **Deciders:** Saikalyan
- **Blocks:** Phase 3 (the corpus build), and after it Phase 8 (the multi-task loss
  reads these states) and Phase 14 (every per-task count in the paper)
- **Related:** `CLAUDE.md` § Data schema (amended 2026-09-15 by this ADR);
  `docs/phases/PHASE_03_data.md` §3.2; `backend/rules/uniprot_label_map.yaml`;
  `backend/rules/organism_groups.yaml`; `docs/THIRD_PARTY.md` §2.10

---

## 1. Context

The data schema was frozen on 2026-09-14, in Phase 2, before anyone parsed a UniProt
record. Checking it against what a Swiss-Prot flat-file entry can actually supply
turned up eight mismatches (Phase 3 review, 2026-09-15):

| # | The schema said | A Swiss-Prot entry actually has |
|---|---|---|
| A1 | `sequence_sha256` unique | many accessions with byte-identical sequences |
| A2 | one `annotation_evidence_code` per record | an ECO code on every location, feature and GO term, plus GO evidence codes, which are a different system |
| A3 | each label is `unknown` or `negative` | a third case: the task does not apply to this organism |
| A4 | "annotated negative" | almost nothing curated as a negative |
| A5 | an `organism_group` column | a taxonomic lineage, and no Gram stain |
| A6 | `cluster_id` on the sequence record | nothing: Phase 4 produces clusters |
| A7 | seven label *columns* | sets (localization, GO), or a peptide plus a position |
| A8 | nothing about completeness | a `Fragment` flag and `NON_TER` features |

Every one of these decides what the model is trained to believe. Reversing any of
them later means rebuilding the corpus, re-splitting and retraining, so this ADR
settles them before the first label is written.

---

## 2. Decision drivers

1. **`unknown` must never become `negative` without anyone noticing.** A masked loss
   that sees an unlabelled protein as a curated negative learns that absence of
   evidence is evidence of absence. The resulting miscalibration would not show up in
   any single metric.
2. **No biology in Python.** Term-to-class mapping and lineage-to-group mapping are
   biology, so they live in versioned YAML under `backend/rules/`, and the build
   records their SHA-256 in the manifest.
3. **Conservative by default.** Where a rule is unclear, the label is `unknown`. A
   smaller training signal is recoverable; a wrong one is not detectable.
4. **Comparable with the tools being replaced**, where their published
   label-construction methods apply. We read those methods as papers. Their
   repositories are restricted (THIRD_PARTY §1.1) and nothing is copied from them.

---

## 3. Decision

### 3.1 Four label states

Each of the seven tasks, for each record, carries exactly one state:

| State | Meaning | `values` |
|---|---|---|
| `annotated` | Curated annotation exists and maps to at least one class | non-empty, sorted |
| `negative` | The §3.3 rule for this task says the absence is informative | empty |
| `unknown` | No usable annotation, and absence is not informative | empty |
| `not_applicable` | The task is not defined for this organism group (masked) | empty |

`not_applicable` is the masked state that `TaskPrediction` already describes on the
serving side. It is **not** `unknown`: a masked task is left out of the loss by design
and left out of coverage statistics. An unknown one is left out of the loss because
nobody knows the answer.

The vocabulary is `protloc.domain.types.LabelState`. `backend/tests/unit/test_records.py`
proves that each state survives the canonical round trip, and that `unknown` and
`negative` produce different bytes.

### 3.2 Which tasks apply to which group

| Task | eukarya | gram_positive · gram_negative · archaea |
|---|---|---|
| `localization` | applies | `not_applicable` |
| `prokaryotic_localization` | `not_applicable` | applies |
| `mitochondrial_transit_peptide` | applies | `not_applicable` |
| `chloroplast_transit_peptide` | applies | `not_applicable` |
| `membrane_type`, `signal_peptide`, `go_function` | applies | applies |

The `localization` row mirrors `prokaryotic_localization`: the eukaryotic compartment
vocabulary is not asked of a bacterium, just as the reverse is masked already.

### 3.3 Where a negative comes from — per task

Every rule below also requires the record to be a **non-fragment** (§3.6). On a
fragment, a negative can't be told apart from a missing terminus, so the state is
`unknown`.

| Task | `annotated` when | `negative` when | otherwise |
|---|---|---|---|
| `signal_peptide` | a `SIGNAL` feature starts at residue 1 with exact ends | no `SIGNAL` feature at all. Swiss-Prot curation records signal peptides as features, so a reviewed full-length entry without one is treated as a curated absence | `unknown` (e.g. a `SIGNAL` with fuzzy ends) |
| `mitochondrial_transit_peptide` | a `TRANSIT` feature at residue 1, exact ends, whose note maps to mitochondrion | no `TRANSIT` feature at all | `unknown` |
| `chloroplast_transit_peptide` | as above, note maps to chloroplast | no `TRANSIT` feature at all | `unknown` |
| `localization` | at least one non-isoform subcellular-location term maps to a class | **never** | `unknown` |
| `prokaryotic_localization` | as above, prokaryotic vocabulary | **never** | `unknown` |
| `membrane_type` | a topology term (e.g. *Single-pass type I membrane protein*) maps to a class | the entry has a curated, mapped subcellular location, no topology term, no location term containing the membrane marker word, and no `TRANSMEM` or `INTRAMEM` feature | `unknown` |
| `go_function` | at least one molecular-function GO term | **never**. GO is open-world | `unknown` |

Two consequences are stated here so they are not rediscovered:

- **Localization has no per-class negatives in the corpus.** If Phase 8 wants
  per-class negatives for a multi-label head (for example, "annotated as nucleus"
  implies "not mitochondrion"), that is a closed-world assumption applied **at
  loss time**, over `annotated` rows only, and it is recorded in the run manifest.
  The corpus does not bake it in.
- **A transit-peptide negative on a non-plant eukaryote is trivially true.**
  `chloroplast_transit_peptide` negatives are therefore dominated by animals and
  fungi. Phase 8 decides whether to down-weight them. The corpus reports the count.

### 3.4 Label values

- `cleavage_site` (signal and transit peptides only) is the **1-based position of
  the last residue of the peptide**, i.e. the `SIGNAL`/`TRANSIT` feature's end. It is
  the same convention as `TaskPrediction.cleavage_site`. Cleavage happens after this
  residue.
- Class strings come from `backend/rules/uniprot_label_map.yaml`. Terms that map to
  no class are dropped and counted in the build report. They never create a class
  implicitly.
- Subcellular-location blocks that are isoform-specific (`[Isoform N]:`) are ignored.
  An entry whose only location blocks are isoform-specific is `unknown` for
  localization. The canonical isoform isn't reliably "Isoform 1", and guessing
  would be biology in code.

### 3.5 Evidence is per label

The single `annotation_evidence_code` column is removed. Each task label carries
`evidence: {value → sorted list of codes}`:

- locations and features: **ECO** identifiers (`ECO:0000269`), taken from the curly
  braces or the `/evidence` qualifier. The citation after `|` is dropped.
- GO terms: the **GO evidence code** (`IDA`, `IEA`, …) from the `DR   GO` line. The
  two code systems are never mixed within a task.

A label with no evidence attached in the source gets an empty list, never an
invented code. `data.labels.accepted_evidence_codes` is frozen as `all`. The corpus
keeps every code, and any evidence filter is a Phase 8 decision applied at loss time
and recorded in the run manifest, which leaves the corpus hash unchanged.

### 3.6 Record-level fields

- `sequence_sha256`: SHA-256 of `sequence_normalized` as ASCII. **Indexed, not
  unique.** Identical sequences across accessions are kept, and Phase 4's leakage
  gate handles them.
- `is_fragment`: true when the description carries `Flags: Fragment` or
  `Fragments`, or the entry has a `NON_TER` feature.
- `organism_group`: from `backend/rules/organism_groups.yaml`, which maps lineage
  tokens with first-match-wins rules. An entry that matches no rule, including every
  virus, is **excluded** from the corpus, and the exclusion is counted by reason in
  the manifest.
- `cluster_id`: removed from the sequence record. It belongs to the split registry
  (Phase 4).

---

## 4. Consequences

- `CLAUDE.md`'s data schema is amended and dated 2026-09-15.
- The corpus is smaller than Swiss-Prot. The excluded counts (unmapped organism
  group, disallowed residue) are manifest fields, not footnotes.
- The two YAML rule files are corpus inputs. Editing either changes the corpus hash,
  and that change is intended.
- Phase 8's loss masking reads `state`. `not_applicable` and `unknown` are both
  masked, but they are counted separately in every coverage table.

## 5. Alternatives rejected

- **Closed-world localization negatives in the corpus.** This would add labels that
  no curator wrote. It is rejected for the corpus and left available at loss time,
  where it is visible and recorded.
- **Include isoform blocks by taking Isoform 1.** Rejected for the reason in §3.4.
- **Default unmatched bacteria to Gram-negative.** Gram-negative is the most common
  case, but a default is a guess that nobody would ever see. Unmatched entries are
  excluded and counted instead.
- **TrEMBL labels.** Rejected by Phase 3 decision D1: unreviewed annotation is not a
  label.
