# THIRD_PARTY.md — ProtLoc Fusion

Every external repository, model checkpoint or dataset this project studies,
depends on, or benchmarks against. Blueprint §28.2 defines the required fields;
this file is the live register.

**Rules**

1. Nothing enters the pipeline until it has a row here.
2. `Pinned at` must be a real commit SHA or release tag, never a branch name.
   Record it with:
   `git ls-remote https://github.com/<owner>/<repo> HEAD` or, once vendored,
   `git -C vendor/<repo> rev-parse HEAD`.
3. The **Redistribution** column governs what may ship inside our Docker image
   and what may be exposed through the public web server. Read it before
   deploying, not after.
4. License facts below were read from each repository's own `LICENSE` file at the
   pinned commit on **2026-09-14**, not from a badge or a GitHub summary. Re-verify
   at any re-pin — licenses change, and two rows below changed between the first
   reading and this one.
5. Every `Pinned at` SHA below is the repository's `HEAD` as reported by
   `git ls-remote https://github.com/<owner>/<repo> HEAD` on **2026-09-14**, with the
   commit date shown beside it. Resolved in Phase 1 (`PROJECT_PLAN.md` §7.3, M2).

---

## 1. Register

| # | Repository | Role in ProtLoc Fusion | License (verified at the pinned commit) | Pinned at | Weights used? | Redistribution |
|---|-----------|------------------------|---------|-----------|---------------|----------------|
| 1 | `graph-part/graph-part` | Homology-aware split generation | BSD-3-Clause | `f470726216bc7ec4cb60f6424548bfd60c550e42` (2023-10-26) | no | Permissive — safe to vendor and ship |
| 2 | `soedinglab/MMseqs2` | Clustering backend / alternative to GraphPart's alignment step | **MIT** — see §1.1 | `d401e78c2d18a822cdb1527d7464a043f6035a15` (2026-09-13) | no | Permissive. Still invoked as an external binary, for operational reasons rather than licensing ones |
| 3 | `facebookresearch/esm` | ESM-2 embeddings (primary representation) | MIT | `2b369911bb5b4b0dda914521b9475cad1656b2ac` (2023-06-27) | **yes** — ESM-2 650M | Permissive; still cite and record checkpoint revision |
| 4 | `efs-opensource/calibration-framework` (netcal) | Temperature / vector scaling, ECE, reliability diagrams | Apache-2.0 | `34b677f42b83803aa35503c5b6fefb1387ea9167` (2026-04-16) | no | Permissive — safe to ship |
| 5 | `huggingface/peft` | LoRA adaptation of the PLM trunk | Apache-2.0 | `e99fdd275075068d093f8996a59bc0eb865f71a2` (2026-09-14) | no | Permissive — safe to ship |
| 6 | `fteufel/signalp-6.0` | Reference implementation for SP + cleavage-site modelling; benchmark baseline | **CC BY-NC-ND 2.0** — see §1.1 | `9bcd1b25fbbf8d8e97d048ec1122f70390253de3` (2025-02-04) | **do not redistribute** | **Restricted.** Non-commercial *and* no-derivatives. Study the method; do not bundle weights, do not adapt the code, do not serve it |
| 7 | `JJAlmagro/TargetP-2.0` | Reference for mTP/cTP modelling; benchmark baseline | CC BY-NC-SA 4.0 | `695c7b252298d3ec7a30491c8026e8839d41f678` (2019-09-05) | **do not redistribute** | **Restricted.** Non-commercial, share-alike. Never vendored |
| 8 | `TviNet/DeepLoc-2.0` | Reference architecture for multi-label localization + attention interpretability | **CC BY-NC-SA 4.0** — see §1.1 | `e6d44d01f6026b756ba35c4a89715e23a2957590` (2026-01-12) | no | **Restricted.** Non-commercial, share-alike. Read the method; never vendored, never copied |
| 9 | `brinkmanlab/psortb_commandline_docker` | Prokaryotic localization benchmark baseline | **No LICENSE file in the repository**; its README states PSORTb itself is GPL-3.0 — see §1.1 | `6a034d8a99f2fe00d9b414bbb201478bf97c9da2` (2026-03-24) | no | **Restricted.** Run as a separate container, invoked over its CLI. Copy nothing; link nothing |
| 10 | `mahdip72/prot2token` | Methodology reference for unified multi-task protein prediction | **AGPL-3.0** (confirmed) + research-use-only | `8bd6ef846e21a08d80bf1a84bbbf923f10835a2d` (2025-06-06) | no | **Hard blocker — see §3** |
| 11 | `mem0ai/mem0` | Project context memory service (development-time only) | Apache-2.0 | `c7ee362aff94a369af70f13f2b4f853f6793ff4c` (2026-09-11) | no | Dev tooling; never part of the served product |
| 12 | `ashleve/lightning-hydra-template` | Config/experiment scaffolding pattern | **No LICENSE file at the pinned commit**; the README's badge claims MIT and its License section is empty — see §1.1 | `bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144` (2023-09-25) | no | **Treat as unlicensed.** The pattern is adopted; no code is copied |

Every `<PIN>` is resolved. Re-pinning a row means re-reading its licence at the new
commit and recording both here in the same change.

### 1.1 What changed when the licences were verified

Four rows moved when the `LICENSE` file at each pinned commit was read rather than
its GitHub summary. Recorded because a licence the register gets wrong is worse
than one it marks unknown.

| # | Was | Is | Why it matters |
|---|---|---|---|
| 2 | GPL-3.0 *(verify)* | **MIT** | MMseqs2 is permissive. The "do not statically link" instruction was written to avoid GPL contamination that does not exist. Keeping it as an external binary is still right — it is a heavyweight C++ tool with its own release cadence — but that is now an engineering choice, not a licence constraint |
| 6 | Academic / non-commercial *(verify — DTU terms)* | **CC BY-NC-ND 2.0** | **ND is stricter than assumed.** No-derivatives forbids adapting the code at all, not merely redistributing it. Reading the method and reimplementing independently is fine; porting or modifying a file is not |
| 8 | *(verify at pin)* | **CC BY-NC-SA 4.0** | DeepLoc-2.0 is restricted, not open. It moves into the same class as rows 6 and 7: read, never copied |
| 9 | GPL *(verify)* | **no LICENSE file** | Absence of a licence is not permission. The README attributes GPL-3.0 to PSORTb itself, but the wrapper repository grants nothing, so it is treated as all-rights-reserved and only ever invoked as a separate container |
| 12 | MIT *(verify)* | **no LICENSE file at the pinned commit** | Only a README badge claims MIT, and the section it links to is empty. The project uses the layout idea, which is not copyrightable; no file is copied, so nothing depends on resolving this |

None of these changes unblocks anything, and two of them tighten a constraint.
Rows 6, 7, 8, 9 and 10 are now all **restricted**: five of the twelve repositories
may be read and cited but never copied, vendored, bundled or served.



---

## 2. Per-repository detail

### 2.1 GraphPart — `graph-part/graph-part`

- **License:** BSD-3-Clause.
- **Install:** `pip install graph-part`. Requires EMBOSS `needleall` on PATH
  (`conda install -c bioconda emboss`, or `sudo apt-get install emboss`).
- **CLI entry point:** `graphpart`.
- **Why it is adopted:** it is the partitioning algorithm from the same DTU
  group behind SignalP and DeepLoc, published in *NAR Genomics and
  Bioinformatics* (Teufel et al., 2023). Adopting it means the leakage-prevention
  claim in the paper rests on a citable, reviewer-recognised method rather than
  on a bespoke script.
- **What it replaces:** the hand-rolled homology clustering step in Blueprint
  §7.3. `scripts/cluster_sequences.py` and `scripts/build_splits.py` become
  wrappers around `graphpart`, not reimplementations.
- **Files copied:** none. Invoked as an external CLI.
- **Citation:** Teufel F, et al. *GraphPart: homology partitioning for
  biological sequence analysis.* NAR Genomics and Bioinformatics, 2023.

### 2.2 MMseqs2 — `soedinglab/MMseqs2`

- **License:** MIT, read from `LICENSE.md` at the pinned commit on 2026-09-14.
  The register previously recorded GPL-3.0; see §1.1.
- **Role:** GraphPart can use alignment backends; MMseqs2 `easy-cluster` is also
  the fallback/comparison clustering path referenced in the GraphPart paper.
  It is also the homology-search engine for the evidence layer (ADR-0003).
- **Integration rule:** invoked as an external binary from a subprocess. The
  reason is operational, not legal — MIT imposes no linking constraint. It is a
  large C++ tool with its own release cadence and index format, and running it
  out of process keeps its version pinned in the image rather than in our
  dependency graph.
- **Open decision:** whether the frozen split uses GraphPart's needleall
  alignment or MMseqs2 pre-clustering. Requires an ADR before the split is
  frozen.

### 2.3 ESM — `facebookresearch/esm`

- **Role:** primary representation. ESM-2 650M is the named candidate
  checkpoint.
- **Critical requirement:** the embedding cache key must include the checkpoint
  identity *and* revision (Blueprint §8.5). An embedding produced by one
  revision is not interchangeable with another. Record the exact HuggingFace
  revision hash, not just `esm2_t33_650M_UR50D`.

### 2.4 netcal — `efs-opensource/calibration-framework`

- **License:** Apache-2.0 (the project switched from MPL-2.0 to APL-2.0).
- **Install:** `python3 -m pip install netcal`.
- **Role:** supplies temperature scaling, vector scaling, binning methods, ECE
  and reliability diagrams — the entire measurement half of Blueprint §13.
- **Why not write it:** ECE is easy to implement and easy to implement subtly
  wrong (binning strategy, class-wise vs top-label, equal-width vs equal-mass).
  A reviewer who knows the literature will ask which variant you used. Using a
  named library with a named binning strategy answers that question.
- **What stays ours:** the abstention policy and the coverage–risk curve.
  netcal calibrates; it does not decide when to refuse.

### 2.5 PEFT — `huggingface/peft`

- **Role:** LoRA adaptation of the ESM-2 trunk (Blueprint §12).
- **Must be recorded per run:** rank `r`, alpha, dropout, target modules,
  trainable-parameter count, peak GPU memory, epoch wall-time, final checkpoint
  size. These are the columns of the trade-off table that makes H4 defensible.

### 2.6 SignalP 6.0 / TargetP 2.0 / DeepLoc — DTU tools

- **Status: reference and benchmark only.** These are the tools ProtLoc Fusion
  is positioned against. Their terms are academic / non-commercial
  (TargetP 2.0 is CC BY-NC-SA; SignalP 6.0 carries DTU academic terms — read
  the actual license text at pin time).
- **Hard rules:**
  - Never vendored into this repository.
  - Never bundled into the Docker image.
  - Never served through the public API.
  - Benchmark numbers are produced by running them **locally, offline**, and
    only the resulting metrics are published.
- **What we legitimately take:** the *methodology* — SignalP's region-based
  decoding for cleavage sites, DeepLoc's attention-based interpretability
  pattern, and their published evaluation protocols. Methods are citable;
  code and weights are not redistributable.

### 2.7 PSORTb — `brinkmanlab/psortb_commandline_docker`

- **Role:** prokaryotic localization baseline for the Gram+/Gram− comparison.
- **Integration rule:** runs as its own container, invoked by
  `scripts/benchmark_external.py`. Its output is parsed into our metric format.
  It never shares a process or an image with our service.

### 2.8 Prot2Token — `mahdip72/prot2token`

**Read this before importing anything from it.**

- **License:** AGPL-3.0, *and* the README states the repository and pre-trained
  models are provided **for research purposes only**.
- **Why that matters here:** ProtLoc Fusion's product definition is a **public
  web server** (Blueprint §3.3). AGPL's network-use provision means that
  serving AGPL-derived code over a network obliges you to offer the complete
  corresponding source of your service under AGPL. Combined with a research-only
  restriction, this is the single highest-risk dependency in the register.
- **Ruling:** **methodology reference only.** Read the paper, study the
  next-token-prediction framing for unifying tasks, cite it — and write no line
  of our code by copying theirs. Do not `pip install prot2token` into the
  serving environment. If a future experiment genuinely needs it, that requires
  an ADR that explicitly addresses the AGPL network clause.
- **Why it is still in this register:** it is the closest published work to
  ProtLoc Fusion's core claim (one shared representation, many protein tasks),
  so the paper's related-work section must engage with it honestly rather than
  omit it.

### 2.9 mem0 — `mem0ai/mem0`

- **License:** Apache-2.0. See `docs/CONTEXT_MEMORY.md` for the full setup.
- **Scope boundary:** development-time tooling only. It stores *project*
  context for coding sessions. It never touches protein data, never runs in the
  serving container, and never appears in a reproducibility manifest.

---

## 3. Contamination rules

Two failure modes will invalidate results, and both are silent:

**License contamination.** Copying a few lines from an AGPL or non-commercial
repository into `src/` makes the whole served application subject to those
terms. The rule is absolute: **restricted repositories are read, never
copied.** If you find yourself with one of their files open beside your editor,
close it and re-derive the logic from the paper.

**Benchmark contamination.** The external tools were trained on UniProt
releases that overlap your test split. A protein in your held-out set may be in
DeepLoc's training set. This does not invalidate *their* published numbers, but
it does mean a head-to-head comparison on your split flatters you or them
unpredictably. Blueprint §7.5's temporal/new-release split exists partly for
this reason. State the caveat in the paper explicitly — a reviewer who spots it
before you do will not be gentle.

---

## 4. Maintenance

- Re-verify every license at pin time and again before submission.
- A dependency without a pinned SHA blocks the Definition of Done (§30.1).
- When a row's license or terms change, record the change here with a date;
  do not silently overwrite the old value.
