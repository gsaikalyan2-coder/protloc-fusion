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
| 3 | `facebookresearch/esm` | ESM-2 embeddings (primary representation) | MIT | `2b369911bb5b4b0dda914521b9475cad1656b2ac` (2023-06-27) | **yes** — ESM-2 150M (`a695f604…`, ADR-0009) | Permissive; still cite and record checkpoint revision |
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

## 1.2 Datasets and data tooling (added Phase 3, 2026-09-15)

Rule 1 covers data as well as code: the corpus does not enter the pipeline until
it has a row. Versions below were read on **2026-09-15**. The tag SHAs came from
`git ls-remote`, and each licence was read from the `LICENSE` file at that tag.

| # | Source | Role in ProtLoc Fusion | License | Pinned at | Redistribution |
|---|---|---|---|---|---|
| 13 | UniProtKB/Swiss-Prot — `uniprot_sprot.dat.gz` (UniProt Consortium) | The training corpus: sequences and curated labels (Phase 3, decision D1: Swiss-Prot only) | **CC BY 4.0** — see §2.10. *Unverified at pin* | The release in `reldate.txt` on the snapshot day, recorded with its SHA-256 in `data/manifests/snapshot.json` (Phase 3 §3.4) | Permitted with attribution. The derived corpus may be published in the HF data repo (ADR-0005) if it carries the UniProt attribution and the release |
| 14 | `iterative/dvc` | Data versioning: pointers in git, bytes outside it | Apache-2.0 | `3.67.1` → `356dfa03278058b02df42124f243c2c345329dae` | Dev and training tooling only. Never in the serving image |
| 15 | `biopython/biopython` | Parsing the Swiss-Prot flat file (`Bio.SwissProt`) | **Biopython License Agreement**; some files dual-licensed with BSD-3-Clause — see §2.11 | `biopython-188` (PyPI `1.88`) → `d7e4b8b19399668b09442a5b35765d9186b5f665` | Permissive, with notice retention. Build-time only. Never in the serving image |

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
- **Split backend — decided.** The frozen split uses GraphPart over
  `needleall` alignments, not MMseqs2 pre-clustering
  (`docs/adr/ADR-0002-graphpart-threshold-backend.md`, Accepted). This line
  read "open decision" until 2026-09-15; the ADR had already closed it.

### 2.3 ESM — `facebookresearch/esm`

- **Role:** primary representation. ESM-2 150M (`facebook/esm2_t30_150M_UR50D`,
  MIT) is the chosen checkpoint, pinned at revision
  `a695f6045e2e32885fa60af20c13cb35398ce30c` (ADR-0009). The weights are
  downloaded from the HuggingFace Hub through `transformers`; the `esm` repository
  itself is not a dependency.
- **Critical requirement:** the embedding cache key must include the checkpoint
  identity *and* revision (Blueprint §8.5). An embedding produced by one
  revision is not interchangeable with another. Record the exact HuggingFace
  revision hash, not just `esm2_t30_150M_UR50D`.

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

### 2.10 UniProtKB/Swiss-Prot

- **License:** CC BY 4.0, as UniProt states for its data. **Not yet verified at
  pin.** On 2026-09-15 the cloud workspace could not reach `ftp.uniprot.org`
  (the proxy refused the connection), so the licence was not read from the
  release itself. `scripts/download_data.py` (Phase 3 §3.4) fetches the
  release's licence file with the snapshot and records its SHA-256. This row
  counts as verified only after that file has been read.
- **Pinned by release, not by commit.** The release string comes from that
  release's `reldate.txt`. It is never typed by hand.
- **Why not the REST API:** it serves only the current release and paginates,
  so it cannot provide a fixed snapshot. The FTP flat file can, and old
  releases stay downloadable under `previous_releases/`.
- **Attribution** belongs in the dataset card, the paper and any published copy
  of the derived corpus.
- **Contamination note:** SignalP 6.0, TargetP 2.0 and DeepLoc 2.0 were trained
  on earlier Swiss-Prot releases. §3 applies.

### 2.11 Biopython

- **License:** the Biopython License Agreement. Some files are dual-licensed
  under BSD-3-Clause, and each file's header says which applies. Both are
  permissive, with copyright notices kept.
- **Role:** reads the flat file's records only. Turning those records into
  ProtLoc labels is our code, in `protloc/data`, governed by ADR-0006.
- **Files copied:** none.

### 2.12 DVC

- **License:** Apache-2.0.
- **Role:** tracks `data/raw/` and `data/processed/`. The pointer files are
  `data/raw.dvc` and `data/processed.dvc`, placed beside those directories
  rather than inside them (Phase 3 §4.5).
- **Remote:** a local remote fits the ≤ 20 GB budget (decision D2). The HF data
  repo (ADR-0005) becomes the shared remote in Phase 16.

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
