# Third-party notices

ProtLoc Fusion's own code is licensed under Apache-2.0 (`LICENSE`). This file states
what third-party material the project **uses or ships**, under which terms, and what it
**must not** redistribute. The full register — pinned commits, roles, how each licence
was verified — is `docs/THIRD_PARTY.md`; data terms are in `docs/DATA_LICENSING.md`.

Licence facts below were read from each project's own `LICENSE` file at the commit
pinned in `docs/THIRD_PARTY.md`. Re-verify at any re-pin and again before submission.

## Used, and permitted to ship

| Component | Role | Licence | Notice obligation |
|---|---|---|---|
| ESM-2 (`facebookresearch/esm`; weights via Hugging Face `facebook/esm2_*`) | Frozen protein-language-model embeddings | MIT | Keep the copyright and permission notice; record the checkpoint revision in every run manifest |
| UniProtKB/Swiss-Prot (UniProt Consortium) | Training corpus: sequences and curated labels | CC BY 4.0 | Attribute UniProt, give the release string, indicate that the data was filtered and re-labelled. See `docs/DATA_LICENSING.md` |
| Gene Ontology (`go-basic.obo`, release `releases/2026-07-26`; Gene Ontology Consortium) | Term ids, names and `is_a` / `part_of` structure for the GO head labels | CC BY 4.0 | Attribute the Gene Ontology Consortium and give the release; pointer and SHA-256 only, `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`. See `docs/DATA_LICENSING.md` |
| GraphPart (`graph-part/graph-part`) | Homology-aware train/validation/test partitioning | BSD-3-Clause | Keep the copyright notice and disclaimer if redistributed |
| MMseqs2 (`soedinglab/MMseqs2`), release `18-8cc5c` | Homology index and search for the evidence layer (Phase 13) | MIT (read from the tagged source on 2026-10-04) | Invoked as an external binary, downloaded at build time and never committed; keep the notice if bundled in an image. The Windows archive also contains `busybox.exe`, whose licence was not read in Phase 13 and which is not redistributed |
| netcal (`efs-opensource/calibration-framework`) | Calibration | Apache-2.0 | Keep the licence and NOTICE if redistributed |
| PEFT (`huggingface/peft`) | LoRA adaptation | Apache-2.0 | As above |
| Biopython | Swiss-Prot flat-file parsing (`Bio.SwissProt`) | Biopython License Agreement (some files BSD-3-Clause) | Keep per-file headers |

Python and npm dependencies are pinned in `backend/pyproject.toml` and
`frontend/package-lock.json`; each carries its own licence, all permissive at the time
of writing. `scripts/check_frontend_deps.py` enforces that the frontend stays React-only
with exact pins.

## Read for reference only — do not copy, vendor or redistribute

These projects are the published tools ProtLoc Fusion is positioned against. Their
licences forbid commercial use and, in places, adaptation or redistribution. Nothing
under these licences contributes source, weights or data files to this repository, and
none may be added to it or to any image built from it.

| Project | Licence | Consequence |
|---|---|---|
| SignalP 6.0 (`fteufel/signalp-6.0`) | CC BY-NC-ND 2.0 | No derivatives: read the method, never adapt the code. Benchmark by running the published tool offline |
| TargetP 2.0 (`JJAlmagro/TargetP-2.0`) | CC BY-NC-SA 4.0 | Non-commercial; never redistribute code or weights |
| DeepLoc 2.0 (`TviNet/DeepLoc-2.0`) | CC BY-NC-SA 4.0 | Non-commercial; never redistribute code or weights |
| PSORTb wrapper (`brinkmanlab/psortb_commandline_docker`) | No `LICENSE` file; README attributes GPL-3.0 to PSORTb | Absence of a licence is not permission; treated as all-rights-reserved |

Published benchmark *datasets* from these groups follow their own terms, which are
recorded when the file is fetched. See `docs/DATA_LICENSING.md`.
