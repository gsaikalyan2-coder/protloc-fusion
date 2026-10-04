# Data licensing and redistribution policy

Written before any data enters the repository (Phase 1). The rule is simple: **git
holds pointers, manifests and checksums; bytes stay out.** Every data artifact is
fetched at build time from its original source, pinned by release and SHA-256.

## What may and may not be committed

| Artifact | Source | Licence | Committed to git? | Redistributed in images / releases? |
|---|---|---|---|---|
| UniProtKB/Swiss-Prot flat file (raw) | UniProt FTP, pinned release | CC BY 4.0 | No (`data/raw/` is ignored) | No — fetched at build time |
| Processed corpus (`corpus.jsonl`) | Derived from Swiss-Prot | CC BY 4.0 (derivative) | No (`data/processed/` is ignored) | Only with the attribution below |
| Manifest, release string, SHA-256, row counts | Generated | Project (Apache-2.0) | **Yes** (`data/manifests/`) | Yes |
| Split registry and `leakage_report.json` | Generated | Project (Apache-2.0) | Yes, once small and text | Yes |
| SignalP 6.0 / TargetP 2.0 / DeepLoc 2.0 benchmark sets | Authors' sites | **Not yet verified** — read the terms at download time | **No** | **No** |
| Gene Ontology `go-basic.obo` (release `2026-07-26`, SHA-256 `b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc`) | geneontology.org | CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>), read at the Phase 12 pin | No — pointer and hash only (`data/raw/` is ignored) | No — fetched at build time; cite the Gene Ontology Consortium and the release |
| GOA GAF snapshot | EBI GOA | **Not used.** Phase 13 reads GO annotations from the corpus `DR GO` lines (embedded in Swiss-Prot 2026_03, CC BY 4.0); no GAF was downloaded | No | No |
| Swiss-Prot FASTA, entry table, MMseqs2 database and index, hit files (`data/evidence/`) | Derived from the corpus (Swiss-Prot 2026_03) | CC BY 4.0 (derivative) | No (`data/evidence/` is ignored) | No — rebuilt with `scripts/homology.py --index` |
| MMseqs2 `18-8cc5c` Windows binary (`data/evidence/bin/`) | github.com/soedinglab/MMseqs2 release, SHA-256 pinned in `configs/training/homology.yaml` | MIT, read from the tagged source on 2026-10-04. The archive also ships `busybox.exe`; its licence was not read in Phase 13 | No | No — downloaded at build time |
| `data/manifests/homology_metrics.json`, `registry/evidence_homology/0.13.0/bundle.json` | Generated | Project (Apache-2.0) | **Yes** (hashes and counts only) | Yes |
| Embeddings (`*.h5`, `*.pt`) | Generated | Derived from ESM-2 (MIT) and the corpus | No | No |
| Trained weights | Generated | See the model card (Phase 20) | No — Hugging Face Hub, pinned by revision | Per model card |

"Not yet verified" means exactly that: the licence has not been read from the source
yet. It is recorded in `docs/THIRD_PARTY.md` and the dataset manifest when the file is
first downloaded, and nothing is redistributed until then.

## Required attribution (UniProt)

Any release, model card, dataset card or UI footer that uses the corpus states:

> Sequences and annotations from UniProtKB/Swiss-Prot, release *YYYY_NN*
> (UniProt Consortium, CC BY 4.0). Filtered, re-labelled and split by ProtLoc Fusion;
> changes are described in the dataset card.

## Rules

1. Nothing under a non-commercial or no-derivatives licence is copied into the repo,
   vendored, or baked into a Docker image — see `THIRD_PARTY_NOTICES.md`.
2. A file's licence is read from the source on the day it is fetched and written into the
   manifest. A badge or a README summary is not a licence.
3. Large or restricted bytes are verified before staging: `git status --ignored`, and
   `git status --short` must show no `data/` content and no `*.pt`, `*.h5`, `*.pkl`.
4. A release that includes data or weights lists, per artifact, whether it is
   redistributed or downloaded at build time.
