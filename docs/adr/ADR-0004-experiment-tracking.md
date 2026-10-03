# ADR-0004 — Experiment tracking

- **Status:** **Accepted** (2026-09-09)
- **Deciders:** Saikalyan
- **Blocks:** Phase 6 (Baselines) — the first phase that produces a comparable number
- **Related:** Blueprint §11.3, §24; `CLAUDE.md` phase rule 9

---

## 1. Context

From Phase 6 onward, every run must be reconstructable: dataset manifest hash, config
hash, git commit, seed, hardware, checkpoint hash. Blueprint §24.1 requires a stable
experiment id (`EXP-0001`) binding all of it.

Training runs on Colab/Kaggle, where sessions terminate without warning — so the tracking
choice is also a durability decision.

---

## 2. Decision drivers

1. **Reproducibility cannot depend on a third-party account.** A paper whose methods are
   only reconstructable by logging into a SaaS dashboard is not reproducible. If the
   account lapses or the free tier changes, the record must survive.
2. **Session termination.** Anything held only in a Colab session's local filesystem is
   lost when the session dies.
3. **Solo operator.** Self-hosting a tracking server is maintenance that competes with the
   Core list.
4. **Comparison ergonomics.** Comparing eight baseline runs and three LoRA configurations
   in a terminal is miserable; a chart view has real value.

---

## 3. Decision

**A two-layer arrangement: the run manifest is the record; Weights & Biases is a viewer.**

### 3.1 Layer 1 — the run manifest (authoritative)

Every run writes a plain-text manifest into the repository, committed to git:

```
experiments/EXP-0031/
├── manifest.yaml       # the reproducibility record
├── config_resolved.yaml# fully resolved Hydra config
├── metrics.json        # final metrics, per task head
├── metrics_history.csv # per-epoch train/val curves
└── notes.md            # what this run was testing, in one paragraph
```

```yaml
# manifest.yaml
experiment_id: EXP-0031
description: LoRA r=16 vs full fine-tune, all seven heads
git_commit: <SHA>
git_dirty: false
dataset_manifest_sha256: <HASH>
split_version: homology_v1
config_sha256: <HASH>
python: 3.11.x
pytorch: <VERSION>
cuda: <VERSION>
gpu: <MODEL>
seed: 42
esm_checkpoint: facebook/esm2_t30_150M_UR50D
esm_revision: <HASH>
checkpoint_sha256: <HASH>
started_at: 2026-09-09T14:03:11Z
finished_at: 2026-09-09T18:47:52Z
status: completed
wandb_run_id: <ID or null>
```

**`git_dirty: true` invalidates a run for publication.** A run from uncommitted code
cannot be reconstructed, so the training script records it and the paper uses only clean
runs.

This layer has no external dependency. Delete every W&B account and the record is intact.

### 3.2 Layer 2 — Weights & Biases (convenience)

W&B logs live curves, per-task metrics, GPU memory and time, hyperparameters and ablation
tags. Its free tier is adequate for a solo academic project.

**Rules:**

- W&B is **never** the source of truth. Any number in the paper is read from
  `metrics.json`, not from a dashboard.
- Runs log `experiment_id` as a tag so a dashboard row maps to a manifest directory.
- A W&B outage or an offline Colab session must not fail training: run with
  `WANDB_MODE=offline` when unavailable and sync later. The manifest is written either way.
- `wandb_run_id` is recorded in the manifest, so the link points from the durable record
  to the convenience layer rather than the reverse.

TensorBoard was considered and is not excluded — it can be enabled alongside for local
inspection without affecting anything above.

### 3.3 Checkpoint durability

Given session termination, checkpointing to durable storage is part of this decision:

- Checkpoint every epoch to a persistent location, not the session's local disk.
- `train.py --resume <EXP-ID>` restores model, optimizer, scheduler and epoch counter.
- A resumed run keeps the **same** `experiment_id` and appends to `metrics_history.csv`;
  resumption is recorded in `notes.md`, not hidden.

Where "durable" points is settled by ADR-0005 §4.3.

---

## 4. Consequences

**Positive.** Reproducibility survives account loss, tier changes and W&B's continued
existence. Manifests are diffable and reviewable in git. Comparison ergonomics are kept.
The `git_dirty` gate catches a common and near-invisible failure.

**Negative.** Two places to write, so the training script must write both — automated in
one `finalize_run()` helper rather than left to discipline. Manifest directories
accumulate in git, but they are small text files.

**Neutral.** `WANDB_API_KEY` remains a Phase 6 secret; absence degrades to offline mode
rather than failing.

---

## 5. Revisit triggers

1. W&B's free tier stops covering the project — drop to TensorBoard plus manifests, losing
   nothing that the paper depends on.
2. A collaborator joins and needs shared dashboards — revisit tier, not architecture.

---

## 6. Verification

- [x] Tracking decided: manifest authoritative, W&B as viewer.
- [ ] `finalize_run()` writes all five manifest files atomically (Phase 6).
- [ ] `git_dirty` recorded and enforced for publication runs.
- [ ] `--resume` restores model, optimizer, scheduler and epoch (Phase 8).
- [ ] Offline W&B mode verified on a session with no network.
