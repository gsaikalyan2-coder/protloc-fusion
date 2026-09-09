# PHASE_PROTOCOL.md — How ProtLoc Fusion is executed

The operating contract for every phase. `PROJECT_PLAN.md` says *what* each phase
achieves; this document says *how a phase is run*, and `docs/phases/PHASE_NN_*.md` gives
the specific instructions for each one.

Read this once. Read the phase file at the start of every phase.

---

## 1. The three gates

Every phase passes through three gates in order. A gate is not advisory — a phase that
skips one is not complete, regardless of what was built.

```
  ENTRY GATE                EXECUTION                    EXIT GATE
  ──────────                ─────────                    ─────────
  Prerequisites             The prompt is run             Verification commands
  verified true             one subtask at a time         produce expected output
        │                          │                             │
        │                   confirm after each               all checks pass
        │                   meaningful subtask                    │
        ▼                          ▼                              ▼
   phase may start          incremental commits          handover written,
                                                         Current Phase updated
```

**Entry gate.** A checklist of statements that must be *verified true*, not assumed. Each
one names how to check it. If any is false, the phase does not start — the blocking item
is fixed first, and if it needs a decision, that decision becomes an ADR.

**Execution.** The phase file contains a **prompt** — the literal text that starts the
work. It is written to be self-sufficient: a session with no memory of this conversation
can read the phase file and begin. Work proceeds one subtask at a time, with a
confirmation checkpoint after each.

**Exit gate.** A set of commands with expected output. Every check is **binary** — it
passes or it does not. "It seems to work" is not a result. The phase closes only when
every check passes and the handover is written.

---

## 2. Anatomy of a phase file

Each `docs/phases/PHASE_NN_*.md` has the same seven sections, always in this order:

| § | Section | Purpose |
|---|---|---|
| 1 | **Prerequisites** | The entry gate. Statements + how to verify each |
| 2 | **The prompt** | Verbatim text that starts the phase. Self-contained |
| 3 | **Deliverables** | Every file this phase produces, by path |
| 4 | **Execution notes** | Rules that apply to this phase specifically |
| 5 | **Verification** | The exit gate. Commands + expected output, all binary |
| 6 | **Handover** | The template to fill before closing |
| 7 | **Failure modes** | What goes wrong here, and the early signal |

---

## 3. Standing rules — apply to every phase

These are the rules from `CLAUDE.md`, restated as they operate inside a phase.

1. **One phase at a time.** Do not begin Phase N+1 until Phase N's exit gate has passed
   and its handover exists.
2. **Confirm after each subtask.** State what was built, what was verified, and what is
   next. Wait for a go-ahead. Do not chain three subtasks and report at the end.
3. **No placeholders in delivered code.** No `# TODO: implement`, no stub that returns a
   constant, no function whose body is `pass`. If something cannot be built yet, it is
   not in this phase's scope — say so rather than shipping a shell.
4. **Test as you build.** Each subtask carries at least one verification step, run at the
   time, not deferred to the exit gate.
5. **Freeze before you measure.** Any value the paper reports is frozen on validation
   before the test set is touched. Touching test to tune a value invalidates the result
   and cannot be undone by re-freezing afterwards.
6. **An ADR for anything irreversible.** If undoing a choice later would mean retraining,
   re-splitting, or re-annotating, it needs an ADR *before* it is made.
7. **Incremental commits.** Conventional Commits, after each meaningful subtask, not one
   commit at the end of a phase.
8. **Every number has a provenance.** A metric in a document names the command, config
   hash and experiment id that produced it. A number with no provenance is removed, not
   footnoted.
9. **Absolute dates.** `2026-09-09`, never "last week" or "recently".
10. **Say when something is unverified.** An estimate, an assumption, or a figure read
    from documentation rather than measured is labelled as such at the point it is used.

---

## 4. What "verified" means

The exit gate distinguishes three things that are easy to conflate:

| Claim | What it requires |
|---|---|
| **Built** | The code exists and imports without error |
| **Runs** | It executes end-to-end on real input without crashing |
| **Verified** | A command produces output that a third party could check against a stated expectation |

Only the third closes a gate. A verification check must name the command and the expected
result:

> ✅ `pytest backend/tests/golden -q` → `14 passed`
> ❌ "Golden set tests are working"

Where a check cannot be automated, it states the manual procedure and what would count as
failure — never "looks correct".

---

## 5. Confirmation checkpoints

At each checkpoint, report in this shape and stop:

```
SUBTASK COMPLETE — <name>

Built:      <files created or changed>
Verified:   <command run> → <actual output>
Decisions:  <anything chosen along the way, and why>
Surprises:  <anything that did not match the plan>
Next:       <the next subtask>

Proceed?
```

**Surprises is not optional.** It is where the plan gets corrected. A phase that reports
no surprises across six subtasks is either trivial or under-reporting.

---

## 6. Handover

Written before a phase closes, to `docs/handover/handover_phase_NN.txt`. Self-contained:
a cold session with no chat history must be able to resume from it alone.

```
PHASE NN — <name>
Closed: <YYYY-MM-DD>
Commit: <SHA>

PROJECT STATE
  <2-3 sentences: what the system is and what now exists>

BUILT THIS PHASE
  <files, grouped by purpose>

DECISIONS MADE
  <each decision, its reason, and the ADR if one was written>

VALUES FROZEN
  <name = value, and where it is recorded>

VERIFICATION RESULTS
  <each exit-gate check and its actual output>

KNOWN GAPS
  <what is deliberately incomplete, and which phase owns it>

NEXT PHASE
  Phase <NN+1> — <name>
  First three actions:
    1. <...>
    2. <...>
    3. <...>
  Blocking prerequisites not yet met: <none | list>
```

---

## 7. When a phase cannot pass its gate

Three legitimate outcomes, and one that is not.

**Split the phase.** The scope was too large. Close the part that passes as `Phase NN-a`,
move the rest to `Phase NN-b`, and record the split in the handover.

**Revise the criterion.** The exit criterion was wrong — usually a performance target set
before anything was measured. Revise it **with evidence**, in `PROJECT_PLAN.md`, stating
the measured value and why the original was unachievable. A target revised to match
whatever was measured, with no argument, is not a revision.

**Stop and write an ADR.** The blocker is a decision, not a task.

**Not legitimate:** marking the phase complete and noting the gap for later. That is how a
project arrives at Phase 14 discovering that Phase 4's splits leak.

---

## 8. Reference map

| Question | Document |
|---|---|
| What is this project and why | `PROJECT_PLAN.md` §1–3 |
| What does phase N achieve | `PROJECT_PLAN.md` §4 |
| How do I run phase N | `docs/phases/PHASE_NN_*.md` |
| What are the coding rules | `CLAUDE.md` |
| Where does this file go | `FOLDER_STRUCTURE.md` |
| What does it look like | `docs/DESIGN.md` |
| Why was X decided | `docs/adr/` |
| Can I use this repo | `docs/THIRD_PARTY.md` |
| What is still unfrozen | `PROJECT_PLAN.md` §7.2 |
