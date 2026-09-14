# Wireframe — Landing

**Route** `/` · **Data** `GET /health`, `GET /version` · **Phase** 12 implements this.

Markdown rather than a mockup: `docs/DESIGN.md` owns colour and spacing, a binary file
cannot be diffed in review, and a mockup that pins a hex competes with the token pipeline.
What a wireframe fixes is layout, hierarchy, state coverage and which token each region
draws from.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ProtLoc Fusion                                    [ ☀/☾ ]  [ Docs ] │  --surface-1
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Seven protein annotations from one shared representation.          │  h1, --text-primary
│                                                                      │
│   Location · membrane type · sorting signal + cleavage site ·        │  --text-secondary
│   GO function · homology evidence · per-residue explanation.         │
│   Calibrated. It abstains when it should.                            │
│                                                                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │  Paste a sequence                                          │    │  --border
│   │  ┌──────────────────────────────────────────────────────┐  │    │
│   │  │ MKTFFVLLLA…                                          │  │    │  --font-mono
│   │  └──────────────────────────────────────────────────────┘  │    │
│   │  Organism group  ( ) Eukarya  ( ) Gram+  ( ) Gram−  ( ) Arch│    │  REQUIRED, no default
│   │                                          [  Predict  ]     │    │  --accent
│   └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│   Or try an example                                                  │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│   │ Cyt c ox │ │ SP-rich  │ │ E. coli  │ │ Chloro-  │               │  from GET /examples
│   │ subunit4 │ │ secreted │ │ OM porin │ │ plast    │               │
│   │ ·cached· │ │ ·cached· │ │ ·cached· │ │ ·cached· │               │  label, never hidden
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
│                                                                      │
│   ─────────────────────────────────────────────────────────────      │  --border
│   Predictions are queued, not instant. A single sequence usually     │  --text-secondary
│   returns in under a minute; a cold start can take longer.           │
│                                                                      │
│   Result links are unguessable but not private — anyone with the     │  PRIVACY NOTE
│   link can read the result. Results are deleted after the            │  ADR-0001 §4.2
│   retention window. We store no accounts.                            │
│                                                                      │
│   model 1.0.0 · rules 1.2.0 · EXP-0031 · config a3f9…                │  reproducibility
└──────────────────────────────────────────────────────────────────────┘  footer
```

## States

1. **Ready** — as drawn. `GET /health` returned `status: "ok"`.
2. **Degraded** — `status: "degraded"`. A banner above the form: *"Predictions are
   unavailable: no model bundle is loaded."* The Predict button is disabled with that
   reason as its accessible name, never a bare greyed-out control. **This is the state the
   service is in for the whole of Phases 2–6.**
3. **Unreachable** — the health request failed. *"Cannot reach the service."* The form
   still renders; submitting shows the network error rather than pretending.
4. **Queue is deep** — `queue_depth` is high. The wait note gains the current estimate,
   hedged: *"about 4 minutes right now"*.

## Rules

- **The organism group has no default.** Sorting signals and compartments differ enough
  between groups that a pre-selected radio would silently change answers for anyone who
  did not notice it.
- **The privacy note is not a footer link.** It is on the page, in body text. The system
  provides no confidentiality and must not imply any.
- **"Cached" is always visible on gallery cards.** An instant answer presented as freshly
  computed misrepresents how the system behaves, which is the one thing a tool about
  trustworthy prediction cannot afford to do.
- The reproducibility footer renders `—` for each `null` field and a note that nothing is
  loaded, never four blanks.
