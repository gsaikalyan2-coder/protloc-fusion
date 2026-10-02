# ADR-0008 — Batch mode is client-side chunking

- **Status:** **Accepted** (2026-10-02)
- **Deciders:** Saikalyan
- **Amends:** [ADR-0001](ADR-0001-serving-strategy.md) §4 (queue everything) and
  Blueprint §20 (batch processing)
- **Related:** ADR-0007

---

## 1. Context

ADR-0001 assumed one always-on worker draining a server-side queue, so a whole-proteome
FASTA could be submitted once and collected hours later. ADR-0007 puts the backend on
Cloud Run scaled to zero. Cloud Run instances are ephemeral: an idle instance is shut down
and takes its memory with it, and nothing on its disk survives. A server-side queue would
lose every queued job, and every partial result, each time the instance is recycled.

Making the queue durable means adding storage and a worker that outlives requests — an
infrastructure and cost commitment the free-only constraint rules out.

## 2. Decision

**Batch mode is client-side chunking.** The API contract (`POST /batch`, `GET /jobs/{id}`)
is unchanged. What changes is who owns the whole-proteome run:

1. The client splits the FASTA into chunks of at most `MAX_BATCH_SEQUENCES` sequences.
2. It submits each chunk as its own job and polls it. Each job is short enough to finish
   on one instance.
3. The client holds the chunk list, the job ids and the finished results.
4. A `410 Gone` (job lost to a restart or past retention) or `503` makes the client
   resubmit that chunk only. Finished chunks are never recomputed.
5. The frontend and any script implement this loop; the server implements none of it.

The server therefore keeps no durable state and promises none. `POST /batch` remains
valid for a FASTA up to one chunk.

## 3. Consequences

**Positive.** No durable queue to build, host or pay for. A lost instance costs one chunk,
not a proteome. The contract stays frozen; ADR-0001's single inference path is intact.

**Negative.** The browser tab (or script) must stay open for a whole-proteome run, and the
user must be told so. Progress is the client's, not the server's. There is no
"submit and come back tomorrow".

**Open, not measured.** On Cloud Run with request-based billing, CPU outside a request is
throttled. A job computed in a background thread after `202` may therefore stall until the
next request arrives. Whether polling requests keep it running, or the service needs
instance-based billing (a different free allowance: 240,000 vCPU-s a month), must be
measured when the worker exists (Phase 7+). **UNVERIFIED.**

## 4. Revisit triggers

1. Funding or a lab server makes a durable queue affordable (ADR-0001 revisit trigger 1).
2. Users need unattended proteome runs.
3. The CPU-throttling question above is answered in a way that breaks per-chunk jobs.
