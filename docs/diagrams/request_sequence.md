# Request sequence

Every prediction is a job (ADR-0001). This is what that looks like on the wire, including
the branches people forget to design for.

---

## 1. Single sequence — the ordinary path

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant A as API (depth 0)
    participant I as inference (10)
    participant W as Worker (Phase 11)

    U->>A: POST /predict {sequence, organism_group}
    A->>I: submit_sequence(request)
    I->>W: enqueue job of size 1
    I-->>A: JobAccepted
    A-->>U: 202 {job_id, result_url, status:"queued",<br/>queue_position, estimated_wait_seconds, envelope}

    Note over U: The result URL is shareable and citable.<br/>It is unguessable but NOT access-controlled.

    loop poll, backing off
        U->>A: GET /jobs/{job_id}
        A->>I: read(job_id)
        I-->>A: JobResource
        A-->>U: 200 {status:"queued"|"running", progress, envelope}
    end

    W->>W: embed · score 7 heads · calibrate ·<br/>check rules · attribute · fetch evidence
    W->>I: results

    U->>A: GET /jobs/{job_id}
    A-->>U: 200 {status:"completed", results:[SequenceResult], envelope}
```

**Step 5 is the one that surprises people.** `POST /predict` returns no prediction, ever.
The server may answer a size-one job inline with `200` if it finishes inside
`INLINE_RESULT_WINDOW_SECONDS`, but that is a runtime convenience, is absent from the
OpenAPI document, and a client that assumes it is broken against any busy deployment.

**Polling cadence.** The frontend backs off rather than hammering: roughly 1 s for the
first 10 s, then 2 s, then 5 s, capped. `estimated_wait_seconds` informs the first
interval and is displayed with a hedging word — it is an estimate from queue depth and
measured throughput, never a promise.

---

## 2. Batch — a proteome

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant A as API
    participant I as inference
    participant W as Worker

    U->>A: POST /batch (multipart: file + organism_group)
    A->>A: read upload
    A->>I: submit_batch(organism_group, bytes)
    I->>I: parse FASTA, count sequences
    I-->>A: JobAccepted {sequence_count: 20000}
    A-->>U: 202

    loop
        U->>A: GET /jobs/{job_id}
        A-->>U: 200 {status:"running",<br/>progress:{total, completed_count, failed_count}}
    end

    Note over W: Chunked and resumable. A worker restart<br/>mid-run resumes without duplicated work.

    U->>A: GET /jobs/{job_id}/results?format=csv
    A-->>U: 200 text/csv, Content-Disposition: attachment
```

A member sequence that fails does **not** fail the job: it lands in `failed_count`, and its
`SequenceResult` carries an `error` with no partial prediction attached. A partial
prediction presented as a result is worse than an admitted failure.

`?format=csv` flattens to one row per sequence and task and necessarily omits per-residue
attribution. The document says so, so nobody discovers it by diffing two downloads.

---

## 3. The branches that are easy to skip

```mermaid
flowchart TD
    S[GET /jobs/id] --> E{job known?}
    E -- never existed --> E404["404 not_found"]
    E -- retention window closed --> E410["410 job_expired"]
    E -- known --> ST{status}
    ST -- queued/running --> R200P["200 · no results ·<br/>DESIGN.md §3.4 pending state"]
    ST -- completed --> R200["200 · results inline"]
    ST -- failed --> R200F["200 · error, no partial result"]
    ST -- cancelled --> R200C["200 · cancellation reason"]

    D[GET /jobs/id/results] --> DS{completed?}
    DS -- no --> D409["409 results_not_ready"]
    DS -- yes --> D200["200 attachment"]

    X[any endpoint] --> XB{bundle loaded?}
    XB -- no --> X503["503 no_registry_bundle<br/>(the state until Phase 7)"]
```

**404 and 410 are deliberately different.** A user holding a URL from last month should be
told the result expired, not that it never existed — the second reading invites them to
doubt whether they ever ran the job.

**A failed job still answers 200.** The job resource exists and describes a failure; the
HTTP status describes whether the *resource* could be read. Clients branch on `status`, and
on the stable `error` code in an error body — never on the human-readable message, which
may be reworded without a contract change.

**Every one of these bodies carries the reproducibility envelope**, including the 404. That
is the point of rule 6: a user reporting a failure can say which model version produced it
without being asked to reproduce it first.

---

## 4. Cancellation

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as API
    participant I as inference

    U->>A: POST /jobs/{job_id}/cancel
    A->>I: cancel(job_id)
    alt queued or running
        I-->>A: CancelResponse {cancelled: true, status:"cancelled"}
    else already terminal
        I-->>A: CancelResponse {cancelled: false, status: <unchanged>}
    end
    A-->>U: 200
```

Cancelling a terminal job is **not** an error. The caller's intent — that this job not run —
is already satisfied, and answering `409` would make a retry after a dropped connection
look like a bug. `cancelled` says whether this call changed anything.

---

## 5. Degradation

The evidence service is allowed to fail without taking the prediction with it (ADR-0003).
When it does, `Evidence.available` is `false` and `unavailable_reason` says why, rather
than the response carrying two empty lists that read as *"no homologs found"*. Those are
different claims, and only one of them is a scientific statement.

The same principle governs the whole contract: **absent, abstained, masked and pending are
four different things**, and none of them is rendered as another.
