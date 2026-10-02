# ADR-0007 — Hosting for the backend

- **Status:** **Accepted** (2026-10-02) for the Phase 2 acceptance host (Render free tier).
  **The host for the real serving image is undecided** (§3.3). Remaining §6 measurements
  are **UNVERIFIED** where marked.
- **Deciders:** Saikalyan
- **Supersedes:** [ADR-0005](ADR-0005-hosting-and-artifact-storage.md)
- **Related:** ADR-0001 (queue everything), ADR-0008 (batch durability)

---

## 1. Context

ADR-0005 chose a free Hugging Face Docker Space (CPU Basic) for the backend. That rested
on Docker Spaces being free. Checked on 2026-10-02 against primary sources, it no longer
is:

- A Hugging Face forum thread dated 2026-07-08 quotes the Space-creation page: *"Static
  Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
  requires a PRO subscription."* It reports no changelog, no pricing-page update and **no
  staff reply**; users there also report existing Docker Spaces becoming unusable without
  upgrading. A second thread (2026-07-18) has one community member saying a PRO account
  can run a Docker Space on cpu-basic with no hourly charge, again with no staff reply.
- `huggingface.co/pricing` lists PRO at **$9/month** and says it includes "Host ZeroGPU,
  Gradio & Docker Spaces". The same page lists CPU Basic as 2 vCPU, 16 GB, free.
- The Docker Spaces documentation page says nothing about the requirement.

**Confidence.** The paywall is established by a primary UI quote relayed in a community
thread and by the pricing page; it is not stated by Hugging Face staff. Treat it as true
and re-check before relying on it either way.

The operator is a student with no funding line. The constraint stated for this phase is
**free options only**.

## 2. Options

| | HF Docker Space on PRO | Google Cloud Run | Render |
|---|---|---|---|
| Cost | **$9/month**, recurring | $0 within the free tier (below); billing account required | Free tier $0; paid starts at $7/month |
| Memory | 16 GB (CPU Basic, per pricing page) | up to **32 GiB**, 8 vCPU per instance | Free: **512 MB**. Paid per the pricing page: 2 GB $25, 4 GB $85, 8 GB $135, 16 GB $200 per month |
| Idle behaviour | Sleeps (not stated on pricing page) | Scales to zero; a cold request pays a container start | Free service spins down after 15 min without inbound traffic; *"about one minute"* to spin up |
| Image size limit | not stated in sources read | Docs: *"There is no direct limit for … the size of container images you can deploy."* | not stated in sources read |
| Request timeout | not measured | up to 60 min per request | not stated in sources read |
| Free-tier allowance | n/a | per month: 180,000 vCPU-s, 360,000 GiB-s, 2 M requests (request-based billing) | 750 free instance hours per workspace per month |
| Fits ESM-2 650M (≈2.6 GB fp32 weights, ADR-0001 §1.1) | Yes | Yes | **Free: no.** Smallest tier that clearly fits: 4 GB at $85/month |
| Outbound ports | 80, 443, 8080 only (brief) | unrestricted | unrestricted |

Cloud Run unit prices beyond the free tier (request-based, default rates): CPU
$0.000024 per vCPU-s active, memory $0.0000025 per GiB-s active, requests $0.40 per
million. Derived from the free allowance: 180,000 vCPU-s is 50 vCPU-hours a month, which
at 4 vCPU is 12.5 hours of active compute.

**Eliminated.** Render's free tier cannot hold the model. HF PRO is the only paid option
that keeps everything on Hugging Face, and it violates the free-only constraint and
ADR-0005's own first driver (a bill that stops means a dead URL in a paper).

## 3. Decision

### 3.0 Revision, same day

The operator declined to use `gcloud`. The decision below was the first recommendation;
it was not deployed. The hello-world acceptance test ran on **Render's free tier**
instead (§3.3). Everything below about Cloud Run stays as the record of why it was
recommended and remains the only documented free option that fits the model.

**Original recommendation: run the backend on Google Cloud Run, scaled to zero** (`--min-instances 0`), built from
`deployment/Dockerfile.serve`.

### 3.1 Cost

$0 while usage stays inside the free tier. It is not "no billing": Cloud Run requires a
billing account, and the free tier is aggregated per billing account. Mitigation: a $1
budget alert at 50 % and 100 % (`deployment/cloudrun.md` §2). **A budget alert notifies;
it does not stop spending.** Beyond the free tier the cost is the unit prices in §2.

### 3.2 Known risk: `/healthz` on `run.app`

Several public repositories report that Google's frontend answers `GET /healthz` on
`*.run.app` with its own 404 before the request reaches the container, and rename the
route to `/health`. I could not confirm this in Google's documentation (the Cloud Run
troubleshooting page does not mention it). The contract now has both `/health` and
`/healthz`, and the acceptance test requests both. If `/healthz` 404s on `run.app`, that
is recorded in §6 and the Phase 2 acceptance wording ("returns 200 from `/healthz`") is
met by `/health` with the deviation stated, rather than by editing the result.

### 3.3 What Render's free tier does and does not prove

It proves a containerised FastAPI is reachable on the public internet and serves
`/healthz`. It does **not** host the real service: the free instance has 512 MB of RAM
(`render.com/pricing`) and ESM-2 650M needs about 2.6 GB for fp32 weights alone (ADR-0001
§1.1). The smallest Render tier that clearly fits is 4 GB at $85/month. So no free host
that fits the real model has been *deployed and verified*. Cloud Run is the only free
option found that fits on paper (32 GiB, free tier), and it is not verified. Decide before
Phase 7: Cloud Run, a smaller checkpoint (ADR-0001 Option A trade-offs), or a paid tier.

## 4. What carries over from ADR-0005

- **§4.2 Frontend:** HF **Static Space**. The Space-creation message quoted above says
  Static Spaces are free for everyone.
- **§4.3 Durable artifacts:** HF Hub model and dataset repositories. Cloud Run's
  filesystem is ephemeral, so the registry bundle and evidence snapshots are pulled from
  the Hub at container start over HTTPS (port 443). That pull is part of the cold start
  and is **not yet measured**.
- **§4.4 cold-start policy:** the gallery is pre-computed and static; `/health` reports
  `warming` distinctly; no keep-alive pinger (`--min-instances 0` is deliberate).
- Environment variables `HF_*_REPO`, `ARTIFACT_CACHE_DIR`, `FRONTEND_ORIGIN` are unchanged.
  Cloud Run is told the container port with `--port 8000`, so `Dockerfile.serve` is
  unchanged.

## 5. Consequences

**Positive.** $0 at low traffic and 32 GiB available, with no 16 GB ceiling. No platform
rule about image size is documented.

**Negative.** A billing account and card are required. A scale-to-zero instance loses all
in-memory state, which is what ADR-0008 addresses. Cold start for the real image is
unknown and may be long (container start, then pulling ≈2.6 GB of weights from the Hub,
then loading them). Moving between hosts is a redeploy, since the image is the same.

## 6. Measurements

Cold-start figures must come from `scripts/measure_cold_start.py`; the phone test is by
hand. Steps are in `deployment/render.md`.

| Measurement | Value |
|---|---|
| `GET /healthz` on the public Render URL returns 200 | **200**, body `{"status":"degraded","registry_state":"empty",...}`. Run 2026-10-02 from the operator's PowerShell with `curl.exe` against `https://protloc-fusion.onrender.com` |
| `GET /health` on the same URL | **200** |
| `GET /healthz` returns 200 from a phone on mobile data | **UNVERIFIED — not yet tested** |
| Hello-world image, cold start from spin-down (Render free; client-observed) | **UNVERIFIED — `scripts/measure_cold_start.py` run in progress** |
| Cold start with a ≈1 GB image | **UNVERIFIED — not measured** |
| Cold start of the real serving image with weights | **UNVERIFIED — Phase 7+** |
| `/healthz` intercepted on `run.app` (Cloud Run) | **UNVERIFIED — Cloud Run not deployed** |

The two 200 responses reported `instance_uptime_seconds` of 87.3 and 89.9: the instance
had been up about 87 s when first asked. That shows the service was running, not how long
a cold start takes, and it is not recorded as one.

The hello-world cold start is a floor, not the serving image's cold start.

## 7. Revisit triggers

1. The Hugging Face Docker-Space requirement is reversed or confirmed by staff.
2. Measured cold start of the real image exceeds what ADR-0001 §4.3 allows (60 s).
3. Usage leaves the Cloud Run free tier.
4. Cloud Run's free tier or billing-account requirement changes.

## 8. Sources (retrieved 2026-10-02)

- HF forum, "Docker SDK now marked as 'Paid' when creating a new Space?":
  https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580
- HF forum, "Can Hugging Face Pro run a Docker Space on the free CPU Basic tier?":
  https://discuss.huggingface.co/t/can-hugging-face-pro-run-a-docker-space-on-the-free-cpu-basic-tier/177957
- HF pricing: https://huggingface.co/pricing
- HF Docker Spaces docs: https://huggingface.co/docs/hub/spaces-sdks-docker
- Cloud Run pricing: https://cloud.google.com/run/pricing
- Cloud Run quotas and limits: https://docs.cloud.google.com/run/quotas
- Render free tier: https://render.com/docs/free
- Render pricing: https://render.com/pricing
- `/healthz` reserved on `run.app` (community reports, not Google docs):
  https://github.com/dleandro/transfer-scout-api/issues/55
