# Docker compose for the live app (feature 15)

Branch `phase/19f-docker-live-encoder` (with feature 13, `docs/phases/live-encoder.md`), from `phase/19e-dashboard-report`. Done 2026-10-24. Additive:
`Dockerfile.serve`, `Dockerfile.train` and `docker-compose.yml` (the Phase 1 contract app and its
CI checks) are unchanged; no backend, frontend, registry or dependency change.

## 1. Why a second stack

`Dockerfile.serve` runs `api/app.py` (the Phase 1 contract app) with no numpy and no `data/`, so
`docker compose up` never served the trained engine the five screens use. The live stack does.

## 2. What was built

- `deployment/Dockerfile.live`: the live API (`api/app_v2.py`). Base dependencies plus the
  `serve` extra (numpy); the build fails if torch is importable. The source keeps its repository
  layout under `/app` (`backend/protloc`, `backend/rules`, `configs/serving`) because the engine
  finds `engine.json` and the rule files relative to the source tree, so the `protloc` wheel is
  uninstalled after its dependencies resolve and `PYTHONPATH=/app/backend`. Non-root (uid 10001),
  `exec uvicorn` as PID 1, health check on `/demo/health` with a 300 s start period (first start
  hashes the residue cache and the index).
- `deployment/Dockerfile.web`: `npm ci` + `npm run build`, served by `nginx-unprivileged` on 8080.
- `deployment/nginx.live.conf`: SPA fallback, `/demo/` forwarded to `api:8010` (resolved per
  request through Docker DNS, so nginx starts before the API), caller's `X-Request-ID` kept or
  nginx's own id passed on, 4 MB body limit, hashed assets cached a year, `index.html` never.
- `deployment/docker-compose.live.yml`: `api` (8010) and `web` (8080); `data/` and `registry/`
  mounted read-only, only the evidence scratch directory writable (a named volume); paths set
  inside the container, everything else from `.env`; `web` waits for `api` to be healthy.
- `Dockerfile.live.dockerignore`, `Dockerfile.web.dockerignore`: allowlists, so `data/` (several
  GB) and `node_modules` never enter a build context. There is still no root `.dockerignore`.
- CI: `compose-valid` also validates the live compose file; new job `docker-live` builds both
  images, asserts no torch and that the live app imports, that the API refuses to start without
  a registry and says why, and that nginx passes `nginx -t`, serves the SPA (including a deep
  link) and answers 502 on `/demo/health` with no API behind it.

## 3. Known limit

Homology evidence is `degraded` inside the container: the evidence bundle pins the Windows
MMseqs2 binary by hash and a Linux container cannot run it. Every model output is served; the
banner, `/demo/health` and each result say evidence is unavailable. Native Windows serving keeps
live evidence. Changing that needs a Linux binary pinned in a new evidence bundle (a registry
change, out of scope here).

## 4. Checks (Linux VM; no Docker there)

- A venv with exactly the image's dependencies (base + `serve`, `protloc` uninstalled, source on
  `PYTHONPATH`): real uvicorn server, real registry and cache; `/demo/health` ok, a predict
  completed (`ER`), torch never imported.
- The same venv with an empty registry: exit code 3, log lists every missing file under
  "the service will not start".
- `docker-compose.live.yml` parses (2 services, 3 volumes); `nginx.live.conf` parses with
  crossplane inside an `http` block (status ok, no errors). `ci.yml` parses (13 jobs).

## 5. Fixed after the first Windows run

`docker compose up` with `ENCODER_MODE=live` in `.env` (set for a native run) crash-looped: the
base stack passed `.env` into the numpy-only image, the engine refused live mode without torch
(exit 3, as designed) and `web` never became healthy. The base stack now pins
`ENCODER_MODE=cache_only` and a blank `ESM_CACHE_DIR` (the `.env` value is a host path); only
`docker-compose.encoder.yml` turns live mode on, together with the torch image.

## 6. Verified on Windows, Docker Desktop (maintainer run, 2026-10-24)

With the fix above, `docker-compose.live.yml` + `docker-compose.encoder.yml` built the torch image,
loaded ESM-2 from `data/esm2`, passed the startup self-check and reported healthy; `/demo/health`
answered 200 to every 30 s health probe (JSON access log, one line each); `down` removed both
containers and the network cleanly.

## 7. UNVERIFIED

- A prediction through the containerised stack (port 8080), and the default (numpy-only) stack
  after the fix.
- First-start time over a Docker Desktop bind mount (hashing about 2.7 GB).
