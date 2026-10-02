# Render deployment and the Phase 2 acceptance test

Decision and evidence: `docs/adr/ADR-0007-hosting.md` (§3.0, §3.3). Render's free tier has
512 MB of RAM, so this proves hosting of the hello-world image only, not the real service.
`deployment/cloudrun.md` is the alternative runbook.

## Deploy (Render dashboard)

1. Push the branch: `git push -u origin phase/02-hosting`.
2. render.com → sign in with GitHub → New → Web Service → `gsaikalyan2-coder/protloc-fusion`.
3. Branch `phase/02-hosting`, Language `Docker`, Root Directory `deployment/hello`,
   Instance Type `Free`. Deploy. The Dockerfile reads Render's `PORT`.

## Test (PowerShell, repository root)

```powershell
$URL = "https://protloc-fusion.onrender.com"
curl.exe -s -w "`
%{http_code}`
" "$URL/healthz"
curl.exe -s -w "`
%{http_code}`
" "$URL/health"
```

Phone: Wi-Fi off, open `$URL/healthz` on mobile data.

## Cold start

Render spins a free service down after 15 minutes without inbound traffic.

```powershell
python scripts/measure_cold_start.py "$URL/healthz" --runs 1 --idle-seconds 1000
```

The script counts a request as cold only when `instance_uptime_seconds` in the response is
under 10.
