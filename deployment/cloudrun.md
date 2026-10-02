# Cloud Run deployment and the Phase 2 acceptance test

Decision and evidence: `docs/adr/ADR-0007-hosting.md`. Run every command below in your
own PowerShell from the repository root (`C:\Users\saika\protloc-fusion`). Nothing here
needs a secret pasted anywhere: `gcloud auth login` opens a browser.

Docker is not needed for steps 1–5. It is needed only for the optional ballast test in
step 7.

## 1. Account and project

```powershell
gcloud auth login
$PROJECT = "protloc-hello-$(Get-Random -Maximum 99999)"
$REGION = "us-central1"
gcloud projects create $PROJECT
gcloud config set project $PROJECT
$BILLING = (gcloud billing accounts list --filter="open=true" --format="value(name)" | Select-Object -First 1)
gcloud billing projects link $PROJECT --billing-account=$BILLING
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com billingbudgets.googleapis.com
```

`$REGION` is `us-central1`, a Tier 1 region. Change it if you prefer; record the choice
in `docs/phases/02.md`.

## 2. $1 budget alert

```powershell
gcloud billing budgets create --billing-account=$BILLING --display-name="protloc-1usd" --budget-amount=1USD --filter-projects="projects/$PROJECT" --threshold-rule=percent=0.5 --threshold-rule=percent=1.0
```

A budget alert **emails you; it does not stop spending.** Delete the services (step 8)
when you are done measuring.

## 3. Deploy the hello-world service

```powershell
gcloud run deploy protloc-hello --project $PROJECT --region $REGION --source deployment/hello --allow-unauthenticated --min-instances 0 --max-instances 1 --cpu 1 --memory 512Mi --port 8080
$URL = (gcloud run services describe protloc-hello --project $PROJECT --region $REGION --format "value(status.url)")
$URL
```

## 4. Check both paths

```powershell
curl.exe -s -w "`n%{http_code}`n" "$URL/healthz"
curl.exe -s -w "`n%{http_code}`n" "$URL/health"
```

Expect JSON with `"registry_state":"empty"` and `200` for each. If `/healthz` returns an
HTML "Error 404 (Not Found)!!1" page and `/health` returns 200, Google's frontend is
intercepting `/healthz` (ADR-0007 §3.2). Record that as the result; do not retry.

## 5. Phone test

Turn Wi-Fi off. On mobile data open `$URL/healthz` in the phone's browser. Record whether
it showed the JSON. This is the acceptance test's "from a phone on mobile data" clause.

## 6. Cold-start measurement

```powershell
python scripts/measure_cold_start.py "$URL/healthz" --runs 5 --idle-seconds 1200 --out "$env:TEMP\cold_start_hello.json"
```

This takes about 1 h 45 min (five 20-minute idles). It makes two requests after each
idle: the first should hit a cold instance, the second a warm one. Cloud Run does not
guarantee shutdown after 20 minutes, so the script verifies coldness from the
`instance_uptime_seconds` field in the response and reports no cold figure if none of the
requests were cold. Paste the `summary:` line into `docs/phases/02.md` and the median
into ADR-0007 §6. If `/healthz` is intercepted, use `"$URL/health"` instead.

## 7. Optional: ≈1 GB image (needs Docker)

Measures image-pull cost at roughly the size of a PyTorch image, using an incompressible
ballast layer. It is a stand-in, not the real serving image.

```powershell
gcloud artifacts repositories create protloc --repository-format=docker --location=$REGION --project $PROJECT
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
$IMAGE = "$REGION-docker.pkg.dev/$PROJECT/protloc/hello-ballast:1g"
docker build --build-arg BALLAST_MB=1000 -t $IMAGE deployment/hello
docker push $IMAGE
gcloud run deploy protloc-hello-ballast --project $PROJECT --region $REGION --image $IMAGE --allow-unauthenticated --min-instances 0 --max-instances 1 --cpu 1 --memory 512Mi --port 8080
$URL2 = (gcloud run services describe protloc-hello-ballast --project $PROJECT --region $REGION --format "value(status.url)")
python scripts/measure_cold_start.py "$URL2/health" --runs 3 --idle-seconds 1200 --out "$env:TEMP\cold_start_ballast.json"
```

## 8. Clean up

```powershell
gcloud run services delete protloc-hello --project $PROJECT --region $REGION --quiet
gcloud run services delete protloc-hello-ballast --project $PROJECT --region $REGION --quiet
```

## Deploying the real serving image (later phases)

`deployment/Dockerfile.serve` listens on 8000. Deploy it with `--port 8000`, set
`CORS_ALLOWED_ORIGINS` to the frontend Static Space origin, and keep
`--min-instances 0`. Not done in Phase 2: no registry bundle exists until Phase 7.
