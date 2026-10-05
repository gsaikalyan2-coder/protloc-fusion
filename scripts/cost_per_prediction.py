"""Cost per prediction from measured compute time and published Cloud Run rates.

Reads ``data/manifests/serving_measurements.json``. Standard library only.
Prices are Cloud Run request-based, Tier 1, and are an INPUT you can override:
they are not measured and may change. Timings come from one local Windows
machine, not the deployment host, and cover the 500-sequence subset (500), not
comparable to DeepLoc or to the Phase 8 full-corpus numbers.

    python scripts/cost_per_prediction.py --vcpu 2 --gib 8
"""

import argparse
import json
from pathlib import Path
import sys

MEASUREMENTS = Path("data/manifests/serving_measurements.json")
FREE_VCPU_S = 180_000.0
FREE_GIB_S = 360_000.0
FREE_REQUESTS = 2_000_000.0


def main() -> int:
    """Compute the cost table, write the JSON report, print the table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vcpu", type=float, default=2.0)
    p.add_argument("--gib", type=float, default=8.0)
    p.add_argument("--vcpu-rate", type=float, default=0.000024)
    p.add_argument("--gib-rate", type=float, default=0.0000025)
    p.add_argument("--request-rate-per-million", type=float, default=0.40)
    p.add_argument("--out", type=Path, default=Path("data/manifests/cost_per_prediction.json"))
    a = p.parse_args()

    m = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    rows = []
    for row in m["batch_sweep"]:
        n = int(row["sequences"])
        secs = float(row["compute_s"]["median"])
        cost = secs * (a.vcpu * a.vcpu_rate + a.gib * a.gib_rate) + a.request_rate_per_million / 1e6
        rows.append(
            {
                "sequences": n,
                "compute_s_median": secs,
                "cost_usd_per_job": round(cost, 8),
                "cost_usd_per_prediction": round(cost / n, 8),
                "predictions_per_hour_one_instance": round(3600 / secs * n, 1),
                "free_tier_predictions_per_month": int(
                    min(
                        FREE_VCPU_S / (secs * a.vcpu),
                        FREE_GIB_S / (secs * a.gib),
                        FREE_REQUESTS,
                    )
                    * n
                ),
            }
        )
    report = {
        "scope": m["scope"],
        "machine_measured_on": m["machine"],
        "assumptions": {
            "vcpu": a.vcpu,
            "gib": a.gib,
            "vcpu_rate_usd_per_s": a.vcpu_rate,
            "gib_rate_usd_per_s": a.gib_rate,
            "request_rate_usd_per_million": a.request_rate_per_million,
            "note": "rates are published list prices, not measured; compute time is from the "
            "measuring machine, not Cloud Run; idle and cold-start time are not billed here",
        },
        "rows": rows,
    }
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    header = (
        f"{'seqs':>5} {'compute s':>10} {'$/job':>12} "
        f"{'$/prediction':>14} {'free-tier preds/mo':>20}"
    )
    print(header)  # noqa: T201
    for r in rows:
        print(  # noqa: T201
            f"{r['sequences']:>5} {r['compute_s_median']:>10} {r['cost_usd_per_job']:>12} "
            f"{r['cost_usd_per_prediction']:>14} {r['free_tier_predictions_per_month']:>20}"
        )
    print(f"\n{m['scope']}\nwrote {a.out}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
