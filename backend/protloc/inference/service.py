"""DEMO SERVICE — turns engine features into the full seven-part result and
implements the novel features:

  predict()    — calibrated multi-label call + abstention + consistency flags
                 + per-residue attribution + homology evidence (Inspector, Evidence)
  scan()       — counterfactual mutation / truncation scanner (Scanner)
  triage()     — rank a batch by how much human attention each protein needs (Triage)
  landscape()  — dual-targeting co-occurrence matrix across a batch (Triage panel)
  TRUST        — temperature + tau fitted on the calibration split, reported on
                 the untouched test split (Trust)
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
from pathlib import Path

from protloc.data import reference_demo as data
from protloc.domain.types import ConfidenceBand, EngineSource, ThresholdPolicy
from protloc.features.classical import (
    LOC_NAMES,
    ORGANISM_GROUPS,
    extract,
    logit,
    membrane_type,
    normalize,
    raw_scores,
    residue_attribution,
    sig,
    sorting_signal,
)

RULES = json.loads((Path(__file__).resolve().parents[2] / "rules" / "consistency.json").read_text())
MODEL_VERSION = "demo-0.1.0"
EXPERIMENT_ID = "DEMO-PRESENTATION"

# Phase 3 provenance. The motif engine has one tau, fitted on a synthetic split, and
# no calibrator in the contract's sense: `calibrated` is False and there is no
# calibrator to version. Phases 9-14 replace these with the registry bundle's values.
ENGINE_SOURCE = EngineSource.MOTIF.value
CALIBRATED = False
CALIBRATION_VERSION: str | None = None
THRESHOLD_POLICY = ThresholdPolicy.FIXED.value
#: The non-localization fields the engine reports, each with the same provenance.
TASK_SOURCE_KEYS = ("localization", "membrane_type", "signal_peptide", "go_function")


def confidence_band(probability: float, threshold: float) -> str:
    """Coarse band for one label, derived from its threshold and nothing else.

    Below the threshold the system is abstaining on the label: ``none``. Above it,
    the interval from the threshold to 1 is split in thirds. No cut-off is a free
    constant, so no tau is hardcoded; Phase 10 replaces this with the calibrator's
    own metadata.
    """
    if probability < threshold:
        return ConfidenceBand.NONE.value
    third = (1.0 - threshold) / 3.0
    if probability < threshold + third:
        return ConfidenceBand.LOW.value
    if probability < threshold + 2.0 * third:
        return ConfidenceBand.MEDIUM.value
    return ConfidenceBand.HIGH.value


# ───────────────────────── calibration (fit once at import) ─────────────────────────
def _calibrate(p: float, T: float) -> float:
    return sig(logit(p) / T)


def _bce(rows: list[tuple[dict, str]], T: float) -> float:
    loss = 0.0
    for scores, label in rows:
        for loc, p in scores.items():
            q = min(max(_calibrate(p, T), 1e-6), 1 - 1e-6)
            y = 1.0 if loc == label else 0.0
            loss -= y * math.log(q) + (1 - y) * math.log(1 - q)
    return loss


def _reliability(
    rows: list[tuple[dict, str]], T: float, bins: int = 10
) -> tuple[list[dict], float]:
    buckets = [[0, 0.0, 0.0] for _ in range(bins)]
    n = 0
    for scores, label in rows:
        for loc, p in scores.items():
            q = _calibrate(p, T)
            b = min(int(q * bins), bins - 1)
            buckets[b][0] += 1
            buckets[b][1] += q
            buckets[b][2] += 1.0 if loc == label else 0.0
            n += 1
    out, ece = [], 0.0
    for i, (c, conf, acc) in enumerate(buckets):
        if c:
            ece += (c / n) * abs(acc / c - conf / c)
        out.append(
            {
                "bin": i,
                "lo": i / bins,
                "hi": (i + 1) / bins,
                "count": c,
                "confidence": round(conf / c, 4) if c else None,
                "accuracy": round(acc / c, 4) if c else None,
            }
        )
    return out, round(ece, 4)


def _selective(rows: list[tuple[dict, str]], T: float, tau: float) -> tuple[float, float]:
    kept = correct = 0
    for scores, label in rows:
        cal = {k: _calibrate(v, T) for k, v in scores.items()}
        top = max(cal, key=cal.get)
        if cal[top] >= tau:
            kept += 1
            correct += top == label
    return (kept / len(rows), (correct / kept) if kept else 1.0)


def _fit() -> dict:
    split = {"calibration": [], "test": []}
    for b in data.BENCHMARK:
        f = extract(b["sequence"])
        split[b["split"]].append((raw_scores(f, b["organism_group"]), b["label"]))
    cal, test = split["calibration"], split["test"]
    grid = [round(0.4 + 0.05 * i, 2) for i in range(53)]
    T = min(grid, key=lambda t: _bce(cal, t))
    # tau: smallest threshold whose selective accuracy on CALIBRATION reaches 0.90
    taus = [round(0.30 + 0.01 * i, 2) for i in range(66)]
    tau = next((t for t in taus if _selective(cal, T, t)[1] >= 0.85), 0.6)
    rel_before, ece_before = _reliability(test, 1.0)
    rel_after, ece_after = _reliability(test, T)
    curve = []
    for t in taus[::3]:
        cov, acc = _selective(test, T, t)
        curve.append({"tau": t, "coverage": round(cov, 4), "selective_accuracy": round(acc, 4)})
    cov, acc = _selective(test, T, tau)
    return {
        "temperature": T,
        "tau": tau,
        "tau_rule": "min tau with selective accuracy >= 0.85 on calibration split",
        "n_calibration": len(cal),
        "n_test": len(test),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "reliability_before": rel_before,
        "reliability_after": rel_after,
        "coverage_curve": curve,
        "test_at_tau": {"coverage": round(cov, 4), "selective_accuracy": round(acc, 4)},
        "label_noise": 0.08,
        "threshold_policy": THRESHOLD_POLICY,
        "calibration_version": CALIBRATION_VERSION,
        "calibrated": CALIBRATED,
    }


TRUST = _fit()
CONFIG_HASH = hashlib.sha256(
    json.dumps(
        {
            "T": TRUST["temperature"],
            "tau": TRUST["tau"],
            "rules": RULES["rules_version"],
            "model": MODEL_VERSION,
        },
        sort_keys=True,
    ).encode()
).hexdigest()[:12]


def provenance() -> dict:
    return {
        "model_version": MODEL_VERSION,
        "config_hash": CONFIG_HASH,
        "rules_version": RULES["rules_version"],
        "experiment_id": EXPERIMENT_ID,
        "predictor": "interpretable motif engine (demo stub, not the trained model)",
    }


# ───────────────────────── evidence ─────────────────────────
def _kmers(s: str, k: int = 3) -> set[str]:
    return {s[i : i + k] for i in range(len(s) - k + 1)}


_REF_KMERS = [(_kmers(r["sequence"]), r) for r in data.REFERENCE]


def evidence(seq: str, group: str, top_k: int = 5) -> list[dict]:
    q = _kmers(seq)
    pre = sorted(_REF_KMERS, key=lambda kr: -len(q & kr[0]) / max(1, len(q | kr[0])))[:15]
    hits = []
    for km, ref in pre:
        sm = difflib.SequenceMatcher(None, seq, ref["sequence"], autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        identity = matched / max(len(seq), len(ref["sequence"]))
        hits.append(
            {
                "accession": ref["accession"],
                "family": ref["family"],
                "organism_group": ref["organism_group"],
                "location": ref["location"],
                "location_name": LOC_NAMES[ref["location"]],
                "evidence_code": ref["evidence_code"],
                "identity": round(identity, 4),
                "kmer_jaccard": round(len(q & km) / max(1, len(q | km)), 4),
                "coverage": round(
                    min(len(seq), len(ref["sequence"])) / max(len(seq), len(ref["sequence"])), 4
                ),
                "same_group": ref["organism_group"] == group,
                "aligned_blocks": [
                    [b.a, b.a + b.size] for b in sm.get_matching_blocks() if b.size >= 8
                ][:40],
            }
        )
    hits.sort(key=lambda h: -h["identity"])
    return hits[:top_k]


# ───────────────────────── rules ─────────────────────────
def _consistency(
    f, top: str, calibrated: dict, sort_sig: dict, mem: dict, ev: list[dict], tau: float
) -> list[dict]:
    fired = []
    for r in RULES["rules"]:
        w, hit = r["when"], False
        if w == "loc_and_signal":
            hit = calibrated.get(r["loc"], 0) >= tau and f.sp_score > 0.5
        elif w == "retention_without_entry":
            hit = f.kdel is not None and f.sp_score < 0.3
        elif w == "dual_nterminal":
            hit = f.sp_score > 0.5 and f.mtp_score > 0.5
        elif w == "tm_but_secreted":
            hit = mem["transmembrane"] > 0.5 and top in ("EXT", "PPL")
        elif w == "pts1_with_sp":
            hit = f.pts1 is not None and f.sp_score > 0.5
        elif w == "evidence_disagrees":
            hit = bool(ev) and ev[0]["identity"] >= 0.4 and ev[0]["location"] != top
        if hit:
            fired.append({"rule_id": r["id"], "severity": r["severity"], "message": r["message"]})
    return fired


# ───────────────────────── predict ─────────────────────────
def predict(seq_raw: str, group: str, seq_id: str = "query", with_evidence: bool = True) -> dict:
    if group not in ORGANISM_GROUPS:
        raise ValueError(f"Unknown organism_group '{group}'")
    seq = normalize(seq_raw)
    if not 30 <= len(seq) <= 3000:
        raise ValueError(f"Sequence length {len(seq)} outside supported range 30–3000")
    f = extract(seq)
    raw = raw_scores(f, group)
    T, tau = TRUST["temperature"], TRUST["tau"]
    cal = {k: round(_calibrate(v, T), 4) for k, v in raw.items()}
    order = ORGANISM_GROUPS[group]["locs"]
    top = max(cal, key=cal.get)
    abstained = cal[top] < tau
    mem = membrane_type(f)
    ss = sorting_signal(f, group)
    ev = evidence(seq, group) if with_evidence else []
    flags = _consistency(f, top, cal, ss, mem, ev, tau)
    ev_agree = None
    if ev:
        ev_agree = {
            "top_hit_location": ev[0]["location"],
            "agrees": ev[0]["location"] == top,
            "identity": ev[0]["identity"],
        }
    return {
        "id": seq_id,
        "length": len(seq),
        "sequence": seq,
        "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
        "organism_group": group,
        "localization": [
            {
                "code": k,
                "name": LOC_NAMES[k],
                "probability": cal[k],
                "raw": raw[k],
                "predicted": cal[k] >= tau,
                "source": ENGINE_SOURCE,
                "threshold": tau,
                "calibrated": CALIBRATED,
                "confidence_band": confidence_band(cal[k], tau),
            }
            for k in order
            if k in cal
        ],
        "top_location": top,
        "abstained": abstained,
        "no_confident_call": all(cal[k] < tau for k in order if k in cal),
        "calibration_version": CALIBRATION_VERSION,
        "threshold_policy": THRESHOLD_POLICY,
        "task_sources": {
            k: {"source": ENGINE_SOURCE, "calibrated": CALIBRATED} for k in TASK_SOURCE_KEYS
        },
        "tau": tau,
        "temperature": T,
        "membrane_type": mem,
        "sorting_signal": ss,
        "go_terms": f.go_hits,
        "features": {
            "signal_peptide_score": round(f.sp_score, 4),
            "h_region": f.h_region,
            "tm_segments": f.tm_segments,
            "nls": f.nls,
            "kdel": f.kdel,
            "pts1": f.pts1,
            "lpxtg": f.lpxtg,
            "mtp_score": round(f.mtp_score, 4),
            "ctp_score": round(f.ctp_score, 4),
        },
        "hydropathy": f.hydropathy,
        "attribution": residue_attribution(seq, f, top),
        "consistency_flags": flags,
        "evidence": ev,
        "evidence_agreement": ev_agree,
        "provenance": provenance(),
    }


# ───────────────────────── scanner ─────────────────────────
def _quick(seq: str, group: str) -> dict[str, float]:
    f = extract(seq)
    return {k: _calibrate(v, TRUST["temperature"]) for k, v in raw_scores(f, group).items()}


def parse_mutations(spec: str, seq: str) -> list[tuple[int, str, str]]:
    muts = []
    for tok in [t.strip().upper() for t in spec.replace(";", ",").split(",") if t.strip()]:
        if len(tok) < 3 or not tok[1:-1].isdigit():
            raise ValueError(f"Bad mutation '{tok}'. Use e.g. K83A")
        wt, pos, alt = tok[0], int(tok[1:-1]), tok[-1]
        if not 1 <= pos <= len(seq):
            raise ValueError(f"Position {pos} out of range 1–{len(seq)}")
        if seq[pos - 1] != wt:
            raise ValueError(f"{tok}: residue {pos} is {seq[pos - 1]}, not {wt}")
        muts.append((pos, wt, alt))
    return muts


def scan(
    seq_raw: str, group: str, target: str | None, mutations: str, substitute: str = "A"
) -> dict:
    seq = normalize(seq_raw)
    if len(seq) > 1200:
        raise ValueError("Scanner supports sequences up to 1200 residues")
    base = _quick(seq, group)
    target = target or max(base, key=base.get)
    if target not in base:
        raise ValueError(f"Target {target} not valid for group {group}")
    # 1) N-terminal truncation series
    trunc = []
    for cut in range(0, min(61, len(seq) - 30), 3):
        s = _quick("M" + seq[cut + 1 :] if cut else seq, group)
        trunc.append({"removed": cut, **{k: round(v, 4) for k, v in s.items()}})
    # 2) substitution scan: every position -> `substitute`
    sub = substitute.upper()[:1] or "A"
    deltas = []
    for i in range(len(seq)):
        if seq[i] == sub:
            deltas.append(0.0)
            continue
        mseq = seq[:i] + sub + seq[i + 1 :]
        deltas.append(round(_quick(mseq, group)[target] - base[target], 4))
    ranked = sorted(range(len(seq)), key=lambda i: deltas[i])[:10]
    # 3) user variant
    variant = None
    if mutations.strip():
        muts = parse_mutations(mutations, seq)
        vs = list(seq)
        for pos, _, alt in muts:
            vs[pos - 1] = alt
        vseq = "".join(vs)
        wt_full = predict(seq, group, "wild_type", with_evidence=False)
        mt_full = predict(vseq, group, "variant", with_evidence=False)
        variant = {
            "mutations": [f"{w}{p}{a}" for p, w, a in muts],
            "wild_type": {
                k: wt_full[k]
                for k in (
                    "localization",
                    "top_location",
                    "abstained",
                    "sorting_signal",
                    "membrane_type",
                    "consistency_flags",
                )
            },
            "variant": {
                k: mt_full[k]
                for k in (
                    "localization",
                    "top_location",
                    "abstained",
                    "sorting_signal",
                    "membrane_type",
                    "consistency_flags",
                )
            },
            "relocalized": wt_full["top_location"] != mt_full["top_location"],
        }
    return {
        "length": len(seq),
        "sequence": seq,
        "organism_group": group,
        "target": target,
        "target_name": LOC_NAMES[target],
        "baseline": {k: round(v, 4) for k, v in base.items()},
        "substitute": sub,
        "position_deltas": deltas,
        "most_sensitive": [
            {"position": i + 1, "residue": seq[i], "delta": deltas[i]} for i in ranked
        ],
        "truncation": trunc,
        "variant": variant,
        "tau": TRUST["tau"],
        "provenance": provenance(),
    }


# ───────────────────────── triage + landscape ─────────────────────────
def attention_score(r: dict) -> tuple[float, list[str]]:
    reasons, score = [], 0.0
    if r["abstained"]:
        score += 3.0
        reasons.append("abstained")
    for fl in r["consistency_flags"]:
        score += {"blocked": 4.0, "warning": 2.0, "uncertain": 1.0}[fl["severity"]]
        reasons.append(f"{fl['rule_id']} {fl['severity']}")
    ev = r["evidence_agreement"]
    if ev is None or ev["identity"] < 0.3:
        score += 1.0
        reasons.append("no close homolog")
    predicted = [l for l in r["localization"] if l["predicted"]]
    if len(predicted) >= 2:
        score += 1.5
        reasons.append("multi-compartment")
    top_p = max(l["probability"] for l in r["localization"])
    score += round(1 - top_p, 3)
    return round(score, 3), reasons


def summarize(r: dict) -> dict:
    score, reasons = attention_score(r)
    return {
        "id": r["id"],
        "length": r["length"],
        "top_location": r["top_location"],
        "top_probability": max(l["probability"] for l in r["localization"]),
        "predicted": [l["code"] for l in r["localization"] if l["predicted"]],
        "abstained": r["abstained"],
        "signal": r["sorting_signal"]["type"],
        "membrane": max(r["membrane_type"], key=r["membrane_type"].get),
        "n_flags": len(r["consistency_flags"]),
        "worst_flag": min(
            (f["severity"] for f in r["consistency_flags"]),
            key=["blocked", "warning", "uncertain"].index,
            default=None,
        ),
        "evidence_agrees": r["evidence_agreement"]["agrees"] if r["evidence_agreement"] else None,
        "attention": score,
        "reasons": reasons,
    }


def landscape(results: list[dict]) -> dict:
    codes = [l["code"] for l in results[0]["localization"]] if results else []
    matrix = {a: dict.fromkeys(codes, 0) for a in codes}
    for r in results:
        pred = [l["code"] for l in r["localization"] if l["predicted"]]
        for a in pred:
            for b in pred:
                matrix[a][b] += 1
    counts = {c: matrix[c][c] for c in codes}
    return {
        "codes": codes,
        "names": {c: LOC_NAMES[c] for c in codes},
        "matrix": matrix,
        "counts": counts,
        "abstained": sum(1 for r in results if r["abstained"]),
        "total": len(results),
    }


def parse_fasta(text: str, limit: int) -> list[tuple[str, str]]:
    recs, cur_id, buf = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                recs.append((cur_id, "".join(buf)))
            cur_id, buf = line[1:].split()[0] if line[1:].split() else f"seq{len(recs) + 1}", []
        else:
            if cur_id is None:
                cur_id = "seq1"
            buf.append(line)
    if cur_id is not None:
        recs.append((cur_id, "".join(buf)))
    if not recs:
        raise ValueError("No sequences found")
    if len(recs) > limit:
        raise ValueError(f"{len(recs)} sequences exceeds the demo batch limit of {limit}")
    return recs
