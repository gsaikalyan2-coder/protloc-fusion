"""DEMO ENGINE — interpretable, motif-driven predictor.

What this is: a deterministic, rule/feature-based stand-in for the real ProtLoc
Fusion model (which does not exist yet — Phases 5-9 of PROJECT_PLAN.md). It
computes REAL sequence features (Kyte-Doolittle hydropathy, signal-peptide
h-region, NLS / KDEL / PTS1 / LPXTG motifs, TM segments, transit-peptide
composition) and maps them to the same seven output groups the real system
will produce. Because every score is driven by real features, mutations and
truncations change the output in biologically sensible ways — which is what
the Mutation Scanner screen demonstrates.

What this is NOT: the paper model. Nothing here is imported by backend/protloc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

AA = set("ACDEFGHIKLMNPQRSTVWY")

KD = {  # Kyte & Doolittle 1982 hydropathy scale
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Canonical order + names follow docs/DESIGN.md §2 (never reordered).
EUK_LOCS = ["CYT", "NUC", "EXT", "MEM", "MIT", "PLA", "ER", "LYS", "GOL", "PER"]
PROK_LOCS = ["CYT", "PCM", "PPL", "POM", "PCW", "EXT"]
LOC_NAMES = {
    "CYT": "Cytoplasm",
    "NUC": "Nucleus",
    "EXT": "Extracellular",
    "MEM": "Cell membrane",
    "MIT": "Mitochondrion",
    "PLA": "Plastid",
    "ER": "Endoplasmic reticulum",
    "LYS": "Lysosome / vacuole",
    "GOL": "Golgi apparatus",
    "PER": "Peroxisome",
    "PCM": "Cytoplasmic membrane",
    "PPL": "Periplasm",
    "POM": "Outer membrane",
    "PCW": "Cell wall / surface",
}
ORGANISM_GROUPS = {
    "eukaryote": {"label": "Animal / fungi", "locs": [l for l in EUK_LOCS if l != "PLA"]},
    "plant": {"label": "Plant", "locs": EUK_LOCS},
    "gram_neg": {"label": "Gram-negative bacteria", "locs": PROK_LOCS},
    "gram_pos": {
        "label": "Gram-positive bacteria",
        "locs": [l for l in PROK_LOCS if l not in ("PPL", "POM")],
    },
}

GO_MOTIFS = [  # (regex, GO id, term) — classic PROSITE-style patterns
    (r"[AG].{4}GK[ST]", "GO:0005524", "ATP binding (Walker A / P-loop)"),
    (r"C.{2,4}C.{12}H.{3,5}H", "GO:0008270", "zinc ion binding (C2H2 zinc finger)"),
    (r"G.S.G", "GO:0016787", "hydrolase activity (GxSxG nucleophile elbow)"),
    (r"[KR]{4}|PKKKRKV", "GO:0003677", "DNA binding (basic cluster)"),
]

NLS_RE = re.compile(r"PKKKRKV|[KR]{4,}|[KR]{2}.{10,12}[KR]{3}")
KDEL_RE = re.compile(r"[KH]DEL$")
PTS1_RE = re.compile(r"[SAC][KRH][LM]$")
LPXTG_RE = re.compile(r"LP.TG")


@dataclass
class Features:
    length: int
    hydropathy: list[float]
    sp_score: float
    h_region: tuple[int, int] | None
    cleavage: int | None
    tm_segments: list[tuple[int, int]]
    nls: list[tuple[int, int]]
    kdel: tuple[int, int] | None
    pts1: tuple[int, int] | None
    lpxtg: tuple[int, int] | None
    mtp_score: float
    ctp_score: float
    myristoyl: bool
    c_aromatic: bool
    basic_tail: bool
    go_hits: list[dict] = field(default_factory=list)


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def normalize(seq: str) -> str:
    s = re.sub(r"\s+", "", seq.upper())
    if s.startswith(">"):
        raise ValueError("Pass raw residues, not FASTA, to normalize()")
    bad = sorted(set(s) - AA - {"X", "U", "B", "Z"})
    if bad:
        raise ValueError(f"Invalid residue characters: {''.join(bad)}")
    return s.replace("U", "C").replace("B", "N").replace("Z", "Q")


def _win_mean(vals: list[float], start: int, w: int) -> float:
    return sum(vals[start : start + w]) / w


def _smooth(vals: list[float], w: int = 9) -> list[float]:
    half = w // 2
    out = []
    for i in range(len(vals)):
        seg = vals[max(0, i - half) : i + half + 1]
        out.append(round(sum(seg) / len(seg), 3))
    return out


def extract(seq: str) -> Features:
    L = len(seq)
    kd = [KD.get(a, 0.0) for a in seq]

    # --- Signal peptide: best hydrophobic 8-window inside residues 2..30
    best, best_i = -9.0, 0
    for i in range(1, max(2, min(32, L) - 11)):
        m = _win_mean(kd, i, 11)
        if m > best:
            best, best_i = m, i
    n_pos = sum(1 for a in seq[:best_i] if a in "KR")
    n_neg = sum(1 for a in seq[:best_i] if a in "DE")
    sp_score = sig(3.5 * (best - 2.3) + 0.5 * (n_pos - n_neg)) if L >= 30 else 0.0
    # A hydrophobic stretch long enough to span a bilayer is a signal ANCHOR (type II), not a cleavable SP
    if (
        L >= 40
        and max(_win_mean(kd, j, 19) for j in range(max(0, best_i - 6), min(best_i + 4, L - 19)))
        > 2.9
    ):
        sp_score *= 0.15
    h_region = (best_i, best_i + 11) if sp_score > 0.3 else None

    cleavage = None
    if h_region:
        h_end = h_region[1]
        for c in range(h_end + 2, min(h_end + 14, L - 1)):  # AxA-like: small at -3 and -1
            if seq[c - 3] in "AGSVTC" and seq[c - 1] in "AGSC":
                cleavage = c
                break
        if cleavage is None:
            cleavage = min(h_end + 6, L - 1)

    # --- TM helices: 19-window mean KD > 1.6, outside the signal peptide
    sp_end = cleavage if (cleavage and sp_score > 0.5) else 0
    tms: list[tuple[int, int]] = []
    i = sp_end
    while i + 19 <= L:
        if _win_mean(kd, i, 19) > 2.0:
            j = i
            while j + 19 <= L and _win_mean(kd, j, 19) > 2.0:
                j += 1
            tms.append((i, min(j + 18, L)))
            i = j + 19
        else:
            i += 1

    nls = [(m.start(), m.end()) for m in NLS_RE.finditer(seq)]
    k = KDEL_RE.search(seq)
    p = PTS1_RE.search(seq)
    lp = LPXTG_RE.search(seq[-40:]) if L > 40 else LPXTG_RE.search(seq)
    lp_span = None
    if lp:
        off = L - 40 if L > 40 else 0
        lp_span = (lp.start() + off, lp.end() + off)

    head = seq[:25]
    arg = sum(1 for a in head if a in "RK") / max(1, len(head))
    acid = sum(1 for a in head if a in "DE") / max(1, len(head))
    hyd = sum(1 for a in head if a in "LAIVF") / max(1, len(head))
    mtp = sig(28 * (arg - 0.15) - 30 * acid + 3 * (hyd - 0.3)) if L >= 40 else 0.0

    head40 = seq[1:41]
    st = sum(1 for a in head40 if a in "ST") / max(1, len(head40))
    acid40 = sum(1 for a in head40 if a in "DE") / max(1, len(head40))
    ctp = sig(20 * (st - 0.25) - 25 * acid40) if L >= 60 else 0.0

    go = []
    for rx, gid, term in GO_MOTIFS:
        m = re.search(rx, seq)
        if m:
            go.append({"go_id": gid, "term": term, "span": [m.start(), m.end()]})
    if len(tms) >= 6:
        go.append(
            {
                "go_id": "GO:0022857",
                "term": "transmembrane transporter activity (≥6 TM helices)",
                "span": [tms[0][0], tms[-1][1]],
            }
        )

    return Features(
        length=L,
        hydropathy=_smooth(kd),
        sp_score=sp_score,
        h_region=h_region,
        cleavage=cleavage,
        tm_segments=tms,
        nls=nls,
        kdel=(k.start(), k.end()) if k else None,
        pts1=(p.start(), p.end()) if p else None,
        lpxtg=lp_span,
        mtp_score=mtp,
        ctp_score=ctp,
        myristoyl=seq.startswith("MG"),
        c_aromatic=seq[-1:] in ("F", "W"),
        basic_tail=sum(1 for a in seq[-15:] if a in "KR") >= 6,
        go_hits=go,
    )


def raw_scores(f: Features, group: str) -> dict[str, float]:
    """Uncalibrated per-compartment probabilities (multi-label)."""
    sp = f.sp_score
    tms = [
        t for t in f.tm_segments if not (f.lpxtg and t[0] >= f.lpxtg[0])
    ]  # sortase C-tail is not a TM anchor
    n_tm = len(tms)
    tm = sig(3.0 * (n_tm - 0.5))
    kdel = 1.0 if f.kdel else 0.0
    pts1 = 1.0 if f.pts1 else 0.0
    nls = min(len(f.nls), 3)

    if group in ("gram_neg", "gram_pos"):
        s = {
            "PCM": tm * 0.92,
            "PPL": sp * (1 - tm) * (0.2 if f.c_aromatic else 0.8),
            "POM": sp * (1 - tm) * (0.95 if f.c_aromatic else 0.08),
            "PCW": sp * (0.9 if f.lpxtg else 0.06),
            "EXT": sp * (1 - tm) * (0.25 if group == "gram_neg" else (0.1 if f.lpxtg else 0.8)),
        }
        s["CYT"] = max(0.02, 0.95 - max(s.values()) * 1.05)
    else:
        mit = f.mtp_score * (1 - sp)
        pla = f.ctp_score * (1 - sp) * (1 - mit) if group == "plant" else 0.0
        type2 = 1.0 if (n_tm == 1 and tms[0][0] < 45) else 0.0
        s = {
            "NUC": sig(
                3.2 * nls - 1.6 - 5 * f.mtp_score
            ),  # not suppressed by SP on purpose: rule R01 catches the conflict
            "EXT": sp * (1 - tm) * (1 - kdel) * (1 - pts1) * 0.97,
            "MEM": tm * (0.35 if kdel else 0.9) * (1 - 0.5 * type2),
            "MIT": mit,
            "PLA": pla,
            "ER": sp * (0.88 if kdel else 0.12) + 0.3 * type2 * 0.2,
            "LYS": sp * (1 - tm) * 0.12,
            "GOL": 0.62 * type2,
            "PER": 0.86 * pts1 * (1 - sp),
        }
        s["NUC"] = max(s["NUC"], 0.02)
        s["CYT"] = max(0.03, 0.92 - max(s.values()) * 1.0)
    allowed = ORGANISM_GROUPS[group]["locs"]
    return {k: round(min(max(v, 0.005), 0.995), 4) for k, v in s.items() if k in allowed}


def membrane_type(f: Features) -> dict[str, float]:
    tm = sig(3.0 * (len(f.tm_segments) - 0.5))
    lipid = (0.7 if f.myristoyl else 0.05) * (1 - tm)
    periph = (0.55 if f.basic_tail else 0.06) * (1 - tm) * (1 - lipid)
    sol = max(0.01, 1 - tm - lipid - periph)
    tot = tm + lipid + periph + sol
    return {
        k: round(v / tot, 4)
        for k, v in {
            "soluble": sol,
            "peripheral": periph,
            "transmembrane": tm,
            "lipid_anchored": lipid,
        }.items()
    }


def sorting_signal(f: Features, group: str) -> dict:
    cands = {"SP": f.sp_score}
    if group in ("eukaryote", "plant"):
        cands["mTP"] = f.mtp_score * (1 - f.sp_score)
    if group == "plant":
        cands["cTP"] = f.ctp_score * (1 - f.sp_score)
    kind = max(cands, key=cands.get)
    p = cands[kind]
    if p < 0.5:
        return {
            "type": "none",
            "probability": round(1 - p, 4),
            "region": None,
            "cleavage_site": None,
            "candidates": {k: round(v, 4) for k, v in cands.items()},
        }
    if kind == "SP":
        region, cs = [0, f.cleavage], f.cleavage
    elif kind == "mTP":
        region, cs = [0, 25], 25
    else:
        region, cs = [0, 45], 45
    return {
        "type": kind,
        "probability": round(p, 4),
        "region": region,
        "cleavage_site": cs,
        "candidates": {k: round(v, 4) for k, v in cands.items()},
    }


def residue_attribution(seq: str, f: Features, top: str) -> list[float]:
    """Per-residue importance for the top call: which residues drive it."""
    w = [0.0] * len(seq)

    def paint(a: int, b: int, v: float) -> None:
        for i in range(max(0, a), min(len(seq), b)):
            w[i] = max(w[i], v)

    if top in ("EXT", "ER", "LYS", "PPL", "POM", "PCW") and f.h_region:
        paint(f.h_region[0], f.h_region[1], 1.0)
        paint(0, f.h_region[0], 0.5)
        if f.cleavage:
            paint(f.cleavage - 3, f.cleavage, 0.7)
    if top in ("ER",) and f.kdel:
        paint(*f.kdel, 1.0)
    if top == "NUC":
        for a, b in f.nls:
            paint(a, b, 1.0)
    if top in ("MEM", "PCM", "GOL"):
        for a, b in f.tm_segments:
            paint(a, b, 0.9)
    if top == "MIT":
        for i, a in enumerate(seq[:25]):
            w[i] = 1.0 if a in "RK" else 0.45 if a in "LAIVF" else 0.15
    if top == "PLA":
        for i, a in enumerate(seq[1:41], start=1):
            w[i] = 0.9 if a in "ST" else 0.2
    if top == "PER" and f.pts1:
        paint(*f.pts1, 1.0)
    if top == "PCW" and f.lpxtg:
        paint(*f.lpxtg, 1.0)
    if top == "POM" and f.c_aromatic:
        paint(len(seq) - 1, len(seq), 1.0)
    if top == "CYT":  # absence of signals: mild weight on the N-terminus
        paint(0, min(30, len(seq)), 0.25)
    return [round(x, 3) for x in w]
