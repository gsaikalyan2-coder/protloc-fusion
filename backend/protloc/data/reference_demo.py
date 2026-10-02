"""DEMO DATA — synthetic, constructed-by-design sequences.

Every sequence here is GENERATED with a planted, known targeting signal, so its
label is true by construction. Nothing is copied from UniProt. Three sets:

  REFERENCE  — 14 families x 5 members: the local "evidence" database
  EXAMPLES   — gallery sequences (mutated family members, so evidence hits exist)
  BENCHMARK  — 360 labelled sequences with 8% label noise, split 50/50 into
               'calibration' (fit T and tau) and 'test' (report only).
"""

from __future__ import annotations

import random

BG = "AAAACDDDEEEEFGGGGHHIKKKLLLMNNNPPPQQQRRRSSSSSTTTTVVWYY"
HYD = "LLLLAAIVVFFM"


def _bg(r: random.Random, n: int) -> str:
    return "".join(r.choice(BG) for _ in range(n))


def _sp(r: random.Random) -> str:  # n-region + h-region + AxA cleavage
    return (
        "M"
        + "".join(r.choice("KR") for _ in range(2))
        + "".join(r.choice(HYD) for _ in range(13))
        + r.choice("SAG")
        + "A"
        + r.choice("SQ")
        + "A"
    )


def _tm(r: random.Random) -> str:
    return "".join(r.choice(HYD) for _ in range(21))


def build(label: str, r: random.Random, n: int = 220) -> str:
    if label == "EXT":
        return _sp(r) + _bg(r, n)
    if label == "ER":
        return _sp(r) + _bg(r, n) + "KDEL"
    if label == "NUC":
        k = r.randint(12, 30)
        return "M" + _bg(r, k) + "PKKKRKV" + _bg(r, n - k)
    if label == "MIT":
        head = "M" + "".join(r.choice("RRRLLSSRLGRSQ") for _ in range(24))
        return head + _bg(r, n)
    if label == "PER":
        return "M" + _bg(r, n) + "SKL"
    if label == "MEM":
        out = "M" + _bg(r, 40)
        for _ in range(r.randint(2, 7)):
            out += _tm(r) + "".join(r.choice("KRNGDEST") for _ in range(12))
        return out + _bg(r, 40)
    if label == "PLA":
        return "MA" + "".join(r.choice("SSSTTAVRLPG") for _ in range(44)) + _bg(r, n)
    if label == "GOL":
        return "M" + "".join(r.choice("STNQK") for _ in range(12)) + _tm(r) + _bg(r, n)
    if label == "PCW":
        return _sp(r) + _bg(r, n) + "LPETG" + _tm(r)[:18] + "KRKE"
    if label == "POM":
        return _sp(r) + _bg(r, n) + "YQF"
    if label == "PPL":
        return _sp(r) + _bg(r, n) + "K"
    if label == "PCM":
        return "M" + _bg(r, 20) + _tm(r) + _bg(r, 30) + _tm(r) + _bg(r, 60)
    # CYT: acidic-leaning background, no signals
    return "MS" + _bg(r, n).replace("K", "E")


def mutate(seq: str, r: random.Random, rate: float) -> str:
    """Point-mutate the body but keep first 50 and last 30 residues (the signals)."""
    s = list(seq)
    for i in range(50, len(s) - 30):
        if r.random() < rate:
            s[i] = r.choice(BG)
    return "".join(s)


FAMILIES = [
    ("eukaryote", "EXT"),
    ("eukaryote", "ER"),
    ("eukaryote", "NUC"),
    ("eukaryote", "MIT"),
    ("eukaryote", "PER"),
    ("eukaryote", "MEM"),
    ("eukaryote", "GOL"),
    ("eukaryote", "CYT"),
    ("plant", "PLA"),
    ("gram_neg", "POM"),
    ("gram_neg", "PPL"),
    ("gram_neg", "PCM"),
    ("gram_pos", "PCW"),
    ("gram_pos", "CYT"),
]


def _reference() -> tuple[list[dict], dict[str, str]]:
    r = random.Random(7)
    refs, backbones = [], {}
    for fi, (grp, lab) in enumerate(FAMILIES):
        root = build(lab, r)
        backbones[f"F{fi:02d}"] = root
        for m in range(5):
            refs.append(
                {
                    "accession": f"DEMO_R{fi:02d}{m}",
                    "family": f"F{fi:02d}",
                    "organism_group": grp,
                    "location": lab,
                    "evidence_code": "ECO:0000269" if m < 2 else "ECO:0000250",
                    "sequence": mutate(root, r, 0.06 * (m + 1)),
                }
            )
    return refs, backbones


REFERENCE, _BACKBONES = _reference()


def _examples() -> list[dict]:
    r = random.Random(11)
    specs = [
        ("secreted_enzyme", "Secreted hydrolase-like protein", "eukaryote", 0, 0.25),
        ("er_resident", "ER-retained chaperone-like protein", "eukaryote", 1, 0.25),
        ("nuclear_factor", "Nuclear transcription-factor-like protein", "eukaryote", 2, 0.3),
        ("mito_matrix", "Mitochondrial matrix enzyme-like protein", "eukaryote", 3, 0.3),
        ("peroxisomal", "Peroxisomal oxidase-like protein", "eukaryote", 4, 0.3),
        ("multi_pass", "Multi-pass membrane transporter-like protein", "eukaryote", 5, 0.2),
        ("chloroplast", "Chloroplast stromal protein-like", "plant", 8, 0.3),
        ("outer_membrane", "Outer-membrane porin-like protein", "gram_neg", 9, 0.3),
        ("sortase_anchored", "LPXTG surface-anchored protein", "gram_pos", 12, 0.3),
    ]
    out = []
    for key, name, grp, fam, rate in specs:
        out.append(
            {
                "id": key,
                "name": name,
                "organism_group": grp,
                "sequence": mutate(_BACKBONES[f"F{fam:02d}"], r, rate),
                "note": f"Synthetic; mutated member of demo family F{fam:02d}",
            }
        )
    # A deliberately contradictory one: signal peptide AND a strong NLS → consistency flag
    conflict = build("EXT", r)
    conflict = conflict[:80] + "PKKKRKV" + conflict[87:120] + "KRKRK" + conflict[125:]
    out.append(
        {
            "id": "conflicted",
            "name": "Contradictory construct (SP + NLS)",
            "organism_group": "eukaryote",
            "sequence": conflict,
            "note": "Built to trigger a consistency rule",
        }
    )
    return out


EXAMPLES = _examples()


def _benchmark() -> list[dict]:
    r = random.Random(2026)
    grp_labels = {
        "eukaryote": ["EXT", "ER", "NUC", "MIT", "PER", "MEM", "GOL", "CYT"],
        "plant": ["PLA", "CYT", "EXT"],
        "gram_neg": ["POM", "PPL", "PCM", "CYT"],
        "gram_pos": ["PCW", "CYT", "EXT"],
    }
    rows = []
    i = 0
    while len(rows) < 360:
        for grp, labs in grp_labels.items():
            lab = r.choice(labs)
            seq = build(lab, r, n=r.randint(150, 320))
            if r.random() < 0.15:  # weaken the signal sometimes → realistic hard cases
                seq = seq[:1] + _bg(r, 12) + seq[13:]
            true = lab if r.random() > 0.08 else r.choice(labs)  # label noise
            rows.append(
                {
                    "id": f"B{i:04d}",
                    "organism_group": grp,
                    "label": true,
                    "sequence": seq,
                    "split": "calibration" if i % 2 == 0 else "test",
                }
            )
            i += 1
    return rows[:360]


BENCHMARK = _benchmark()
