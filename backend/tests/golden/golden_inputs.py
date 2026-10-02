"""Golden-set inputs: every sequence, how it was made, and what it is meant to probe.

Honesty rule. No sequence here is described as a real protein. The 10 gallery
sequences and everything derived from them are SYNTHETIC: ``reference_demo`` plants
a known targeting signal in generated background residues (nothing is copied from
UniProt). ``origin`` says which kind each entry is:

* ``gallery``    -- one of ``reference_demo.EXAMPLES``, verbatim.
* ``synthetic``  -- a fresh ``reference_demo.build(label, Random(seed))`` sequence.
* ``engineered`` -- a documented, deterministic edit of a gallery or synthetic
  sequence (point mutant, truncation, poly-residue, boundary length, chimera,
  organism-group swap, rejected input).
* ``real``       -- a reviewed database entry. None exist: the one attempt to fetch
  reviewed UniProt entries was refused by the sandbox proxy (HTTP 403), so the gap
  is recorded in ``docs/phases/04.md``. An entry with this origin must carry a
  ``source`` with an accession; ``test_golden_coverage`` enforces it.

Categories. ``declared_categories`` is the intent of the construction (what the
planted signal is), not ground truth, and is never asserted against the engine's
output. The categories the engine's behaviour defines -- ``gram_pos``, ``gram_neg``,
``multi_label``, ``ambiguous`` -- are derived from each entry's recorded output by
``golden_lib.derive_categories`` and are not declared here.

This module is test support. It is deterministic: running it twice gives identical
entries, and ``test_golden_coverage`` fails if the stored inputs drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from protloc.data import reference_demo

#: Intent of each gallery sequence (the planted signal), keyed by gallery id.
GALLERY_DECLARED: dict[str, tuple[str, ...]] = {
    "secreted_enzyme": ("secreted",),
    "er_resident": ("other_compartment",),
    "nuclear_factor": ("other_compartment",),
    "mito_matrix": ("mitochondrial",),
    "peroxisomal": ("other_compartment",),
    "multi_pass": ("membrane",),
    "chloroplast": ("chloroplast",),
    "outer_membrane": ("membrane",),
    "sortase_anchored": ("other_compartment",),
    "conflicted": ("secreted",),
}

#: (id, organism_group, label passed to reference_demo.build, seed, declared categories)
SYNTHETIC_SPECS: list[tuple[str, str, str, int, tuple[str, ...]]] = [
    ("syn_ext_euk_1", "eukaryote", "EXT", 4101, ("secreted",)),
    ("syn_ext_euk_2", "eukaryote", "EXT", 4102, ("secreted",)),
    ("syn_mem_euk_1", "eukaryote", "MEM", 4111, ("membrane",)),
    ("syn_mem_euk_2", "eukaryote", "MEM", 4112, ("membrane",)),
    ("syn_cyt_euk_1", "eukaryote", "CYT", 4121, ("cytoplasmic",)),
    ("syn_cyt_euk_2", "eukaryote", "CYT", 4122, ("cytoplasmic",)),
    ("syn_mit_euk_1", "eukaryote", "MIT", 4131, ("mitochondrial",)),
    ("syn_mit_euk_2", "eukaryote", "MIT", 4132, ("mitochondrial",)),
    ("syn_nuc_euk_1", "eukaryote", "NUC", 4141, ("other_compartment",)),
    ("syn_per_euk_1", "eukaryote", "PER", 4142, ("other_compartment",)),
    ("syn_er_euk_1", "eukaryote", "ER", 4143, ("other_compartment",)),
    ("syn_pla_plant_1", "plant", "PLA", 4151, ("chloroplast",)),
    ("syn_pla_plant_2", "plant", "PLA", 4152, ("chloroplast",)),
    ("syn_pom_gramneg_1", "gram_neg", "POM", 4161, ("membrane",)),
    ("syn_pom_gramneg_2", "gram_neg", "POM", 4162, ("membrane",)),
    ("syn_ppl_gramneg_1", "gram_neg", "PPL", 4163, ("other_compartment",)),
    ("syn_pcm_gramneg_1", "gram_neg", "PCM", 4164, ("membrane",)),
    ("syn_cyt_gramneg_1", "gram_neg", "CYT", 4165, ("cytoplasmic",)),
    ("syn_pcw_grampos_1", "gram_pos", "PCW", 4171, ("other_compartment",)),
    ("syn_pcw_grampos_2", "gram_pos", "PCW", 4172, ("other_compartment",)),
    ("syn_cyt_grampos_1", "gram_pos", "CYT", 4173, ("cytoplasmic",)),
    ("syn_cyt_grampos_2", "gram_pos", "CYT", 4174, ("cytoplasmic",)),
    ("syn_ext_grampos_1", "gram_pos", "EXT", 4175, ("secreted",)),
]

_SYNTHETIC_NOTE = "Synthetic; planted signal in generated background. Not a real protein."


@dataclass(frozen=True)
class _Input:
    """One golden input as the file stores it."""

    id: str
    origin: str
    organism_group: str
    sequence: str
    declared_categories: tuple[str, ...]
    derivation: str
    parent: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origin": self.origin,
            "real_protein": False,
            "organism_group": self.organism_group,
            "sequence": self.sequence,
            "declared_categories": list(self.declared_categories),
            "derivation": self.derivation,
            "parent": self.parent,
            "source": None,
        }


def _entry(*fields: Any) -> dict[str, Any]:
    """Positional ``_Input`` fields in, stored dict out."""
    return _Input(*fields).as_dict()


def _replace(seq: str, old: str, new: str) -> str:
    """Replace ``old`` by ``new``; a no-op is an error, so no mutant is silently identical."""
    if old not in seq:
        msg = f"motif {old!r} not found; the edit would be a no-op"
        raise ValueError(msg)
    return seq.replace(old, new, 1)


def _gallery() -> list[dict[str, Any]]:
    return [
        _entry(
            e["id"],
            "gallery",
            e["organism_group"],
            e["sequence"],
            GALLERY_DECLARED[e["id"]],
            f"reference_demo.EXAMPLES verbatim. {e['note']}. {_SYNTHETIC_NOTE}",
        )
        for e in reference_demo.EXAMPLES
    ]


def _synthetic() -> list[dict[str, Any]]:
    out = []
    for entry_id, group, label, seed, cats in SYNTHETIC_SPECS:
        rng = random.Random(seed)  # noqa: S311 -- deterministic fixture, not security
        seq = reference_demo.build(label, rng)
        out.append(
            _entry(
                entry_id,
                "synthetic",
                group,
                seq,
                cats,
                f"reference_demo.build({label!r}, random.Random({seed})). {_SYNTHETIC_NOTE}",
            )
        )
    return out


def _engineered(by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    def seq(entry_id: str) -> str:
        return str(by_id[entry_id]["sequence"])

    def group(entry_id: str) -> str:
        return str(by_id[entry_id]["organism_group"])

    edge = ("edge_case",)
    out: list[dict[str, Any]] = []

    def add(entry_id: str, grp: str, sequence: str, derivation: str, parent: str | None) -> None:
        out.append(_entry(entry_id, "engineered", grp, sequence, edge, derivation, parent))

    def mutant(entry_id: str, parent: str, sequence: str, what: str) -> None:
        add(entry_id, group(parent), sequence, f"{what} of {parent}. {_SYNTHETIC_NOTE}", parent)

    sec = seq("secreted_enzyme")
    mutant(
        "eng_secreted_sp_hregion_to_D",
        "secreted_enzyme",
        sec[:3] + "D" * 13 + sec[16:],
        "residues 4-16 (the signal-peptide h-region) set to D",
    )
    mito = seq("mito_matrix")
    mutant(
        "eng_mito_head_basic_to_E",
        "mito_matrix",
        mito[:25].translate(str.maketrans("RK", "EE")) + mito[25:],
        "R and K in residues 1-25 (the presequence) set to E",
    )
    mutant(
        "eng_er_kdel_to_kdaa",
        "er_resident",
        _replace(seq("er_resident"), "KDEL", "KDAA"),
        "terminal KDEL replaced by KDAA",
    )
    mutant(
        "eng_nuc_nls_to_alanines",
        "nuclear_factor",
        _replace(seq("nuclear_factor"), "PKKKRKV", "AAAAAAA"),
        "PKKKRKV replaced by AAAAAAA",
    )
    mutant(
        "eng_per_skl_to_ska",
        "peroxisomal",
        _replace(seq("peroxisomal"), "SKL", "SKA"),
        "terminal SKL replaced by SKA",
    )
    mutant(
        "eng_sortase_lpetg_to_apetg",
        "sortase_anchored",
        _replace(seq("sortase_anchored"), "LPETG", "APETG"),
        "LPETG replaced by APETG",
    )
    mutant(
        "eng_chloroplast_ntrunc45",
        "chloroplast",
        "M" + seq("chloroplast")[46:],
        "residues 2-46 removed (N-terminal transit region)",
    )
    mutant(
        "eng_multipass_trunc60",
        "multi_pass",
        seq("multi_pass")[:60],
        "truncated to the first 60 residues",
    )
    mutant(
        "eng_porin_cterm_F_to_A",
        "outer_membrane",
        seq("outer_membrane")[:-1] + "A",
        "terminal F replaced by A",
    )
    conflict = _replace(_replace(seq("conflicted"), "PKKKRKV", "AAAAAAA"), "KRKRK", "AAAAA")
    mutant("eng_conflicted_nls_removed", "conflicted", conflict, "both NLS motifs replaced by A")

    mutant("eng_secreted_trunc30", "secreted_enzyme", sec[:30], "truncated to 30 residues")
    mutant("eng_secreted_trunc60", "secreted_enzyme", sec[:60], "truncated to 60 residues")
    mutant("eng_mito_trunc40", "mito_matrix", mito[:40], "truncated to 40 residues")

    for target in ("gram_neg", "gram_pos", "plant"):
        add(
            f"eng_secreted_as_{target}",
            target,
            sec,
            f"same sequence as secreted_enzyme, organism_group {target}",
            "secreted_enzyme",
        )
    add(
        "eng_mito_as_plant",
        "plant",
        mito,
        "same sequence as mito_matrix, organism_group plant",
        "mito_matrix",
    )
    add(
        "eng_porin_as_grampos",
        "gram_pos",
        seq("outer_membrane"),
        "same sequence as outer_membrane, organism_group gram_pos",
        "outer_membrane",
    )

    for res, name in (("A", "polyA"), ("L", "polyL"), ("K", "polyK"), ("E", "polyE")):
        add(
            f"eng_{name}_100",
            "eukaryote",
            res * 100,
            f"100 x {res}; a homopolymer, not a protein",
            None,
        )
    add(
        "eng_myristoyl_polyS_60",
        "eukaryote",
        "MG" + "S" * 58,
        "MG + 58 x S; myristoylation start with a serine-rich body",
        None,
    )
    add(
        "eng_all20_x5_100",
        "eukaryote",
        "ACDEFGHIKLMNPQRSTVWY" * 5,
        "the 20 amino acids in alphabetical one-letter order, repeated 5 times",
        None,
    )

    cyt = seq("syn_cyt_euk_1")
    add(
        "eng_len30_min",
        "eukaryote",
        cyt[:30],
        "first 30 residues of syn_cyt_euk_1 (lower length bound)",
        "syn_cyt_euk_1",
    )
    ext = seq("syn_ext_euk_1")
    padded = ext + reference_demo._bg(random.Random(4999), 3000 - len(ext))  # noqa: S311
    add(
        "eng_len3000_max",
        "eukaryote",
        padded,
        "syn_ext_euk_1 padded to 3000 residues with random.Random(4999) background (upper bound)",
        "syn_ext_euk_1",
    )
    letters = (
        ext[:70] + "X" + ext[71:100] + "U" + ext[101:130] + "B" + ext[131:160] + "Z" + ext[161:]
    )
    add(
        "eng_ambiguity_codes_XUBZ",
        "eukaryote",
        letters,
        "syn_ext_euk_1 with X, U, B, Z substituted at residues 71, 101, 131, 161",
        "syn_ext_euk_1",
    )
    wrapped = "\n".join(sec[i : i + 60].lower() + "  " for i in range(0, len(sec), 60))
    add(
        "eng_raw_lowercase_wrapped",
        "eukaryote",
        wrapped,
        "secreted_enzyme lower-cased, wrapped at 60 columns with trailing spaces",
        "secreted_enzyme",
    )
    add(
        "eng_retention_without_entry",
        "eukaryote",
        cyt + "KDEL",
        "syn_cyt_euk_1 + KDEL: an ER-retention motif on a sequence with no signal peptide",
        "syn_cyt_euk_1",
    )
    add(
        "eng_chimera_mtp_sp",
        "eukaryote",
        mito[:25] + sec[1:],
        "mito_matrix residues 1-25 fused to secreted_enzyme residues 2-end (dual N-term signal)",
        "mito_matrix",
    )

    add(
        "eng_reject_len29",
        "eukaryote",
        cyt[:29],
        "29 residues: one below the lower bound",
        "syn_cyt_euk_1",
    )
    add(
        "eng_reject_len3001",
        "eukaryote",
        padded + "A",
        "3001 residues: one above the upper bound",
        "eng_len3000_max",
    )
    add(
        "eng_reject_invalid_residue_J",
        "eukaryote",
        ext[:10] + "J" + ext[11:40],
        "40 residues with J at residue 11 (not an amino-acid letter)",
        "syn_ext_euk_1",
    )
    add(
        "eng_reject_unknown_group",
        "archaea",
        ext,
        "valid sequence, unsupported organism_group",
        "syn_ext_euk_1",
    )
    return out


def build_entries() -> list[dict[str, Any]]:
    """All golden-set inputs, in a fixed order."""
    gallery = _gallery()
    synthetic = _synthetic()
    by_id = {e["id"]: e for e in gallery + synthetic}
    engineered = _engineered(by_id)
    entries = gallery + synthetic + engineered
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        msg = "duplicate golden ids"
        raise ValueError(msg)
    return entries
