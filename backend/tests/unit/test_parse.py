"""Flat-file entries become the labels ADR-0006 says they should.

Every fixture below is hand-written in Swiss-Prot flat-file syntax, and each one
exercises one rule of ADR-0006 §3.3. The rule files are the real ones in
``backend/rules/``, so these tests also fail when someone edits a rule file in a
way that changes a label.
"""

from io import StringIO
from pathlib import Path

from Bio import SwissProt
import pytest

from protloc.data.parse import EntryError, EntryOutcome, parse_entry
from protloc.data.records import SequenceRecord
from protloc.data.rules import CorpusRules, load_corpus_rules
from protloc.domain.types import LabelState, OrganismGroup, TaskName

RULES_DIR = Path(__file__).resolve().parents[2] / "rules"
RELEASE = "2099_01"

HUMAN_LINEAGE = (
    "OC   Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; Euteleostomi;\n"
    "OC   Mammalia; Eutheria; Euarchontoglires; Primates; Haplorrhini;\n"
    "OC   Catarrhini; Hominidae; Homo.\n"
)
ECOLI_LINEAGE = (
    "OC   Bacteria; Pseudomonadati; Pseudomonadota; Gammaproteobacteria;\n"
    "OC   Enterobacterales; Enterobacteriaceae; Escherichia.\n"
)
BACILLUS_LINEAGE = (
    "OC   Bacteria; Bacillati; Bacillota; Bacilli; Bacillales; Bacillaceae;\nOC   Bacillus.\n"
)
VEILLONELLA_LINEAGE = (
    "OC   Bacteria; Bacillati; Bacillota; Negativicutes; Veillonellales;\n"
    "OC   Veillonellaceae; Veillonella.\n"
)
MYCOPLASMA_LINEAGE = (
    "OC   Bacteria; Bacillati; Mycoplasmatota; Mollicutes; Mycoplasmataceae;\nOC   Mycoplasma.\n"
)
VIRUS_LINEAGE = "OC   Viruses; Riboviria; Orthornavirae.\n"


def _entry(  # noqa: PLR0913 - one keyword per flat-file line type
    *,
    sequence: str = "MKKLLPTAAAGLLLLAAQPAMAHHHHHHWWWW",
    lineage: str = HUMAN_LINEAGE,
    taxid: str = "9606",
    flags: str = "",
    comments: str = "",
    dr: str = "",
    ft: str = "",
) -> SwissProt.Record:
    blocks = ["     " + sequence[i : i + 60] for i in range(0, len(sequence), 60)]
    text = (
        f"ID   TEST1_HUMAN             Reviewed;         {len(sequence)} AA.\n"
        "AC   P99999; Q00001;\n"
        "DT   01-JAN-2000, integrated into UniProtKB/Swiss-Prot.\n"
        "DT   01-JAN-2000, sequence version 1.\n"
        "DT   01-JAN-2020, entry version 10.\n"
        "DE   RecName: Full=Test protein;\n"
        + (f"DE   Flags: {flags};\n" if flags else "")
        + "OS   Test organism.\n"
        + lineage
        + f"OX   NCBI_TaxID={taxid};\n"
        + comments
        + dr
        + ft
        + f"SQ   SEQUENCE   {len(sequence)} AA;  3000 MW;  0000000000000000 CRC64;\n"
        + "\n".join(blocks)
        + "\n//\n"
    )
    return next(SwissProt.parse(StringIO(text)))


@pytest.fixture(scope="module")
def rules() -> CorpusRules:
    return load_corpus_rules(RULES_DIR)


def _record(outcome: EntryOutcome) -> SequenceRecord:
    assert outcome.record is not None, outcome.excluded
    return outcome.record


def _state(record: SequenceRecord, task: TaskName) -> LabelState:
    return record.label(task).state


SIGNAL_FT = 'FT   SIGNAL          1..22\nFT                   /evidence="ECO:0000255"\n'
SL_MITO = (
    "CC   -!- SUBCELLULAR LOCATION: Mitochondrion inner membrane\n"
    "CC       {ECO:0000269|PubMed:123}; Single-pass type I membrane protein\n"
    "CC       {ECO:0000255}. Cytoplasm. Note=Also elsewhere. {ECO:0000305}.\n"
)
SL_NUCLEUS = "CC   -!- SUBCELLULAR LOCATION: Nucleus {ECO:0000269|PubMed:9}.\n"
SL_ISOFORM_ONLY = "CC   -!- SUBCELLULAR LOCATION: [Isoform 2]: Nucleus.\n"


def test_signal_peptide_is_annotated_with_its_cleavage_site(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(ft=SIGNAL_FT), RELEASE, rules))
    label = record.label(TaskName.SIGNAL_PEPTIDE)
    assert label.state is LabelState.ANNOTATED
    assert label.values == ("signal_peptide",)
    assert label.cleavage_site == 22
    assert label.evidence == (("signal_peptide", ("ECO:0000255",)),)


def test_no_signal_feature_on_a_full_length_entry_is_negative(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(), RELEASE, rules))
    assert _state(record, TaskName.SIGNAL_PEPTIDE) is LabelState.NEGATIVE


def test_no_signal_feature_on_a_fragment_is_unknown(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(flags="Fragment"), RELEASE, rules))
    assert record.is_fragment
    assert _state(record, TaskName.SIGNAL_PEPTIDE) is LabelState.UNKNOWN
    assert _state(record, TaskName.MITOCHONDRIAL_TRANSIT_PEPTIDE) is LabelState.UNKNOWN


def test_non_ter_feature_marks_a_fragment(rules: CorpusRules) -> None:
    ft = "FT   NON_TER         1\n"
    assert _record(parse_entry(_entry(ft=ft), RELEASE, rules)).is_fragment


def test_fuzzy_signal_peptide_is_unknown(rules: CorpusRules) -> None:
    ft = "FT   SIGNAL          1..?22\n"
    record = _record(parse_entry(_entry(ft=ft), RELEASE, rules))
    assert _state(record, TaskName.SIGNAL_PEPTIDE) is LabelState.UNKNOWN


def test_location_terms_map_by_longest_prefix(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(comments=SL_MITO), RELEASE, rules))
    label = record.label(TaskName.LOCALIZATION)
    assert label.state is LabelState.ANNOTATED
    assert label.values == ("cytoplasm", "mitochondrion")
    assert label.evidence == (("mitochondrion", ("ECO:0000269",)),)


def test_note_text_does_not_become_a_location(rules: CorpusRules) -> None:
    outcome = parse_entry(_entry(comments=SL_MITO), RELEASE, rules)
    assert "Also elsewhere" not in outcome.diagnostics.unmapped_location_terms


def test_topology_gives_membrane_type(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(comments=SL_MITO), RELEASE, rules))
    label = record.label(TaskName.MEMBRANE_TYPE)
    assert label.values == ("single_pass_type_i",)
    assert label.evidence == (("single_pass_type_i", ("ECO:0000255",)),)


def test_soluble_curated_location_is_a_membrane_negative(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(comments=SL_NUCLEUS), RELEASE, rules))
    assert _state(record, TaskName.MEMBRANE_TYPE) is LabelState.NEGATIVE


def test_transmembrane_feature_forbids_a_membrane_negative(rules: CorpusRules) -> None:
    ft = "FT   TRANSMEM        5..25\n"
    record = _record(parse_entry(_entry(comments=SL_NUCLEUS, ft=ft), RELEASE, rules))
    assert _state(record, TaskName.MEMBRANE_TYPE) is LabelState.UNKNOWN


def test_no_location_means_unknown_never_negative(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(), RELEASE, rules))
    assert _state(record, TaskName.LOCALIZATION) is LabelState.UNKNOWN
    assert _state(record, TaskName.MEMBRANE_TYPE) is LabelState.UNKNOWN
    assert _state(record, TaskName.GO_FUNCTION) is LabelState.UNKNOWN


def test_isoform_only_location_is_unknown(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(comments=SL_ISOFORM_ONLY), RELEASE, rules))
    assert _state(record, TaskName.LOCALIZATION) is LabelState.UNKNOWN


def test_unmapped_location_terms_are_reported(rules: CorpusRules) -> None:
    comments = "CC   -!- SUBCELLULAR LOCATION: Cytoplasmic vesicle.\n"
    outcome = parse_entry(_entry(comments=comments), RELEASE, rules)
    assert outcome.diagnostics.unmapped_location_terms == ["Cytoplasmic vesicle"]
    assert _state(_record(outcome), TaskName.LOCALIZATION) is LabelState.UNKNOWN


def test_mitochondrial_transit_peptide(rules: CorpusRules) -> None:
    ft = (
        "FT   TRANSIT         1..25\n"
        'FT                   /note="Mitochondrion"\n'
        'FT                   /evidence="ECO:0000269|PubMed:1, ECO:0000305"\n'
    )
    record = _record(parse_entry(_entry(ft=ft), RELEASE, rules))
    mtp = record.label(TaskName.MITOCHONDRIAL_TRANSIT_PEPTIDE)
    assert mtp.state is LabelState.ANNOTATED
    assert mtp.cleavage_site == 25
    assert mtp.evidence == (("mitochondrial_transit_peptide", ("ECO:0000269", "ECO:0000305")),)
    # A transit peptide exists, just not a chloroplast one: unknown, not negative.
    assert _state(record, TaskName.CHLOROPLAST_TRANSIT_PEPTIDE) is LabelState.UNKNOWN


def test_no_transit_feature_is_negative_for_both(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(), RELEASE, rules))
    assert _state(record, TaskName.MITOCHONDRIAL_TRANSIT_PEPTIDE) is LabelState.NEGATIVE
    assert _state(record, TaskName.CHLOROPLAST_TRANSIT_PEPTIDE) is LabelState.NEGATIVE


def test_go_function_keeps_go_evidence_codes(rules: CorpusRules) -> None:
    dr = (
        "DR   GO; GO:0005737; C:cytoplasm; IBA:GO_Central.\n"
        "DR   GO; GO:0005524; F:ATP binding; IEA:UniProtKB-KW.\n"
        "DR   GO; GO:0004672; F:protein kinase activity; IDA:UniProtKB.\n"
    )
    record = _record(parse_entry(_entry(dr=dr), RELEASE, rules))
    label = record.label(TaskName.GO_FUNCTION)
    assert label.values == ("GO:0004672", "GO:0005524")
    assert label.evidence == (("GO:0004672", ("IDA",)), ("GO:0005524", ("IEA",)))


def test_eukaryote_masks_the_prokaryotic_task(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(comments=SL_NUCLEUS), RELEASE, rules))
    assert record.organism_group is OrganismGroup.EUKARYA
    assert _state(record, TaskName.PROKARYOTIC_LOCALIZATION) is LabelState.NOT_APPLICABLE


def test_prokaryote_masks_the_eukaryotic_tasks(rules: CorpusRules) -> None:
    comments = "CC   -!- SUBCELLULAR LOCATION: Periplasm {ECO:0000269}.\n"
    entry = _entry(lineage=ECOLI_LINEAGE, taxid="83333", comments=comments)
    record = _record(parse_entry(entry, RELEASE, rules))
    assert record.organism_group is OrganismGroup.GRAM_NEGATIVE
    assert record.label(TaskName.PROKARYOTIC_LOCALIZATION).values == ("periplasmic",)
    for task in (
        TaskName.LOCALIZATION,
        TaskName.MITOCHONDRIAL_TRANSIT_PEPTIDE,
        TaskName.CHLOROPLAST_TRANSIT_PEPTIDE,
    ):
        assert _state(record, task) is LabelState.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("lineage", "expected"),
    [
        (BACILLUS_LINEAGE, OrganismGroup.GRAM_POSITIVE),
        (VEILLONELLA_LINEAGE, OrganismGroup.GRAM_NEGATIVE),
        (ECOLI_LINEAGE, OrganismGroup.GRAM_NEGATIVE),
    ],
)
def test_lineage_rules_are_first_match(
    rules: CorpusRules, lineage: str, expected: OrganismGroup
) -> None:
    record = _record(parse_entry(_entry(lineage=lineage, taxid="1"), RELEASE, rules))
    assert record.organism_group is expected


@pytest.mark.parametrize(
    ("lineage", "reason"),
    [
        (VIRUS_LINEAGE, "organism_group:virus"),
        (MYCOPLASMA_LINEAGE, "organism_group:wall_less_bacterium"),
        ("OC   Bacteria; Candidatus Nonexistota.\n", "organism_group:unmapped_bacterial_phylum"),
    ],
)
def test_unassignable_organisms_are_excluded_with_a_reason(
    rules: CorpusRules, lineage: str, reason: str
) -> None:
    outcome = parse_entry(_entry(lineage=lineage, taxid="1"), RELEASE, rules)
    assert outcome.record is None
    assert outcome.excluded == reason


def test_nonstandard_codes_are_kept_verbatim(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(sequence="MUOXBZAAAA"), RELEASE, rules))
    assert record.sequence_normalized == "MUOXBZAAAA"


def test_j_excludes_the_record(rules: CorpusRules) -> None:
    outcome = parse_entry(_entry(sequence="MJAAAA"), RELEASE, rules)
    assert outcome.excluded == "residue:J"


def test_a_code_outside_the_policy_stops_the_build(rules: CorpusRules) -> None:
    with pytest.raises(EntryError, match="outside the residue policy"):
        parse_entry(_entry(sequence="MAAAA1"), RELEASE, rules)


def test_every_record_has_all_seven_labels(rules: CorpusRules) -> None:
    record = _record(parse_entry(_entry(), RELEASE, rules))
    assert [task for task, _ in record.labels] == list(TaskName)
    assert record.source_release == RELEASE
    assert record.taxonomy_id == 9606
    assert record.organism == "Test organism"
