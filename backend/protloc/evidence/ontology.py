"""Gene Ontology names for the terms evidence transfers (Phase 14).

Layer L5 (evidence), standard library only. The Phase 13 hit-term table lists GO ids; a person
reading a result needs the term's name and aspect, which live in the ontology release the bundle
pins by SHA-256. This module reads names and namespaces out of that OBO file and the hit-term
table that maps database accessions to their propagated terms. It reads no model output.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

_TERM_STANZA = "[Term]"
_ASPECTS = frozenset({"molecular_function", "biological_process", "cellular_component"})


@dataclass(frozen=True, slots=True)
class OboTerm:
    """One non-obsolete GO term."""

    go_id: str
    name: str
    aspect: str


def parse_obo(text: str) -> dict[str, OboTerm]:
    """Names and aspects of every non-obsolete term in a ``go-basic.obo`` document."""
    terms: dict[str, OboTerm] = {}
    stanza: dict[str, str] = {}
    obsolete = False

    def close() -> None:
        go_id, name, aspect = stanza.get("id"), stanza.get("name"), stanza.get("namespace")
        if go_id and name and aspect in _ASPECTS and not obsolete:
            terms[go_id] = OboTerm(go_id, name, aspect)

    inside = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("["):
            if inside:
                close()
            inside, stanza, obsolete = line == _TERM_STANZA, {}, False
        elif inside and line:
            key, _colon, value = line.partition(": ")
            if key == "is_obsolete" and value == "true":
                obsolete = True
            elif key in ("id", "name", "namespace") and key not in stanza:
                stanza[key] = value
    if inside:
        close()
    return terms


def read_obo(path: Path) -> dict[str, OboTerm]:
    """:func:`parse_obo` of a file."""
    return parse_obo(path.read_text(encoding="utf-8"))


class TermStore(Mapping[str, frozenset[str]]):
    """The accession -> propagated GO terms table, read lazily.

    ``scripts/homology.py`` writes one ``accession<TAB>term;term;...`` line per Swiss-Prot entry
    (548,702 of them). Holding every term set in memory would cost hundreds of megabytes the
    service cannot spare, so only the byte offset of each line is kept and a line is parsed when
    a hit asks for it.
    """

    def __init__(self, path: Path) -> None:
        """Index the table; the file must stay unchanged while the store is open."""
        self._path = path
        self._offsets: dict[str, int] = {}
        offset = 0
        with path.open("rb") as handle:
            for line in handle:
                accession = line.split(b"\t", 1)[0].decode("ascii")
                self._offsets[accession] = offset
                offset += len(line)

    def __getitem__(self, accession: str) -> frozenset[str]:
        """The terms of one entry; ``KeyError`` for an accession the table does not list."""
        offset = self._offsets[accession]
        with self._path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline().decode("ascii").rstrip("\r\n")
        _accession, _tab, cell = line.partition("\t")
        return frozenset(t for t in cell.split(";") if t)

    def __iter__(self) -> Iterator[str]:
        """Every accession in the table."""
        return iter(self._offsets)

    def __len__(self) -> int:
        """Number of entries."""
        return len(self._offsets)
