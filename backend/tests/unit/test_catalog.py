"""The cached-sequence catalog (browse all 500, look one up by accession).

The module is standard library only, so the reader is tested on small hand-written files; the
two routes are tested against a stub engine, as in ``test_api_errors.py``.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from protloc.inference import catalog
from protloc.inference.catalog import MAX_PAGE, load_catalog, load_for

CACHE = "abc_subset500_residues"


def _entry(accession: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": accession,
        "entry_name": f"{accession}_HUMAN",
        "name": f"{accession}_HUMAN ({accession})",
        "organism": "Homo sapiens (Human)",
        "organism_group": "eukaryote",
        "sequence": "MKT" + accession,
        "sequence_sha256": "0" * 64,
        "length": 3 + len(accession),
        "partition": 1,
        "in_training_set": True,
        "database_labels": {
            "localization": ["nucleus"],
            "signal_peptide": [],
            "signal_peptide_cleavage_site": None,
        },
        "note": "test",
    }
    base.update(over)
    return base


def _write(path: Path, rows: list[Any], **over: Any) -> Path:
    document = {"schema_version": 1, "residue_cache": CACHE, "entries": rows, **over}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _all(_sequence: str) -> bool:
    return True


@pytest.fixture
def three(tmp_path: Path) -> catalog.Catalog:
    entries = [
        _entry("P11111"),
        _entry(
            "Q22222",
            organism="Arabidopsis thaliana",
            organism_group="plant",
            partition=5,
            in_training_set=False,
            database_labels={
                "localization": ["plastid", "cytoplasm"],
                "signal_peptide": [],
                "signal_peptide_cleavage_site": None,
            },
        ),
        _entry("O33333", partition=3),
    ]
    return load_catalog(_write(tmp_path / "c.json", entries), CACHE, _all)


def test_a_good_file_loads_every_scorable_entry(three: catalog.Catalog) -> None:
    assert three.available
    assert three.state() == {
        "available": True,
        "reason": None,
        "entries": 3,
        "dropped_not_cached": 0,
    }


def test_an_entry_the_engine_cannot_score_is_dropped_and_counted(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.json", [_entry("P11111"), _entry("Q22222")])
    loaded = load_catalog(path, CACHE, lambda sequence: sequence.endswith("P11111"))
    assert [e["id"] for e in loaded.entries] == ["P11111"]
    assert loaded.dropped == 1


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"schema_version": 2}, "schema"),
        ({"residue_cache": "other"}, "different residue cache"),
        ({"entries": "nope"}, "no entry list"),
    ],
)
def test_a_wrong_file_is_unavailable_with_a_reason(
    tmp_path: Path, over: dict[str, Any], reason: str
) -> None:
    loaded = load_catalog(_write(tmp_path / "c.json", [], **over), CACHE, _all)
    assert not loaded.available
    assert reason in str(loaded.reason)
    assert loaded.search()["total"] == 0


def test_a_missing_or_broken_file_never_raises(tmp_path: Path) -> None:
    assert "build_catalog" in str(load_catalog(tmp_path / "absent.json", CACHE, _all).reason)
    broken = tmp_path / "b.json"
    broken.write_text("{not json", encoding="utf-8")
    assert "unreadable" in str(load_catalog(broken, CACHE, _all).reason)


def test_a_malformed_entry_is_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.json", [_entry("P11111"), {"id": "X"}, "junk"])
    loaded = load_catalog(path, CACHE, _all)
    assert [e["id"] for e in loaded.entries] == ["P11111"]
    assert loaded.dropped == 2


def test_lookup_is_by_accession_or_entry_name_any_case(three: catalog.Catalog) -> None:
    for key in ("Q22222", "q22222", " Q22222 ", "q22222_human"):
        found = three.lookup(key)
        assert found is not None
        assert found["id"] == "Q22222"
        assert found["sequence"] == "MKTQ22222"
    assert three.lookup("P99999") is None


def test_search_matches_accession_entry_name_and_organism(three: catalog.Catalog) -> None:
    assert [e["id"] for e in three.search("arabidopsis")["entries"]] == ["Q22222"]
    assert [e["id"] for e in three.search("o333")["entries"]] == ["O33333"]
    assert three.search("")["total"] == 3


def test_filters_combine(three: catalog.Catalog) -> None:
    assert three.search(location="plastid")["total"] == 1
    assert three.search(group="eukaryote")["total"] == 2
    assert three.search(partition=5)["total"] == 1
    assert three.search(location="nucleus", partition=3)["total"] == 1
    assert three.search(location="nucleus", group="plant")["total"] == 0


def test_a_listing_carries_no_sequence_and_pages(three: catalog.Catalog) -> None:
    page = three.search(limit=2, offset=1)
    assert (page["total"], page["offset"], page["limit"]) == (3, 1, 2)
    assert len(page["entries"]) == 2
    assert all("sequence" not in e for e in page["entries"])
    assert three.search(limit=10_000)["limit"] == MAX_PAGE


def test_locations_lists_each_once(three: catalog.Catalog) -> None:
    assert three.locations() == ["cytoplasm", "nucleus", "plastid"]


def test_load_for_reads_the_cache_name_from_the_engine_config(tmp_path: Path) -> None:
    config = tmp_path / "engine.json"
    config.write_text(json.dumps({"encoder": {"residue_cache": CACHE}}), encoding="utf-8")
    (tmp_path / "gallery").mkdir()
    _write(tmp_path / "gallery" / "catalog.json", [_entry("P11111")])
    assert load_for(tmp_path, config, _all).state()["entries"] == 1
    config.write_text("{}", encoding="utf-8")
    assert "no residue cache" in str(load_for(tmp_path, config, _all).reason)


# -- the two routes --------------------------------------------------------------------


class _Stub:
    groups: ClassVar[tuple[str, ...]] = ("eukaryote", "plant")
    cached_sequences = 500
    provenance: ClassVar[dict[str, str]] = {
        "model_version": "test",
        "config_hash": "c",
        "rules_version": "r",
    }


@pytest.fixture
def client(three: catalog.Catalog) -> Iterator[Any]:
    pytest.importorskip("numpy")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from protloc.api import app_v2  # noqa: PLC0415
    from protloc.inference import service  # noqa: PLC0415

    service.configure(_Stub())  # type: ignore[arg-type]
    catalog.configure(three)
    yield TestClient(app_v2.app)
    service.configure(None)
    catalog.configure(catalog.unavailable("test over"))


def test_the_listing_route_filters_and_reports_itself(client: Any) -> None:
    body = client.get("/demo/catalog", params={"location": "plastid"}).json()
    assert body["total"] == 1
    assert body["entries"][0]["id"] == "Q22222"
    assert body["catalog"]["available"] is True
    assert body["locations"] == ["cytoplasm", "nucleus", "plastid"]
    assert body["provenance"]["model_version"] == "test"


def test_the_listing_route_validates_its_parameters(client: Any) -> None:
    assert client.get("/demo/catalog", params={"partition": 9}).status_code == 422
    assert client.get("/demo/catalog", params={"limit": 0}).status_code == 422
    r = client.get("/demo/catalog", params={"group": "martian"})
    assert (r.status_code, r.json()["error"]["code"]) == (400, "unknown_group")


def test_the_lookup_route_returns_the_sequence(client: Any) -> None:
    body = client.get("/demo/catalog/q22222").json()
    assert body["id"] == "Q22222"
    assert body["sequence"] == "MKTQ22222"
    assert body["organism_group"] == "plant"


def test_an_unknown_accession_is_a_stable_404(client: Any) -> None:
    r = client.get("/demo/catalog/P99999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "accession_not_cached"
    assert "500 cached" in r.json()["detail"]


def test_an_unloaded_catalog_is_a_stable_503(client: Any) -> None:
    catalog.configure(catalog.unavailable("no file"))
    r = client.get("/demo/catalog/P11111")
    assert (r.status_code, r.json()["error"]["code"]) == (503, "catalog_unavailable")
    assert client.get("/demo/catalog").json()["catalog"]["reason"] == "no file"
