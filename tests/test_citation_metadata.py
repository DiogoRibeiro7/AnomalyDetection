"""Tests keeping Zenodo and CFF citation metadata valid and in sync."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ZENODO_PATH = ROOT / ".zenodo.json"
CITATION_PATH = ROOT / "CITATION.cff"
PYPROJECT_PATH = ROOT / "pyproject.toml"

CONCEPT_DOI = "10.5281/zenodo.21496904"

# Relation and resource type vocabularies accepted by the Zenodo deposit schema.
ZENODO_RELATIONS = {
    "isCitedBy",
    "cites",
    "isSupplementTo",
    "isSupplementedBy",
    "isContinuedBy",
    "continues",
    "isDescribedBy",
    "describes",
    "isDocumentedBy",
    "documents",
    "isPartOf",
    "hasPart",
    "isReferencedBy",
    "references",
    "isNewVersionOf",
    "isPreviousVersionOf",
    "isDerivedFrom",
    "isSourceOf",
    "isIdenticalTo",
    "isAlternateIdentifier",
    "requires",
    "isRequiredBy",
    "compiles",
    "isCompiledBy",
}


def _load_zenodo() -> dict[str, Any]:
    return json.loads(ZENODO_PATH.read_text(encoding="utf-8"))


def _load_citation() -> dict[str, Any]:
    return yaml.safe_load(CITATION_PATH.read_text(encoding="utf-8"))


def _project_version() -> str:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_metadata_files_parse() -> None:
    assert isinstance(_load_zenodo(), dict)
    assert isinstance(_load_citation(), dict)


def test_versions_are_synchronised() -> None:
    version = _project_version()
    assert _load_zenodo()["version"] == version
    assert str(_load_citation()["version"]) == version


def test_zenodo_required_fields_present() -> None:
    zenodo = _load_zenodo()
    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"
    assert zenodo["license"] == "mit"
    assert zenodo["title"].strip()
    assert len(zenodo["description"]) > 200
    assert len(zenodo["keywords"]) >= 5
    assert len(set(zenodo["keywords"])) == len(zenodo["keywords"])


def test_zenodo_creators_are_complete() -> None:
    creators = _load_zenodo()["creators"]
    assert creators
    for creator in creators:
        assert "," in creator["name"], "Zenodo expects 'Family, Given' names."
        # Bare identifier form, without the https://orcid.org/ prefix.
        assert not creator["orcid"].startswith("http")
        assert creator["affiliation"].strip()


def test_zenodo_related_identifiers_use_known_relations() -> None:
    for related in _load_zenodo().get("related_identifiers", []):
        assert related["relation"] in ZENODO_RELATIONS
        assert related["identifier"].startswith("https://")


def test_citation_and_zenodo_agree_on_shared_fields() -> None:
    zenodo = _load_zenodo()
    citation = _load_citation()
    assert citation["title"] == zenodo["title"]
    assert citation["license"].lower() == zenodo["license"]
    assert citation["doi"] == CONCEPT_DOI
    assert set(citation["keywords"]) <= set(zenodo["keywords"])


def test_citation_author_matches_zenodo_creator() -> None:
    creator = _load_zenodo()["creators"][0]
    author = _load_citation()["authors"][0]
    family, given = (part.strip() for part in creator["name"].split(",", 1))
    assert author["family-names"] == family
    assert author["given-names"] == given
    assert author["orcid"] == f"https://orcid.org/{creator['orcid']}"
    assert author["affiliation"] == creator["affiliation"]


@pytest.mark.parametrize("doi_holder", ["zenodo", "citation"])
def test_concept_doi_is_advertised(doi_holder: str) -> None:
    if doi_holder == "zenodo":
        assert CONCEPT_DOI in _load_zenodo()["description"]
    else:
        values = {entry["value"] for entry in _load_citation()["identifiers"]}
        assert CONCEPT_DOI in values
