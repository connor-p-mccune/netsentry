"""Provenance tests: file hashing, requirement parsing, SBOM structure/validity,
and the integrity gate (verify passes on a good bundle, fails on a tampered one)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from netsentry.config import Settings
from netsentry.governance.provenance import (
    _normalize_pypi_name,
    _parse_requirement_name,
    build_sbom,
    declared_dependencies,
    sha256_file,
    verify_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    f = tmp_path / "blob.bin"
    f.write_bytes(b"netsentry" * 1000)
    assert sha256_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("scikit-learn>=1.4", "scikit-learn"),
        ("uvicorn[standard]>=0.29", "uvicorn"),
        ("torch>=2.2", "torch"),
        ("netsentry[serve]", None),  # self-reference is skipped
    ],
)
def test_parse_requirement_name(spec: str, expected: str | None) -> None:
    assert _parse_requirement_name(spec) == expected


def test_normalize_pypi_name() -> None:
    assert _normalize_pypi_name("scikit_learn") == "scikit-learn"
    assert _normalize_pypi_name("PyYAML") == "pyyaml"


def test_declared_dependencies_span_core_and_extras() -> None:
    deps = declared_dependencies(PYPROJECT)
    assert "core" in deps["numpy"]  # a core dependency
    assert deps["lightgbm"] == {"train"}  # an optional-extra dependency
    assert "netsentry" not in deps  # self-references dropped


def test_build_sbom_is_cyclonedx_shaped(settings: Settings) -> None:
    sbom = build_sbom(settings, PYPROJECT)
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.5"
    assert sbom["serialNumber"].startswith("urn:uuid:")
    assert sbom["metadata"]["component"]["name"] == "netsentry"
    # numpy is a core dep and always installed in the test env -> a versioned component.
    numpy = next(c for c in sbom["components"] if c["name"] == "numpy")
    assert numpy["type"] == "library"
    assert numpy["purl"] == f"pkg:pypi/numpy@{numpy['version']}"
    assert {"name": "netsentry:dependency-group", "value": "core"} in numpy["properties"]


def test_build_sbom_round_trips_as_json(settings: Settings) -> None:
    sbom = build_sbom(settings, PYPROJECT)
    assert json.loads(json.dumps(sbom)) == sbom  # fully JSON-serialisable


def _write_manifest(path: Path, bundle: Path, sha: str) -> None:
    path.write_text(json.dumps({"bundle": {"name": bundle.name, "sha256": sha}}), encoding="utf-8")


def test_verify_passes_on_matching_hash(tmp_path: Path) -> None:
    bundle = tmp_path / "b.joblib"
    bundle.write_bytes(b"model-weights")
    manifest = tmp_path / "m.json"
    _write_manifest(manifest, bundle, sha256_file(bundle))

    result = verify_manifest(manifest)
    assert result.ok
    assert all(passed for _, passed, _ in result.checks)


def test_verify_fails_on_tampered_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "b.joblib"
    bundle.write_bytes(b"model-weights")
    manifest = tmp_path / "m.json"
    _write_manifest(manifest, bundle, sha256_file(bundle))
    bundle.write_bytes(b"model-weights-TAMPERED")  # swap the artifact after signing

    result = verify_manifest(manifest)
    assert not result.ok
    assert any(name == "bundle-sha256" and not passed for name, passed, _ in result.checks)


def test_verify_fails_when_bundle_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    _write_manifest(manifest, tmp_path / "gone.joblib", "deadbeef")
    result = verify_manifest(manifest)
    assert not result.ok
    assert result.checks[0][0] == "bundle-present"
