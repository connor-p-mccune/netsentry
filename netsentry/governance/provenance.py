"""Supply-chain SBOM + model-integrity provenance — "can you trust this artifact?".

A trained model is a binary blob that decides whether traffic is malicious; a
reviewer should be able to answer *what went into it* and *has it been tampered
with* without rerunning training. Two standard governance artifacts do that:

- A **CycloneDX SBOM** of the project's declared dependencies resolved to the
  versions installed in the build environment — the software bill of materials a
  supply-chain audit (or a CVE sweep against the dependency graph) starts from.
- A **model manifest**: the SHA-256 of the deployed bundle, a digest of the
  resolved training config, the git commit, the runtime, and a summary of the
  bundle's own contents. ``verify`` recomputes the hashes and fails loudly on a
  mismatch — the integrity gate you run at deploy or in CI before promoting a model.

The SBOM is hand-built to the CycloneDX 1.5 schema (rather than through a library
whose API churns) so it stays a stable, dependency-free, spec-valid artifact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from netsentry import __version__
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

SBOM_NAME = "sbom.json"
MANIFEST_NAME = "model_manifest.json"
REPORT_NAME = "provenance.md"
_SPEC_VERSION = "1.5"
_HASH_CHUNK = 1 << 20  # 1 MiB streaming read, so hashing a large bundle is bounded


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file (bounded memory for large bundles)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    """SHA-256 of a canonical (sorted-key) JSON encoding — a stable content digest."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    """Current git commit (best-effort; None outside a repo or without git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _normalize_pypi_name(name: str) -> str:
    """PEP 503 normalized project name (for a stable Package URL)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement_name(spec: str) -> str | None:
    """Extract the bare distribution name from a requirement spec, or None to skip."""
    match = re.match(r"^([A-Za-z0-9._-]+)", spec.strip())
    if match is None:
        return None
    name = match.group(1)
    return None if name.lower() == "netsentry" else name  # skip self-references


def declared_dependencies(pyproject_path: Path) -> dict[str, set[str]]:
    """Map each declared dependency name -> the set of extras/groups that pull it."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    groups: dict[str, list[str]] = {"core": project.get("dependencies", [])}
    for group, specs in project.get("optional-dependencies", {}).items():
        groups[group] = specs

    deps: dict[str, set[str]] = {}
    for group, specs in groups.items():
        for spec in specs:
            name = _parse_requirement_name(spec)
            if name is not None:
                deps.setdefault(name, set()).add(group)
    return deps


def build_sbom(settings: Settings, pyproject_path: Path) -> dict[str, Any]:
    """CycloneDX 1.5 SBOM of declared deps resolved to installed versions.

    Scope is deliberately the *declared* dependency set (not the full transitive
    closure of the environment): a bounded, meaningful, reproducible bill of
    materials. Declared-but-not-installed optional extras are recorded in the
    report, but only actually-present, versioned components enter ``components``.
    """
    deps = declared_dependencies(pyproject_path)
    components: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in sorted(deps):
        try:
            resolved = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        purl_name = _normalize_pypi_name(name)
        components.append(
            {
                "type": "library",
                "name": name,
                "version": resolved,
                "purl": f"pkg:pypi/{purl_name}@{resolved}",
                "properties": [
                    {"name": "netsentry:dependency-group", "value": g} for g in sorted(deps[name])
                ],
            }
        )

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": _SPEC_VERSION,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tools": [
                {"vendor": "NetSentry", "name": "netsentry-provenance", "version": __version__}
            ],
            "component": {
                "type": "application",
                "name": "netsentry",
                "version": __version__,
                "purl": f"pkg:pypi/netsentry@{__version__}",
            },
            "properties": [
                {"name": "netsentry:sbom-scope", "value": "declared-dependencies"},
                {"name": "netsentry:declared-not-installed", "value": ", ".join(missing) or "none"},
            ],
        },
        "components": components,
    }
    return sbom


def _bundle_summary(bundle: Any) -> dict[str, Any]:
    """A tamper-evident summary of what the bundle carries (not the weights)."""
    meta = bundle.metadata
    return {
        "version": meta.get("version"),
        "task": meta.get("task"),
        "split_strategy": meta.get("split_strategy"),
        "backend": meta.get("backend"),
        "classes": meta.get("classes"),
        "n_features": meta.get("n_features"),
        "calibration": meta.get("calibration"),
        "threshold_profiles": sorted(bundle.thresholds),
        "has_calibrator": bundle.calibrator is not None,
        "has_anomaly_detector": bundle.anomaly_detector is not None,
        "has_conformal": "conformal" in meta,
        "has_drift_reference": "drift_reference" in meta,
    }


def build_manifest(
    settings: Settings, bundle_path: Path, *, sbom: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Integrity + provenance manifest for one deployed bundle."""
    from netsentry.models.registry import load_bundle

    bundle = load_bundle(bundle_path)
    config_json = settings.model_dump(mode="json")
    manifest = {
        "schema": "netsentry/model-manifest@1",
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "bundle": {
            "name": bundle_path.name,
            "sha256": sha256_file(bundle_path),
            "size_bytes": bundle_path.stat().st_size,
        },
        "model": _bundle_summary(bundle),
        "config": {
            "seed": settings.seed,
            "digest_sha256": _sha256_json(config_json),
        },
    }
    if sbom is not None:
        manifest["sbom"] = {
            "component_count": len(sbom["components"]),
            "digest_sha256": _sha256_json(sbom),
        }
    return manifest


@dataclass
class VerifyResult:
    """Outcome of verifying a bundle against its manifest."""

    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append((name, passed, detail))
        self.ok = self.ok and passed


def verify_manifest(manifest_path: Path, bundle_path: Path | None = None) -> VerifyResult:
    """Recompute the bundle hash and compare it to the manifest — the integrity gate."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = VerifyResult(ok=True)

    recorded = manifest.get("bundle", {})
    resolved_bundle = bundle_path or (manifest_path.parent / str(recorded.get("name", "")))
    if not resolved_bundle.exists():
        result.add("bundle-present", False, f"bundle not found at {resolved_bundle}")
        return result
    result.add("bundle-present", True, str(resolved_bundle))

    actual = sha256_file(resolved_bundle)
    expected = str(recorded.get("sha256", ""))
    match = actual == expected
    detail = "hash matches manifest" if match else f"expected {expected[:12]}…, got {actual[:12]}…"
    result.add("bundle-sha256", match, detail)
    return result


def _resolve_bundle_path(settings: Settings) -> Path:
    """Locate a bundle to attest, building a serving bundle if none exists yet."""
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.bundle import build_serving_bundle

    configured = settings.serving.artifact_path or latest_bundle(settings)
    if configured is not None and Path(configured).exists():
        return Path(configured)
    logger.info("No model bundle found; building a serving bundle (requires `prep`).")
    return build_serving_bundle(settings)


def run_provenance_report(settings: Settings) -> Path:
    """Write the SBOM, the model manifest, and the human-readable provenance report."""
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = repo_root / "pyproject.toml"
    bundle_path = _resolve_bundle_path(settings)

    sbom = build_sbom(settings, pyproject)
    manifest = build_manifest(settings, bundle_path, sbom=sbom)

    reports_dir = settings.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / SBOM_NAME).write_text(json.dumps(sbom, indent=2), encoding="utf-8")
    (reports_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out_path = reports_dir / REPORT_NAME
    out_path.write_text(_render(sbom, manifest), encoding="utf-8")
    logger.info(
        "Wrote provenance artifacts",
        extra={"report": str(out_path), "components": len(sbom["components"])},
    )
    return out_path


def _render(sbom: dict[str, Any], manifest: dict[str, Any]) -> str:
    bundle = manifest["bundle"]
    model = manifest["model"]
    props = {p["name"]: p["value"] for p in sbom["metadata"].get("properties", [])}
    missing = props.get("netsentry:declared-not-installed", "none")
    model_line = (
        f"{model['backend']} · {model['task']} · {model['n_features']} features · "
        f"{len(model['classes'] or [])} classes"
    )
    components_line = (
        f"calibrator: {model['has_calibrator']} · anomaly detector: "
        f"{model['has_anomaly_detector']} · conformal: {model['has_conformal']}"
    )
    profiles = ", ".join(f"`{p}`" for p in model["threshold_profiles"]) or "none"
    runtime = f"Python {manifest['runtime']['python']} · {manifest['runtime']['platform']}"
    config_line = (
        f"`{manifest['config']['digest_sha256'][:24]}…` (seed {manifest['config']['seed']})"
    )

    comp_rows = ["| component | version | groups |", "|---|---|---|"]
    for c in sbom["components"]:
        groups = ", ".join(p["value"] for p in c.get("properties", []))
        comp_rows.append(f"| `{c['name']}` | {c['version']} | {groups} |")

    return f"""# NetSentry — Provenance & Supply Chain

_Generated by `netsentry provenance`. The machine-readable artifacts live beside
this file: [`{SBOM_NAME}`]({SBOM_NAME}) (CycloneDX {sbom['specVersion']}) and
[`{MANIFEST_NAME}`]({MANIFEST_NAME}) (model integrity manifest)._

A trained model is an opaque binary that decides whether traffic is malicious. Two
questions a reviewer or a deploy gate should be able to answer without retraining:
**what went into it**, and **has it been altered**. The SBOM answers the first; the
manifest and `netsentry verify` answer the second.

## Model manifest

| field | value |
|---|---|
| bundle | `{bundle['name']}` ({bundle['size_bytes']:,} bytes) |
| bundle SHA-256 | `{bundle['sha256']}` |
| git commit | `{manifest['git_commit'] or 'unknown'}` |
| runtime | {runtime} |
| config digest | {config_line} |
| model | {model_line} |
| attached components | {components_line} |
| threshold profiles | {profiles} |

Verify integrity against the manifest before promoting a model:

```bash
netsentry verify            # recomputes the bundle SHA-256, fails on mismatch
```

This is the gate that catches a corrupted download, a swapped artifact, or a
tampered model — the model-serving analogue of checking a package signature.

## Software bill of materials (CycloneDX {sbom['specVersion']})

Scope: the project's **declared** dependencies resolved to the versions installed
in the build environment — a bounded, auditable BOM, not the full transitive
environment. This is what a CVE sweep or a licence audit consumes.
Declared but not installed in this environment: {missing}.

{chr(10).join(comp_rows)}

The `purl` (Package URL) on each SBOM component is the key vulnerability scanners
(Grype, Trivy, Dependency-Track) match CVE advisories against, so this file drops
straight into an existing supply-chain pipeline.
"""
