"""Versioned external-code benchmark manifest.

The manifest records public or local cross-code artifacts that can be used for
research-grade parity work. It does not execute the comparisons; it defines the
fixtures, claim level, expected metrics, and file hashes needed for reproducible
benchmark packets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkArtifact:
    """One file used by an external-code benchmark case."""

    role: str
    path: str
    required: bool = True

    def resolved_path(self) -> Path:
        """Return the expanded local artifact path."""
        return Path(self.path).expanduser()

    def exists(self) -> bool:
        """Return whether the artifact exists locally."""
        return self.resolved_path().exists()

    def sha256(self) -> str | None:
        """Return the artifact SHA-256 hash, or ``None`` if the file is absent."""
        artifact = self.resolved_path()
        if not artifact.exists() or not artifact.is_file():
            return None
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable artifact record."""
        payload = asdict(self)
        payload["exists"] = self.exists()
        if include_hash:
            payload["sha256"] = self.sha256()
        return payload


@dataclass(frozen=True)
class BenchmarkManifestEntry:
    """One benchmark case and its reproducibility contract."""

    benchmark_id: str
    code: str
    geometry: str
    physics_model: str
    claim_level: str
    status: str
    artifacts: tuple[BenchmarkArtifact, ...]
    expected_metrics: dict[str, float]
    validation_gate_ids: tuple[str, ...]
    next_parity_step: str

    def available(self) -> bool:
        """Return whether all required artifacts are present locally."""
        return all(artifact.exists() for artifact in self.artifacts if artifact.required)

    def to_dict(self, *, include_hashes: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable manifest entry."""
        payload = asdict(self)
        payload["available"] = self.available()
        payload["artifacts"] = [artifact.to_dict(include_hash=include_hashes) for artifact in self.artifacts]
        return payload


def default_benchmark_manifest() -> tuple[BenchmarkManifestEntry, ...]:
    """Return the current built-in cross-code benchmark manifest."""
    return (
        BenchmarkManifestEntry(
            benchmark_id="stellgap_ctok_local_continuum_gap",
            code="STELLGAP",
            geometry="CTOK public test problem",
            physics_model="local 3-D Boozer shear-Alfven continuum and local gap parity",
            claim_level="local continuum/gap parity, not global eigenmode parity",
            status="active",
            artifacts=(
                BenchmarkArtifact(
                    role="boozmn geometry",
                    path="/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc",
                ),
                BenchmarkArtifact(
                    role="STELLGAP alfven_post reference",
                    path="/Users/rogerio/local/Stellgap_test_problem/alfven_post",
                ),
                BenchmarkArtifact(
                    role="STELLGAP alfven_spec raw output",
                    path="/Users/rogerio/local/Stellgap_test_problem/alfven_spec",
                    required=False,
                ),
            ),
            expected_metrics={
                "p95_sorted_spectrum_relative_error_max": 4.0e-2,
                "p95_largest_gap_center_relative_error_max": 6.0e-2,
                "p95_largest_gap_width_relative_error_max": 6.0e-2,
            },
            validation_gate_ids=(
                "stellgap_ctok_local_continuum",
                "stellgap_ctok_largest_gap_parity",
                "stellgap_public_postprocess_parity",
            ),
            next_parity_step=(
                "Add matched global eigenfunction/frequency outputs and compare native "
                "FAX radial modes against STELLGAP/AE3D on the same CTOK convention."
            ),
        ),
        BenchmarkManifestEntry(
            benchmark_id="ae3d_public_matrix_mode_shape",
            code="AE3D",
            geometry="CTOK public generalized matrix problem",
            physics_model="public matrix/eigenmode artifact parity",
            claim_level="artifact/matrix/mode-shape parity, not native FAX operator parity",
            status="active",
            artifacts=(
                BenchmarkArtifact(role="AE3D stiffness matrix", path="/Users/rogerio/local/AE3D_test_problem/a_matrix.dat"),
                BenchmarkArtifact(role="AE3D mass matrix", path="/Users/rogerio/local/AE3D_test_problem/b_matrix.dat"),
                BenchmarkArtifact(role="AE3D eigenmode table", path="/Users/rogerio/local/AE3D_test_problem/egn_mode_asci.dat"),
                BenchmarkArtifact(role="AE3D eigenvalue table", path="/Users/rogerio/local/AE3D_test_problem/egn_values.dat"),
                BenchmarkArtifact(role="AE3D raw omega2 table", path="/Users/rogerio/local/AE3D_test_problem/omega2.dat"),
            ),
            expected_metrics={
                "max_frequency_relative_error": 2.0e-6,
                "max_mass_weighted_mode_shape_error": 2.0e-6,
                "max_matrix_relative_asymmetry": 2.0e-6,
            },
            validation_gate_ids=(
                "ae3d_public_matrix_symmetry",
                "ae3d_public_matrix_eigenmode_parity",
                "ae3d_public_dense_spectrum_parity",
                "ae3d_public_global_mode_shape_parity",
            ),
            next_parity_step=(
                "Map AE3D basis/boundary conventions to a native FAX global operator "
                "and compare FAX-computed modes against AE3D modes."
            ),
        ),
    )


def benchmark_manifest_summary(*, include_hashes: bool = False) -> dict[str, Any]:
    """Return a JSON-serializable summary of built-in benchmark artifacts."""
    entries = default_benchmark_manifest()
    available_count = sum(entry.available() for entry in entries)
    return {
        "entry_count": len(entries),
        "available_count": available_count,
        "all_required_available": available_count == len(entries),
        "entries": [entry.to_dict(include_hashes=include_hashes) for entry in entries],
    }


def benchmark_manifest_markdown(*, include_hashes: bool = False) -> str:
    """Return a Markdown report for the built-in benchmark manifest."""
    summary = benchmark_manifest_summary(include_hashes=include_hashes)
    lines = [
        "# FAX Benchmark Manifest",
        "",
        f"Entries available locally: **{summary['available_count']}/{summary['entry_count']}**.",
        "",
        "| Benchmark | Code | Status | Available | Claim Level | Next Parity Step |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in default_benchmark_manifest():
        lines.append(
            f"| `{entry.benchmark_id}` | {entry.code} | {entry.status} | "
            f"{'yes' if entry.available() else 'no'} | {entry.claim_level} | {entry.next_parity_step} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for entry in default_benchmark_manifest():
        lines.append(f"### {entry.benchmark_id}")
        for artifact in entry.artifacts:
            record = artifact.to_dict(include_hash=include_hashes)
            suffix = f" sha256={record['sha256']}" if include_hashes and record.get("sha256") else ""
            lines.append(
                f"- {artifact.role}: `{artifact.path}` "
                f"({'present' if record['exists'] else 'missing'}, required={artifact.required}){suffix}"
            )
        lines.append("")
    return "\n".join(lines)
