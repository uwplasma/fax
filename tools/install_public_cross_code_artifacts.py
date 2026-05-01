#!/usr/bin/env python3
"""Install public ORNL STELLGAP/AE3D validation artifacts for FAX tests.

The FAX validation gates intentionally read the public ORNL test packets from
``/Users/rogerio/local`` because those paths also match the local development
machines used for cross-code work. This script makes that layout reproducible
on CI runners without committing large third-party artifacts into the FAX repo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tarfile


@dataclass(frozen=True)
class ArchiveSpec:
    """One public archive that should be extracted under the validation root."""

    repo_name: str
    repo_url: str
    archive_name: str
    target_dir: str
    required_files: tuple[str, ...]


ARCHIVES: tuple[ArchiveSpec, ...] = (
    ArchiveSpec(
        repo_name="Stellgap",
        repo_url="https://github.com/ORNL-Fusion/Stellgap.git",
        archive_name="stellgap_test_problem.tar.gz",
        target_dir="Stellgap_test_inputs",
        required_files=("boozmn_ctok.nc", "fourier.dat", "plasma.dat", "tae_data_boozer"),
    ),
    ArchiveSpec(
        repo_name="Stellgap",
        repo_url="https://github.com/ORNL-Fusion/Stellgap.git",
        archive_name="stellgap_test_problem_answers.tar.gz",
        target_dir="Stellgap_test_problem",
        required_files=("alfven_post", "alfven_spec"),
    ),
    ArchiveSpec(
        repo_name="AE3D",
        repo_url="https://github.com/ORNL-Fusion/AE3D.git",
        archive_name="ae3d_test_problem.tar.gz",
        target_dir="AE3D_test_inputs",
        required_files=("boozmn_ctok.nc", "ae_metric.dat", "fourier.dat", "plasma.dat"),
    ),
    ArchiveSpec(
        repo_name="AE3D",
        repo_url="https://github.com/ORNL-Fusion/AE3D.git",
        archive_name="ae3d_test_problem_answers.tar.gz",
        target_dir="AE3D_test_problem",
        required_files=("a_matrix.dat", "b_matrix.dat", "egn_mode_asci.dat", "egn_values.dat", "omega2.dat"),
    ),
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _ensure_repo(root: Path, spec: ArchiveSpec) -> Path:
    repo = root / spec.repo_name
    if repo.exists():
        return repo
    _run(["git", "clone", "--depth", "1", spec.repo_url, str(repo)])
    return repo


def _safe_extract_tar(archive: Path, target: Path) -> None:
    """Extract regular files/directories while rejecting traversal and links."""
    target_resolved = target.resolve()
    with tarfile.open(archive, mode="r:gz") as stream:
        for member in stream.getmembers():
            destination = (target / member.name).resolve()
            if not str(destination).startswith(str(target_resolved) + "/") and destination != target_resolved:
                raise ValueError(f"unsafe path in {archive}: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"refusing link member in {archive}: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"refusing non-file member in {archive}: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"could not read file member in {archive}: {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def _has_required_files(target: Path, required_files: tuple[str, ...]) -> bool:
    return all((target / name).exists() for name in required_files)


def install_public_cross_code_artifacts(root: Path) -> list[str]:
    """Install all public artifact packets and return user-facing status lines."""
    root.mkdir(parents=True, exist_ok=True)
    status: list[str] = []
    for spec in ARCHIVES:
        repo = _ensure_repo(root, spec)
        archive = repo / spec.archive_name
        if not archive.exists():
            raise FileNotFoundError(f"missing expected public archive: {archive}")
        target = root / spec.target_dir
        target.mkdir(parents=True, exist_ok=True)
        if _has_required_files(target, spec.required_files):
            status.append(f"{spec.target_dir}: already present")
            continue
        _safe_extract_tar(archive, target)
        if not _has_required_files(target, spec.required_files):
            missing = [name for name in spec.required_files if not (target / name).exists()]
            raise FileNotFoundError(f"{spec.target_dir} is missing required files after extraction: {missing}")
        status.append(f"{spec.target_dir}: extracted from {spec.repo_name}/{spec.archive_name}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/Users/rogerio/local",
        help="Artifact root expected by FAX validation gates. Default: /Users/rogerio/local",
    )
    args = parser.parse_args()
    for line in install_public_cross_code_artifacts(Path(args.root)):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
