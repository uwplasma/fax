from __future__ import annotations

from importlib import util
import io
from pathlib import Path
import sys
import tarfile

import pytest


def _load_installer_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "install_public_cross_code_artifacts.py"
    spec = util.spec_from_file_location("install_public_cross_code_artifacts", path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_cross_code_artifact_safe_extract_accepts_plain_files(tmp_path: Path):
    module = _load_installer_module()
    archive = tmp_path / "safe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        payload = b"ok\n"
        member = tarfile.TarInfo("nested/file.txt")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))

    target = tmp_path / "target"
    target.mkdir()
    module._safe_extract_tar(archive, target)

    assert (target / "nested" / "file.txt").read_text(encoding="utf-8") == "ok\n"


def test_public_cross_code_artifact_safe_extract_rejects_path_traversal(tmp_path: Path):
    module = _load_installer_module()
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        payload = b"bad\n"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))

    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError, match="unsafe path"):
        module._safe_extract_tar(archive, target)
