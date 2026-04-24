"""Known local/literature reference cases for development and validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ReferenceCase:
    """Metadata and local paths for a reference case.

    Paths may point outside the ``fax`` repository because early validation uses
    neighboring checkouts such as ``vmec_jax`` and ``simsopt``.
    """

    case_id: str
    description: str
    vmec_input: Path
    wout: Path | None
    references: tuple[str, ...]
    tags: tuple[str, ...]

    def missing_paths(self) -> tuple[Path, ...]:
        """Return required paths that are unavailable on this machine."""
        paths: list[Path] = [self.vmec_input]
        if self.wout is not None:
            paths.append(self.wout)
        return tuple(path for path in paths if not path.exists())

    def is_available(self) -> bool:
        """Return whether all required local paths exist."""
        return not self.missing_paths()


def local_root() -> Path:
    """Return the local workspace root used for external reference cases."""
    return Path(os.environ.get("FAX_LOCAL_ROOT", Path.home() / "local")).expanduser()


def landreman_paul_qa_lowres(*, root: str | Path | None = None) -> ReferenceCase:
    """Return the first VMEC-backed stellarator reference target for ``fax``.

    The case is the low-resolution version of the Landreman-Paul precise
    quasi-axisymmetric configuration. It is useful as the first stellarator
    geometry target because it is literature anchored, small enough for local
    tests, and already used in the neighboring ``vmec_jax`` and
    ``booz_xform_jax`` workflows.
    """
    base = Path(root).expanduser() if root is not None else local_root()
    vmec_input = Path(
        os.environ.get(
            "FAX_LP_QA_VMEC_INPUT",
            str(base / "vmec_jax" / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"),
        )
    ).expanduser()

    wout_env = os.environ.get("FAX_LP_QA_WOUT")
    wout_candidates = [
        Path(wout_env).expanduser() if wout_env else None,
        base / "vmec_jax" / "examples" / "data" / "wout_LandremanPaul2021_QA_lowres.nc",
        base / "simsopt" / "tests" / "test_files" / "wout_LandremanPaul2021_QA_lowres.nc",
    ]
    wout = next((path for path in wout_candidates if path is not None and path.exists()), wout_candidates[1])

    return ReferenceCase(
        case_id="LandremanPaul2021_QA_lowres",
        description="Low-resolution precise quasi-axisymmetric stellarator configuration.",
        vmec_input=vmec_input,
        wout=wout,
        references=(
            "https://arxiv.org/abs/2108.03711",
            "https://doi.org/10.1103/PhysRevLett.128.035001",
            "https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/"
            "shear-alfven-continuum-of-quasisymmetric-stellarators/"
            "7B50E360D4324D8AF93DA73EB25C35D8",
        ),
        tags=("stellarator", "quasi-axisymmetric", "vmec", "lowres", "first-reference"),
    )

