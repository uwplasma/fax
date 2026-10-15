"""Known local/literature reference cases for development and validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ReferenceCase:
    """Metadata and local paths for a reference case.

    Paths may point outside the ``fax`` repository because early validation uses
    neighboring checkouts such as ``vmex`` and ``simsopt``.
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


@dataclass(frozen=True)
class VmecEquilibriaCase:
    """One equilibrium from the external ``landreman/vmec_equilibria`` repo.

    The repository is intentionally kept outside the FAX source tree because it
    is a large data checkout. Paths are resolved at runtime from
    ``FAX_VMEC_EQUILIBRIA_ROOT`` or the default local workspace clone.
    """

    case_id: str
    family: str
    description: str
    relative_wout: Path
    relative_vmec_input: Path | None
    references: tuple[str, ...]
    tags: tuple[str, ...]
    relative_boozmn: Path | None = None

    def wout_path(self, *, root: str | Path | None = None) -> Path:
        """Return the absolute local VMEC ``wout`` path."""
        return vmec_equilibria_root(root=root) / self.relative_wout

    def vmec_input_path(self, *, root: str | Path | None = None) -> Path | None:
        """Return the absolute local VMEC input path, if cataloged."""
        if self.relative_vmec_input is None:
            return None
        return vmec_equilibria_root(root=root) / self.relative_vmec_input

    def boozmn_path(self, *, root: str | Path | None = None) -> Path | None:
        """Return the absolute local Boozer ``boozmn`` path, if cataloged."""
        if self.relative_boozmn is None:
            return None
        return vmec_equilibria_root(root=root) / self.relative_boozmn

    def missing_paths(self, *, root: str | Path | None = None) -> tuple[Path, ...]:
        """Return required local paths that are unavailable."""
        paths: list[Path] = [self.wout_path(root=root)]
        vmec_input = self.vmec_input_path(root=root)
        if vmec_input is not None:
            paths.append(vmec_input)
        return tuple(path for path in paths if not path.exists())

    def is_available(self, *, root: str | Path | None = None) -> bool:
        """Return whether all required local paths exist."""
        return not self.missing_paths(root=root)

    def is_boozmn_available(self, *, root: str | Path | None = None) -> bool:
        """Return whether a cataloged local ``boozmn`` artifact exists."""
        boozmn = self.boozmn_path(root=root)
        return boozmn is not None and boozmn.exists()


def local_root() -> Path:
    """Return the local workspace root used for external reference cases."""
    return Path(os.environ.get("FAX_LOCAL_ROOT", Path.home() / "local")).expanduser()


def vmec_equilibria_root(*, root: str | Path | None = None) -> Path:
    """Return the local clone root for ``landreman/vmec_equilibria``."""
    if root is not None:
        return Path(root).expanduser()
    return Path(
        os.environ.get(
            "FAX_VMEC_EQUILIBRIA_ROOT",
            str(local_root() / "vmec_equilibria"),
        )
    ).expanduser()


def vmec_equilibria_catalog() -> tuple[VmecEquilibriaCase, ...]:
    """Return the curated multi-geometry VMEC-equilibria benchmark catalog.

    These are not bundled into the Python package. They are external literature
    or device equilibria used to exercise FAX on a broad set of already
    converged VMEC ``wout`` files: quasi-helical, quasi-axisymmetric, heliotron,
    heliac, optimized stellarator, reactor, and axisymmetric references.
    """
    repo = "https://github.com/landreman/vmec_equilibria"
    return (
        VmecEquilibriaCase(
            case_id="hsx_qhs_vac",
            family="HSX",
            description="HSX quasi-helically symmetric vacuum equilibrium.",
            relative_wout=Path("HSX/QHS_vac/wout_HSX_QHS_vac.nc"),
            relative_vmec_input=None,
            references=(repo, "HSX/QHS_vac/README"),
            tags=("stellarator", "quasi-helical", "vacuum", "boozmn-available"),
            relative_boozmn=Path("HSX/QHS_vac/boozmn_HSX_QHS_vac.nc"),
        ),
        VmecEquilibriaCase(
            case_id="hsx_mirror_flip14",
            family="HSX",
            description="HSX mirror configuration with public Boozer harmonics.",
            relative_wout=Path("HSX/Mirror_flip14/wout_MIRFL14_11p.nc"),
            relative_vmec_input=None,
            references=(repo, "HSX/Mirror_flip14"),
            tags=("stellarator", "quasi-helical", "mirror", "boozmn-available"),
            relative_boozmn=Path("HSX/Mirror_flip14/boozmn_MIRFL14_11p.nc"),
        ),
        VmecEquilibriaCase(
            case_id="hsx_filtered",
            family="HSX",
            description="HSX configuration filtered to reduce modular-ripple harmonics.",
            relative_wout=Path("HSX/filtered_to_remove_modular_ripple/wout_hsxt.nc"),
            relative_vmec_input=Path("HSX/filtered_to_remove_modular_ripple/input.hsxt"),
            references=(repo, "HSX/filtered_to_remove_modular_ripple/README"),
            tags=("stellarator", "quasi-helical", "filtered"),
        ),
        VmecEquilibriaCase(
            case_id="nuhrenberg_zille_qhs_ns51",
            family="Nuhrenberg-Zille",
            description="Nuhrenberg-Zille 1988 quasi-helically symmetric reference.",
            relative_wout=Path("NuhrenbergZille_1988_QHS/ns51/wout_NuhrenbergZille_1988_QHS.nc"),
            relative_vmec_input=Path("NuhrenbergZille_1988_QHS/ns51/input.NuhrenbergZille_1988_QHS"),
            references=(repo, "Nuhrenberg and Zille 1988 quasi-helical stellarator reference"),
            tags=("stellarator", "quasi-helical", "literature"),
        ),
        VmecEquilibriaCase(
            case_id="qhs46_ns201",
            family="QHS46",
            description="QHS46 WISTELL quasi-helical base case.",
            relative_wout=Path("QHS46/ns201/wout_qhs46_ns201.nc"),
            relative_vmec_input=Path("QHS46/ns201/input.qhs46_ns201"),
            references=(repo, "QHS46/wistell_website_base_case/README"),
            tags=("stellarator", "quasi-helical", "wistell"),
        ),
        VmecEquilibriaCase(
            case_id="ku_boozer_n4qh_b4_a79a",
            family="Ku-Boozer",
            description="Ku-Boozer 2011 Nuclear Fusion quasi-helical stellarator with public Boozer harmonics.",
            relative_wout=Path("KuBoozer_2011_NuclFusion_QHS/n4qh.b4.a79a/wout_n4qh.b4.a79a.nc"),
            relative_vmec_input=Path("KuBoozer_2011_NuclFusion_QHS/n4qh.b4.a79a/input.n4qh.b4.a79a"),
            references=(repo, "Ku and Boozer 2011 Nuclear Fusion quasi-helical stellarator"),
            tags=("stellarator", "quasi-helical", "literature", "boozmn-available"),
            relative_boozmn=Path("KuBoozer_2011_NuclFusion_QHS/n4qh.b4.a79a/boozmn_n4qh.b4.a79a.nc"),
        ),
        VmecEquilibriaCase(
            case_id="w7x_standard",
            family="W7-X",
            description="W7-X standard configuration from the public VMEC-equilibria set.",
            relative_wout=Path("W7-X/Standard/wout.nc"),
            relative_vmec_input=Path("W7-X/Standard/input.txt"),
            references=(repo, "W7-X/Standard/README"),
            tags=("stellarator", "optimized", "w7-x"),
        ),
        VmecEquilibriaCase(
            case_id="w7x_high_mirror",
            family="W7-X",
            description="W7-X high-mirror configuration from the public VMEC-equilibria set.",
            relative_wout=Path("W7-X/HighMirror/wout.nc"),
            relative_vmec_input=Path("W7-X/HighMirror/input.txt"),
            references=(repo, "W7-X/HighMirror/README"),
            tags=("stellarator", "optimized", "w7-x"),
        ),
        VmecEquilibriaCase(
            case_id="ncsx_li383",
            family="NCSX",
            description="NCSX li383 finite-beta stellarator equilibrium.",
            relative_wout=Path("NCSX/li383_1.4m/wout_li383_1.4m.nc"),
            relative_vmec_input=Path("NCSX/li383_1.4m/input.li383_1.4m"),
            references=(repo, "NCSX/li383_1.4m"),
            tags=("stellarator", "quasi-axisymmetric", "ncsx"),
        ),
        VmecEquilibriaCase(
            case_id="aries_cs",
            family="ARIES-CS",
            description="ARIES-CS reactor-scale compact-stellarator equilibrium.",
            relative_wout=Path("ARIES-CS/n3are_R7.75B5.7/wout_n3are_R7.75B5.7.nc"),
            relative_vmec_input=Path("ARIES-CS/n3are_R7.75B5.7/input.n3are_R7.75B5.7"),
            references=(repo, "ARIES-CS/README"),
            tags=("stellarator", "reactor", "compact-stellarator"),
        ),
        VmecEquilibriaCase(
            case_id="lhd_090493t2330",
            family="LHD",
            description="Large Helical Device experimental equilibrium.",
            relative_wout=Path("LHD/090493t2330/wout_LHD_090493t2330.nc"),
            relative_vmec_input=None,
            references=(repo, "LHD/090493t2330/README"),
            tags=("stellarator", "heliotron", "experiment"),
        ),
        VmecEquilibriaCase(
            case_id="tjii_100_44_64",
            family="TJ-II",
            description="TJ-II heliac equilibrium.",
            relative_wout=Path("TJ-II/100_44_64_0/wout_100_44_64_0.0.nc"),
            relative_vmec_input=Path("TJ-II/100_44_64_0/input.100_44_64_0.0"),
            references=(repo, "TJ-II/100_44_64_0"),
            tags=("stellarator", "heliac", "experiment"),
        ),
        VmecEquilibriaCase(
            case_id="iter_model",
            family="ITER",
            description="Axisymmetric ITER-like VMEC reference for tokamak contrast.",
            relative_wout=Path("ITER_model/ns201/wout_ITERModel.nc"),
            relative_vmec_input=Path("ITER_model/ns201/input.ITERModel"),
            references=(repo, "ITER_model/README"),
            tags=("tokamak", "axisymmetric", "control"),
        ),
    )


def available_vmec_equilibria_cases(*, root: str | Path | None = None) -> tuple[VmecEquilibriaCase, ...]:
    """Return cataloged VMEC-equilibria cases available on this machine."""
    return tuple(case for case in vmec_equilibria_catalog() if case.is_available(root=root))


def available_vmec_equilibria_boozmn_cases(*, root: str | Path | None = None) -> tuple[VmecEquilibriaCase, ...]:
    """Return cataloged VMEC-equilibria cases with local ``boozmn`` artifacts."""
    return tuple(case for case in vmec_equilibria_catalog() if case.is_boozmn_available(root=root))


def landreman_paul_qa_lowres(*, root: str | Path | None = None) -> ReferenceCase:
    """Return the first VMEC-backed stellarator reference target for ``fax``.

    The case is the low-resolution version of the Landreman-Paul precise
    quasi-axisymmetric configuration. It is useful as the first stellarator
    geometry target because it is literature anchored, small enough for local
    tests, and already used in the neighboring ``vmex`` and
    ``booz_xform_jax`` workflows.
    """
    base = Path(root).expanduser() if root is not None else local_root()
    vmec_input = Path(
        os.environ.get(
            "FAX_LP_QA_VMEC_INPUT",
            str(base / "vmex" / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"),
        )
    ).expanduser()

    wout_env = os.environ.get("FAX_LP_QA_WOUT")
    wout_candidates = [
        Path(wout_env).expanduser() if wout_env else None,
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
