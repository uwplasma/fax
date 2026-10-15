"""Configuration loading for ``fax`` TOML input files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(ValueError):
    """Raised when a ``fax`` configuration file is invalid."""


@dataclass(frozen=True)
class RunConfig:
    """Top-level run controls."""

    case: str = "fax_case"
    model: str = "continuum"
    output_dir: str = "outputs"


@dataclass(frozen=True)
class EquilibriumConfig:
    """Equilibrium and coordinate-source controls."""

    source: str = "vmex"
    vmec_input: str | None = None
    wout: str | None = None
    use_boozer: bool = False
    boozmn: str | None = None


@dataclass(frozen=True)
class GridConfig:
    """Resolution controls for radial and Fourier representations."""

    radial_points: int = 64
    m_max: int = 8
    n_max: int = 8
    field_periods: int | None = None


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics feature flags."""

    include_sound_coupling: bool = False
    include_fast_ions: bool = False
    fast_ion_model: str = "none"
    density_scale: float = 1.0


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical-method and JAX execution controls."""

    precision: str = "float64"
    jit: bool = True
    autodiff: bool = True
    eigensolver: str = "dense"


@dataclass(frozen=True)
class FaxConfig:
    """Structured configuration used by public APIs and the CLI."""

    run: RunConfig = field(default_factory=RunConfig)
    equilibrium: EquilibriumConfig = field(default_factory=EquilibriumConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "FaxConfig":
        """Build a typed config from a parsed TOML mapping."""
        if not isinstance(mapping, dict):
            raise ConfigError("configuration root must be a TOML table")

        known_sections = {"run", "equilibrium", "grid", "physics", "numerics"}
        unknown = sorted(set(mapping) - known_sections)
        if unknown:
            raise ConfigError(f"unknown configuration section(s): {', '.join(unknown)}")

        return cls(
            run=_build_dataclass(RunConfig, mapping.get("run", {})),
            equilibrium=_build_dataclass(EquilibriumConfig, mapping.get("equilibrium", {})),
            grid=_build_dataclass(GridConfig, mapping.get("grid", {})),
            physics=_build_dataclass(PhysicsConfig, mapping.get("physics", {})),
            numerics=_build_dataclass(NumericsConfig, mapping.get("numerics", {})),
        ).validated()

    def validated(self) -> "FaxConfig":
        """Return ``self`` after enforcing basic, model-independent constraints."""
        if not self.run.case:
            raise ConfigError("run.case must be non-empty")
        if self.run.model not in {"continuum", "global_eigenmode", "fast_ion_stability"}:
            raise ConfigError(
                "run.model must be one of: continuum, global_eigenmode, fast_ion_stability"
            )
        if self.equilibrium.source not in {"vmex", "wout", "analytic"}:
            raise ConfigError("equilibrium.source must be one of: vmex, wout, analytic")
        if self.grid.radial_points < 4:
            raise ConfigError("grid.radial_points must be at least 4")
        if self.grid.m_max < 0 or self.grid.n_max < 0:
            raise ConfigError("grid.m_max and grid.n_max must be non-negative")
        if self.grid.field_periods is not None and self.grid.field_periods <= 0:
            raise ConfigError("grid.field_periods must be positive when provided")
        if self.physics.density_scale <= 0:
            raise ConfigError("physics.density_scale must be positive")
        if self.physics.fast_ion_model not in {"none", "moment", "drift_kinetic", "delta_f"}:
            raise ConfigError("physics.fast_ion_model must be one of: none, moment, drift_kinetic, delta_f")
        if self.numerics.precision not in {"float32", "float64"}:
            raise ConfigError("numerics.precision must be float32 or float64")
        if self.numerics.eigensolver not in {"dense", "krylov", "lobpcg"}:
            raise ConfigError("numerics.eigensolver must be one of: dense, krylov, lobpcg")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _build_dataclass(cls: type[Any], data: Any) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} must be a TOML table")

    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ConfigError(f"unknown key(s) in {cls.__name__}: {', '.join(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> FaxConfig:
    """Load a ``FaxConfig`` from a TOML file."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse TOML in {config_path}: {exc}") from exc

    return FaxConfig.from_mapping(raw)

