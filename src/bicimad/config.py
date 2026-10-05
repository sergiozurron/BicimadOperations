"""Paths, constants, and DATA_SCOPE resolution for the BiciMAD pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

MADRID_TZ = "Europe/Madrid"
MADRID_BBOX = {
    "lat_min": 40.30,
    "lat_max": 40.57,
    "lon_min": -3.85,
    "lon_max": -3.52,
}

MIN_TRIP_MINUTES = 1.0
MAX_TRIP_MINUTES = 240.0
CAPACITY_TOLERANCE = 2  # booked docks can make dock_bikes + free_bases slightly exceed capacity
RANDOM_SEED = 42

Scope = Literal["demo", "2022", "all"]


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or cwd) until this project is found.

    `data/raw` is created by the download step, so a checkout that has not
    fetched the EMT archives yet is still recognised via `src/bicimad`.
    """
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "src" / "bicimad" / "__init__.py").is_file():
            return candidate
    for candidate in [cur, *cur.parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate
    return cur


@dataclass
class ProjectConfig:
    root: Path = field(default_factory=find_project_root)
    data_scope: Scope = "demo"
    demo_months: tuple[str, ...] = ("2022-10",)
    rebuild_parquet: bool = False
    random_seed: int = RANDOM_SEED
    time_bin: str = "h"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def trips_raw(self) -> Path:
        return self.raw_dir / "trips"

    @property
    def status_raw(self) -> Path:
        return self.raw_dir / "status"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def feature_dir(self) -> Path:
        return self.root / "data" / "feature_store"

    @property
    def models_dir(self) -> Path:
        return self.root / "data" / "models"

    @property
    def eval_dir(self) -> Path:
        return self.root / "data" / "evaluation"

    @property
    def weather_dir(self) -> Path:
        return self.processed_dir / "weather"

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.feature_dir, self.models_dir, self.eval_dir, self.weather_dir):
            d.mkdir(parents=True, exist_ok=True)

    def months_for_scope(self) -> list[str]:
        inventory = discover_months(self)
        if self.data_scope == "demo":
            return [m for m in self.demo_months if m in inventory["union"]]
        if self.data_scope == "2022":
            return [m for m in inventory["union"] if m.startswith("2022-")]
        return inventory["union"]


def _month_from_stem(stem: str) -> str | None:
    """Parse YYYY-MM from the heterogeneous BiciMAD filenames."""
    patterns = [
        r"(?:Bicimad_(?:Estacions|Stations)_)?(\d{4})(\d{2})(?:_stations)?$",
        r"^(\d{4})(\d{2})_(?:Usage_Bicimad|movements)$",
        r"^trips_(\d{2})_(\d{2})_",
    ]
    for i, pat in enumerate(patterns):
        m = re.search(pat, stem)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if i == 2:  # trips_22_10_October
            year = 2000 + int(a)
            return f"{year:04d}-{int(b):02d}"
        return f"{a}-{b}"
    return None


def discover_trip_files(cfg: ProjectConfig) -> dict[str, Path]:
    """Map YYYY-MM -> preferred trip file (CSV wins over JSON on overlap)."""
    json_map: dict[str, Path] = {}
    csv_map: dict[str, Path] = {}
    for path in sorted(cfg.trips_raw.iterdir()):
        if not path.is_file():
            continue
        month = _month_from_stem(path.stem)
        if month is None:
            continue
        if path.suffix.lower() == ".csv":
            csv_map[month] = path
        else:
            json_map[month] = path
    out = dict(json_map)
    out.update(csv_map)
    return dict(sorted(out.items()))


def discover_status_files(cfg: ProjectConfig) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(cfg.status_raw.iterdir()):
        if not path.is_file():
            continue
        month = _month_from_stem(path.stem)
        if month is None:
            continue
        out[month] = path
    return dict(sorted(out.items()))


def discover_months(cfg: ProjectConfig) -> dict[str, list[str]]:
    trips = set(discover_trip_files(cfg))
    status = set(discover_status_files(cfg))
    return {
        "trips": sorted(trips),
        "status": sorted(status),
        "union": sorted(trips | status),
        "both": sorted(trips & status),
        "trips_only": sorted(trips - status),
        "status_only": sorted(status - trips),
    }


def trip_schema_for(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv"):
        return "csv"
    if "usage_bicimad" in name:
        return "json_v1"
    return "json_v2"


def months_in(months: Iterable[str], year: int | None = None) -> list[str]:
    months = list(months)
    if year is None:
        return months
    prefix = f"{year:04d}-"
    return [m for m in months if m.startswith(prefix)]
