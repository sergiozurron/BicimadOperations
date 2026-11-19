"""Ingest adapters for the three trip schemas and hourly station-status NDJSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    MADRID_TZ,
    ProjectConfig,
    discover_status_files,
    discover_trip_files,
    trip_schema_for,
)

TRIP_COLUMNS = [
    "trip_id",
    "start_ts",
    "end_ts",
    "start_station",
    "end_station",
    "travel_time_min",
    "user_type",
    "age_range",
    "bike_id",
    "lock_type",
    "user_day_code",
    "source_schema",
    "source_file",
]

STATUS_COLUMNS = [
    "ts",
    "station_id",
    "name",
    "number",
    "address",
    "lat",
    "lon",
    "total_bases",
    "dock_bikes",
    "free_bases",
    "activate",
    "no_available",
    "reservations_count",
    "light",
]


def _to_utc_datetime(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, dict) and "$date" in value:
        raw = str(value["$date"])
        # Mongo export: 2018-09-01T01:00:00.000+0200 -> +02:00
        raw = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw)
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        return ts
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts


def _nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _extract_oid(obj) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "$oid" in obj:
            return str(obj["$oid"])
        return str(obj.get("_id") or obj)
    return str(obj)


def parse_trips_json_line(obj: dict, schema: str, source_file: str) -> dict:
    start_ts = _to_utc_datetime(obj.get("unplug_hourTime"))
    travel_sec = pd.to_numeric(obj.get("travel_time"), errors="coerce")
    travel_min = float(travel_sec) / 60.0 if pd.notna(travel_sec) else np.nan
    end_ts = start_ts + pd.to_timedelta(travel_min, unit="m") if pd.notna(start_ts) else pd.NaT
    return {
        "trip_id": _extract_oid(obj.get("_id")),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "start_station": obj.get("idunplug_station"),
        "end_station": obj.get("idplug_station"),
        "travel_time_min": travel_min,
        "user_type": obj.get("user_type"),
        "age_range": obj.get("ageRange"),
        "bike_id": pd.NA,
        "lock_type": "STATION",
        "user_day_code": obj.get("user_day_code"),
        "source_schema": schema,
        "source_file": source_file,
    }


def parse_trips_json_file(path: Path, schema: str, chunk_size: int = 80_000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    buf: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj.pop("track", None)
            buf.append(parse_trips_json_line(obj, schema, path.name))
            if len(buf) >= chunk_size:
                frames.append(pd.DataFrame(buf))
                buf = []
    if buf:
        frames.append(pd.DataFrame(buf))
    if not frames:
        return pd.DataFrame(columns=TRIP_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["start_station"] = _nullable_int(df["start_station"])
    df["end_station"] = _nullable_int(df["end_station"])
    df["user_type"] = _nullable_int(df["user_type"])
    df["age_range"] = _nullable_int(df["age_range"])
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True, errors="coerce")
    df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True, errors="coerce")
    return df


def parse_trips_csv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        skip_blank_lines=True,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    df = df.dropna(how="all")
    # rows that are only empty strings / separators
    nonempty = df.apply(lambda col: col.astype(str).str.strip().ne("") & col.astype(str).ne("nan")).any(axis=1)
    df = df.loc[nonempty].copy()
    start_ts = pd.to_datetime(df.get("unlock_date"), utc=True, errors="coerce")
    end_ts = pd.to_datetime(df.get("lock_date"), utc=True, errors="coerce")
    travel = pd.to_numeric(df.get("trip_minutes"), errors="coerce")
    bike = df["idBike"].astype("string") if "idBike" in df.columns else pd.Series("unk", index=df.index, dtype="string")
    stamp = start_ts.dt.strftime("%Y%m%d%H%M%S").astype("string")
    out = pd.DataFrame(
        {
            "trip_id": bike.fillna("unk") + "_" + stamp.fillna("na"),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_station": _nullable_int(df.get("station_unlock")),
            "end_station": _nullable_int(df.get("station_lock")),
            "travel_time_min": travel,
            "user_type": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "age_range": pd.Series(pd.NA, index=df.index, dtype="Int64"),
            "bike_id": _nullable_int(df.get("idBike")),
            "lock_type": df.get("locktype", pd.Series("STATION", index=df.index)).fillna("STATION"),
            "user_day_code": pd.Series(pd.NA, index=df.index, dtype="string"),
            "source_schema": "csv",
            "source_file": path.name,
        }
    )
    return out


def parse_trips_file(path: Path) -> pd.DataFrame:
    schema = trip_schema_for(path)
    if schema == "csv":
        return parse_trips_csv_file(path)
    return parse_trips_json_file(path, schema)


def trip_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Start and end times for trips that have both.

    The integers use the same unit as ``Series.astype("int64")`` on a UTC
    timestamp, so they compare directly with status snapshot times. The GPS
    ``track`` is dropped before the trip is kept. A trip with no end, or with
    an end at or before the start, is omitted. The two arrays are paired and
    unsorted.
    """
    schema = trip_schema_for(path)
    if schema == "csv":
        starts, ends = _csv_intervals(path)
    else:
        starts, ends = _json_intervals(path)
    if starts.size == 0:
        raise RuntimeError(f"{path.name} contained no trips with a start and an end")
    return starts, ends


def count_trips_in_progress(
    snapshot_ns: np.ndarray,
    start_ns: np.ndarray,
    end_ns: np.ndarray,
) -> np.ndarray:
    """Count trips with ``start <= t < end`` at each snapshot.

    Starts and ends are sorted once and scanned in time order. The result
    follows the order of ``snapshot_ns``.
    """
    snaps = np.asarray(snapshot_ns, dtype=np.int64)
    if snaps.size == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(snaps, kind="mergesort")
    ordered = snaps[order]
    starts = np.sort(np.asarray(start_ns, dtype=np.int64))
    ends = np.sort(np.asarray(end_ns, dtype=np.int64))
    started = np.searchsorted(starts, ordered, side="right")
    finished = np.searchsorted(ends, ordered, side="right")
    counted = started - finished
    if np.any(counted < 0):
        raise RuntimeError("in-progress trip count went negative")
    out = np.empty(snaps.shape, dtype=np.int64)
    out[order] = counted
    return out


def _csv_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = _csv_header_line(path)
    if _csv_has_interval_columns(header):
        frame = _read_interval_columns(path, sep=";")
    else:
        stripped = header.replace(";", "")
        if not _csv_has_interval_columns(stripped):
            raise RuntimeError(f"{path.name} is missing columns: ['lock_date', 'unlock_date']")
        return _intervals_from_damaged_csv(path)
    start = pd.to_datetime(frame["unlock_date"], utc=True, errors="coerce")
    end = pd.to_datetime(frame["lock_date"], utc=True, errors="coerce")
    ok = start.notna() & end.notna() & end.gt(start)
    return (
        start.loc[ok].astype("int64").to_numpy(),
        end.loc[ok].astype("int64").to_numpy(),
    )


_ISO_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _intervals_from_damaged_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """October 2021 is published with semicolons and quotes inside fields.

    After the inserted semicolons are removed, each intact row carries three
    timestamps: the lock time inside the trip id, then ``unlock_date``, then
    ``lock_date``. The last two are the interval.
    """
    text = path.read_text(encoding="utf-8", errors="replace").replace(";", "")
    unlocks: list[str] = []
    locks: list[str] = []
    for line in text.splitlines():
        found = _ISO_STAMP.findall(line)
        if len(found) < 2:
            continue
        unlocks.append(found[-2])
        locks.append(found[-1])
    start = pd.to_datetime(pd.Series(unlocks), utc=True, errors="coerce")
    end = pd.to_datetime(pd.Series(locks), utc=True, errors="coerce")
    ok = start.notna() & end.notna() & end.gt(start)
    return (
        start[ok].astype("int64").to_numpy(),
        end[ok].astype("int64").to_numpy(),
    )


def _csv_header_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readline()


def _csv_has_interval_columns(header: str) -> bool:
    return "unlock_date" in header and "lock_date" in header


def _read_interval_columns(path: Path, sep: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=sep,
        usecols=["unlock_date", "lock_date"],
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )


def _json_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stamps: list[str] = []
    seconds: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            try:
                obj = _loads_trip_line(line) if '"track"' in line else json.loads(line)
            except (json.JSONDecodeError, RuntimeError) as exc:
                raise RuntimeError(f"{path.name} line {line_no} is not a trip record") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path.name} line {line_no} is not a trip record")
            raw_time = obj.get("unplug_hourTime")
            if isinstance(raw_time, dict):
                raw_time = raw_time.get("$date")
            if raw_time is None:
                continue
            travel = obj.get("travel_time")
            try:
                duration = float(travel)
            except (TypeError, ValueError):
                continue
            if not duration > 0:
                continue
            stamps.append(str(raw_time))
            seconds.append(duration)
            if len(stamps) % 250_000 == 0:
                print(f"  {path.name}: {len(stamps):,} trips", flush=True)
    if not stamps:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    normalized = pd.Series(stamps).str.replace(
        r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True
    )
    start = pd.to_datetime(normalized, utc=True, errors="coerce")
    duration_s = np.asarray(seconds, dtype=np.float64)
    ok = start.notna().to_numpy()
    start_ns = start.loc[ok].astype("int64").to_numpy()
    end_ns = start_ns + np.round(duration_s[ok] * _datetime_int_scale(start.dtype)).astype(
        np.int64
    )
    return start_ns, end_ns


def _datetime_int_scale(dtype) -> int:
    """Seconds-to-integer factor for the unit of a pandas datetime dtype."""
    unit = getattr(dtype, "unit", "ns")
    scale = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}
    try:
        return scale[unit]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported datetime unit {unit}") from exc


def _loads_trip_line(line: str) -> dict:
    """Parse one trip, dropping the GPS track before it is kept."""
    payload = line
    if '"track"' in line:
        try:
            payload = _drop_json_field(line, "track")
        except RuntimeError:
            payload = line
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        if payload is line:
            raise
        obj = json.loads(line)
    if isinstance(obj, dict):
        obj.pop("track", None)
    return obj


def _drop_json_field(text: str, field: str) -> str:
    """Remove the first JSON field named ``field``, including its value."""
    needle = f'"{field}"'
    start = text.find(needle)
    if start < 0:
        return text
    index = start + len(needle)
    length = len(text)
    while index < length and text[index].isspace():
        index += 1
    if index >= length or text[index] != ":":
        return text
    index += 1
    while index < length and text[index].isspace():
        index += 1
    value_end = _skip_json_value(text, index)
    right = value_end
    cursor = value_end
    while cursor < length and text[cursor].isspace():
        cursor += 1
    if cursor < length and text[cursor] == ",":
        right = cursor + 1
        left = start
    else:
        left = start
        comma = start - 1
        while comma >= 0 and text[comma].isspace():
            comma -= 1
        if comma >= 0 and text[comma] == ",":
            left = comma
    return text[:left] + text[right:]


def _skip_json_value(text: str, index: int) -> int:
    length = len(text)
    if index >= length:
        raise RuntimeError("JSON value is missing")
    char = text[index]
    if char == '"':
        return _skip_json_string(text, index)
    if char in "{[":
        depth = 0
        while index < length:
            current = text[index]
            if current == '"':
                index = _skip_json_string(text, index)
                continue
            if current in "{[":
                depth += 1
            elif current in "}]":
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        raise RuntimeError("unterminated JSON value")
    while index < length and text[index] not in ",}]":
        index += 1
    return index


def _skip_json_string(text: str, index: int) -> int:
    index += 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    raise RuntimeError("unterminated JSON string")


def parse_status_file(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _to_utc_datetime(obj.get("_id"))
            for st in obj.get("stations") or []:
                rows.append(
                    {
                        "ts": ts,
                        "station_id": st.get("id"),
                        "name": st.get("name"),
                        "number": st.get("number"),
                        "address": st.get("address"),
                        "lat": st.get("latitude"),
                        "lon": st.get("longitude"),
                        "total_bases": st.get("total_bases"),
                        "dock_bikes": st.get("dock_bikes"),
                        "free_bases": st.get("free_bases"),
                        "activate": st.get("activate"),
                        "no_available": st.get("no_available"),
                        "reservations_count": st.get("reservations_count"),
                        "light": st.get("light"),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    df = pd.DataFrame(rows)
    df["station_id"] = _nullable_int(df["station_id"])
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in (
        "total_bases",
        "dock_bikes",
        "free_bases",
        "activate",
        "no_available",
        "reservations_count",
        "light",
    ):
        df[col] = _nullable_int(df[col])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def load_trips_for_months(cfg: ProjectConfig, months: list[str]) -> pd.DataFrame:
    files = discover_trip_files(cfg)
    frames = [parse_trips_file(files[m]) for m in months if m in files]
    if not frames:
        return pd.DataFrame(columns=TRIP_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def load_status_for_months(cfg: ProjectConfig, months: list[str]) -> pd.DataFrame:
    files = discover_status_files(cfg)
    frames = [parse_status_file(files[m]) for m in months if m in files]
    if not frames:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_station_catalog(status: pd.DataFrame) -> pd.DataFrame:
    """Latest known metadata per station_id (name, coords, capacity)."""
    if status.empty:
        return pd.DataFrame(
            columns=["station_id", "name", "number", "address", "lat", "lon", "total_bases"]
        )
    ordered = status.sort_values("ts")
    last = ordered.groupby("station_id", as_index=False).last()
    catalog = last[["station_id", "name", "number", "address", "lat", "lon", "total_bases"]].copy()
    first_seen = ordered.groupby("station_id")["ts"].min().rename("first_seen")
    last_seen = ordered.groupby("station_id")["ts"].max().rename("last_seen")
    n_snap = ordered.groupby("station_id").size().rename("n_snapshots")
    catalog = catalog.merge(first_seen, on="station_id").merge(last_seen, on="station_id")
    catalog = catalog.merge(n_snap, on="station_id")
    return catalog


def to_madrid(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts.dt.tz_convert(MADRID_TZ)


def hourly_bin(ts: pd.Series) -> pd.Series:
    """Floor to the local hour, surviving the October DST fallback."""
    madrid = to_madrid(ts)
    naive = madrid.dt.tz_localize(None)
    floored = naive.dt.floor("h")
    return floored.dt.tz_localize(MADRID_TZ, ambiguous=True, nonexistent="shift_forward")


_COUNT_CACHE = "snapshot_station_counts.parquet"
_COUNT_MANIFEST = "snapshot_station_counts.manifest.json"


def snapshot_station_counts(path: Path) -> pd.DataFrame:
    """One row per status snapshot in ``path``.

    ``n_stations`` is the number of distinct station ids listed in the snapshot.
    ``n_active`` counts those with ``activate == 1``. When a snapshot never
    carries an ``activate`` field, ``n_active`` is missing for that row.
    """
    rows: list[dict] = []
    for obj in _iter_status_snapshots(path):
        n_stations, n_active = _count_listed_stations(obj.get("stations"), path)
        rows.append(
            {
                "ts": _to_utc_datetime(obj.get("_id")),
                "n_stations": n_stations,
                "n_active": n_active,
            }
        )
    if not rows:
        raise RuntimeError(f"{path.name} contained no station snapshots")
    frame = pd.DataFrame(rows)
    frame["n_stations"] = frame["n_stations"].astype("Int64")
    frame["n_active"] = pd.array(frame["n_active"], dtype="Int64")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    missing_ts = int(frame["ts"].isna().sum())
    if missing_ts:
        raise RuntimeError(f"{path.name} has {missing_ts} snapshots without a timestamp")
    return frame


def snapshot_docked_bikes(path: Path) -> pd.DataFrame:
    """One row per status snapshot in ``path``.

    ``n_docked`` sums ``dock_bikes`` over every listed station, including
    stations marked unavailable. A missing or non-numeric ``dock_bikes``
    counts as zero.
    """
    frame = _snapshot_station_sum(path, "dock_bikes", "n_docked")
    return frame


def snapshot_dock_capacity(path: Path) -> pd.DataFrame:
    """One row per status snapshot in ``path``.

    ``n_docks`` sums ``total_bases`` over every listed station, including
    stations marked unavailable. A missing or non-numeric ``total_bases``
    counts as zero.
    """
    return _snapshot_station_sum(path, "total_bases", "n_docks")


def _snapshot_station_sum(path: Path, field: str, column: str) -> pd.DataFrame:
    rows: list[dict] = []
    for obj in _iter_status_snapshots(path):
        rows.append(
            {
                "ts": _to_utc_datetime(obj.get("_id")),
                column: _sum_station_int(obj.get("stations"), path, field),
            }
        )
    if not rows:
        raise RuntimeError(f"{path.name} contained no station snapshots")
    frame = pd.DataFrame(rows)
    frame[column] = frame[column].astype("int64")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    missing_ts = int(frame["ts"].isna().sum())
    if missing_ts:
        raise RuntimeError(f"{path.name} has {missing_ts} snapshots without a timestamp")
    return frame


def _sum_station_int(stations, path: Path, field: str) -> int:
    if not isinstance(stations, list):
        raise RuntimeError(f"{path.name} has a snapshot whose stations field is not a list")
    total = 0
    for station in stations:
        if isinstance(station, dict):
            total += _dock_bikes_value(station.get(field))
    return total


def _dock_bikes_value(value) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return 0
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            number = float(text)
        except ValueError:
            return 0
        if number != number:
            return 0
        return int(number)
    return 0


def load_snapshot_station_counts(cfg: ProjectConfig) -> pd.DataFrame:
    """Station counts for every status file under ``cfg``.

    The result is cached at ``data/processed/infrastructure``. A later call
    reuses the cache when the status filenames and sizes are unchanged, and
    rescans otherwise.
    """
    folder = cfg.processed_dir / "infrastructure"
    cache = folder / _COUNT_CACHE
    manifest_path = folder / _COUNT_MANIFEST
    manifest = _status_manifest(cfg)
    if not manifest:
        raise RuntimeError(f"No status files in {cfg.status_raw}")
    if cache.is_file() and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved == manifest:
            return pd.read_parquet(cache)
    files = discover_status_files(cfg)
    frames: list[pd.DataFrame] = []
    for month, path in files.items():
        print(f"counting stations in {path.name}", flush=True)
        frame = snapshot_station_counts(path)
        frame["month"] = month
        frame["source_file"] = path.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    folder.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def _status_manifest(cfg: ProjectConfig) -> list[dict]:
    return [
        {"month": month, "file": path.name, "bytes": path.stat().st_size}
        for month, path in discover_status_files(cfg).items()
    ]


_MONGO_NUMBER = re.compile(r"Number(?:Int|Long)\(\s*(-?\d+)\s*\)")
_MONGO_NUMBER_STR = re.compile(r'NumberLong\(\s*"(-?\d+)"\s*\)')


def _iter_status_snapshots(path: Path):
    """Yield snapshot objects from NDJSON or concatenated JSON documents.

    Files through mid-2019 are pretty-printed Mongo shell exports: several
    documents in one file, with ``NumberInt`` / ``NumberLong`` wrappers.
    Later months are one JSON object per line.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = ""
        while True:
            line = fh.readline()
            if line == "":
                raise RuntimeError(f"{path.name} is empty")
            if line.strip():
                first = line
                break
        try:
            obj = json.loads(first)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "stations" in obj:
            yield obj
            for line_no, line in enumerate(fh, start=2):
                line = line.strip()
                if not line:
                    continue
                try:
                    nxt = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path.name} line {line_no} is not a JSON snapshot") from exc
                if not isinstance(nxt, dict) or "stations" not in nxt:
                    raise RuntimeError(f"{path.name} line {line_no} is not a station snapshot")
                yield nxt
            return
    yield from _iter_concatenated_snapshots(path)


def _iter_concatenated_snapshots(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    text = _MONGO_NUMBER_STR.sub(r"\1", text)
    text = _MONGO_NUMBER.sub(r"\1", text)
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    found = 0
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{path.name} could not be parsed as station snapshots at character {index}"
            ) from exc
        index = end
        found += 1
        if isinstance(obj, dict) and "stations" in obj:
            yield obj
            continue
        if (
            isinstance(obj, list)
            and obj
            and all(isinstance(item, dict) and "id" in item for item in obj)
        ):
            yield {"_id": None, "stations": obj}
            continue
        raise RuntimeError(f"{path.name} value {found} is not a station snapshot")
    if found == 0:
        raise RuntimeError(f"{path.name} contained no station snapshots")


def _count_listed_stations(stations, path: Path) -> tuple[int, int | None]:
    if not isinstance(stations, list):
        raise RuntimeError(f"{path.name} has a snapshot whose stations field is not a list")
    by_id: dict[object, bool | None] = {}
    saw_activate = False
    for station in stations:
        if not isinstance(station, dict):
            continue
        station_id = station.get("id")
        if station_id is None:
            continue
        flag: bool | None = None
        if "activate" in station:
            saw_activate = True
            flag = _activate_flag(station.get("activate"))
        by_id[station_id] = flag
    n_active = sum(1 for flag in by_id.values() if flag) if saw_activate else None
    return len(by_id), n_active


def _activate_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip() == "1"
    return False
