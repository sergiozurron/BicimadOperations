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
