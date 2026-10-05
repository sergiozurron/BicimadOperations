"""Data-quality audit and cleaning rules with sample-size impact logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CAPACITY_TOLERANCE,
    MADRID_BBOX,
    MAX_TRIP_MINUTES,
    MIN_TRIP_MINUTES,
)


def _pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 3)


def trip_quality_report(trips: pd.DataFrame) -> pd.DataFrame:
    n = len(trips)
    missing_start = int(trips["start_ts"].isna().sum()) if n else 0
    missing_origin = int(trips["start_station"].isna().sum()) if n else 0
    missing_dest = int(trips["end_station"].isna().sum()) if n else 0
    missing_tt = int(trips["travel_time_min"].isna().sum()) if n else 0
    tt = pd.to_numeric(trips.get("travel_time_min"), errors="coerce") if n else pd.Series(dtype=float)
    too_short = int((tt < MIN_TRIP_MINUTES).sum()) if n else 0
    too_long = int((tt > MAX_TRIP_MINUTES).sum()) if n else 0
    same_station = int(
        (trips["start_station"] == trips["end_station"]).fillna(False).sum()
    ) if n else 0
    dockless = int((trips.get("lock_type", "STATION") != "STATION").fillna(False).sum()) if n else 0
    dup_cols = [c for c in ("user_day_code", "start_ts", "start_station", "end_station") if c in trips]
    dups = int(trips.duplicated(dup_cols).sum()) if n and dup_cols else 0
    if n and "bike_id" in trips.columns and trips["bike_id"].notna().any():
        dups = int(trips.duplicated(["bike_id", "start_ts"]).sum())
    rows = [
        ("completeness", "missing start_ts", missing_start, _pct(missing_start, n)),
        ("completeness", "missing start_station", missing_origin, _pct(missing_origin, n)),
        ("completeness", "missing end_station", missing_dest, _pct(missing_dest, n)),
        ("completeness", "missing travel_time_min", missing_tt, _pct(missing_tt, n)),
        ("validity", f"travel_time < {MIN_TRIP_MINUTES} min", too_short, _pct(too_short, n)),
        ("validity", f"travel_time > {MAX_TRIP_MINUTES} min", too_long, _pct(too_long, n)),
        ("consistency", "same origin and destination", same_station, _pct(same_station, n)),
        ("coverage", "non-station (dockless/FREE) trips", dockless, _pct(dockless, n)),
        ("uniqueness", "duplicate trip keys", dups, _pct(dups, n)),
    ]
    return pd.DataFrame(rows, columns=["dimension", "check", "n_flagged", "pct"])


def status_quality_report(status: pd.DataFrame) -> pd.DataFrame:
    n = len(status)
    if n == 0:
        return pd.DataFrame(columns=["dimension", "check", "n_flagged", "pct"])
    missing_ts = int(status["ts"].isna().sum())
    missing_id = int(status["station_id"].isna().sum())
    missing_bikes = int(status["dock_bikes"].isna().sum())
    cap = pd.to_numeric(status["total_bases"], errors="coerce")
    bikes = pd.to_numeric(status["dock_bikes"], errors="coerce")
    free = pd.to_numeric(status["free_bases"], errors="coerce")
    occ_over = int((bikes > cap).fillna(False).sum())
    occ_neg = int((bikes < 0).fillna(False).sum())
    sum_over = int((bikes + free > cap + CAPACITY_TOLERANCE).fillna(False).sum())
    lat = pd.to_numeric(status["lat"], errors="coerce")
    lon = pd.to_numeric(status["lon"], errors="coerce")
    outside = int(
        (
            (lat < MADRID_BBOX["lat_min"])
            | (lat > MADRID_BBOX["lat_max"])
            | (lon < MADRID_BBOX["lon_min"])
            | (lon > MADRID_BBOX["lon_max"])
        )
        .fillna(False)
        .sum()
    )
    inactive = int((status["activate"] == 0).fillna(False).sum())
    empty = int((bikes == 0).fillna(False).sum())
    full = int((free == 0).fillna(False).sum())
    rows = [
        ("completeness", "missing ts", missing_ts, _pct(missing_ts, n)),
        ("completeness", "missing station_id", missing_id, _pct(missing_id, n)),
        ("completeness", "missing dock_bikes", missing_bikes, _pct(missing_bikes, n)),
        ("validity", "dock_bikes > total_bases", occ_over, _pct(occ_over, n)),
        ("validity", "dock_bikes < 0", occ_neg, _pct(occ_neg, n)),
        ("consistency", "bikes+free > capacity + tolerance", sum_over, _pct(sum_over, n)),
        ("validity", "coordinates outside Madrid bbox", outside, _pct(outside, n)),
        ("coverage", "inactive stations (activate=0)", inactive, _pct(inactive, n)),
        ("accuracy", "empty snapshots (dock_bikes=0)", empty, _pct(empty, n)),
        ("accuracy", "full snapshots (free_bases=0)", full, _pct(full, n)),
    ]
    return pd.DataFrame(rows, columns=["dimension", "check", "n_flagged", "pct"])


def clean_trips(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned station-to-station trips, drop log)."""
    log = []
    n0 = len(trips)
    df = trips.copy()
    log.append(("input", n0, 0))

    mask = df["start_ts"].notna() & df["travel_time_min"].notna()
    log.append(("drop missing timestamp or duration", len(df), int((~mask).sum())))
    df = df.loc[mask]

    tt = pd.to_numeric(df["travel_time_min"], errors="coerce")
    mask = (tt >= MIN_TRIP_MINUTES) & (tt <= MAX_TRIP_MINUTES)
    log.append(("drop travel_time outside [1 min, 4 h]", len(df), int((~mask).sum())))
    df = df.loc[mask]

    # keep dockless aside: station-level modelling needs both stations
    station_ok = df["start_station"].notna() & df["end_station"].notna()
    if "lock_type" in df.columns:
        station_ok = station_ok & df["lock_type"].fillna("STATION").eq("STATION")
    log.append(("drop dockless / missing station ids", len(df), int((~station_ok).sum())))
    df = df.loc[station_ok]

    same = df["start_station"] == df["end_station"]
    short_loop = same & (pd.to_numeric(df["travel_time_min"], errors="coerce") < 3)
    log.append(("drop near-zero same-station loops (<3 min)", len(df), int(short_loop.sum())))
    df = df.loc[~short_loop]

    subset = ["start_ts", "start_station", "end_station", "travel_time_min"]
    dups = df.duplicated(subset)
    log.append(("drop exact duplicates", len(df), int(dups.sum())))
    df = df.loc[~dups]

    log.append(("output", len(df), n0 - len(df)))
    impact = pd.DataFrame(log, columns=["rule", "n_before_or_current", "n_dropped"])
    impact["n_remaining"] = [n0] + list(impact["n_before_or_current"].iloc[1:].values)
    # simpler remaining column
    remaining = [n0]
    dropped_list = impact["n_dropped"].tolist()
    rem = n0
    remaining = []
    for i, d in enumerate(dropped_list):
        if i == 0:
            remaining.append(n0)
        else:
            rem = rem - d
            remaining.append(rem)
    impact["n_remaining"] = remaining
    impact["pct_of_input_dropped_cum"] = [
        round(100.0 * (n0 - r) / n0, 3) if n0 else 0.0 for r in remaining
    ]
    return df.reset_index(drop=True), impact


def clean_status(status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = []
    n0 = len(status)
    df = status.copy()
    log.append(("input", n0, 0))

    mask = df["ts"].notna() & df["station_id"].notna()
    log.append(("drop missing ts or station_id", len(df), int((~mask).sum())))
    df = df.loc[mask]

    for col in ("total_bases", "dock_bikes", "free_bases", "lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = df["total_bases"].fillna(0) > 0
    log.append(("drop zero/missing capacity", len(df), int((~mask).sum())))
    df = df.loc[mask]

    over = df["dock_bikes"] > df["total_bases"]
    n_clip = int(over.fillna(False).sum())
    df.loc[over, "dock_bikes"] = df.loc[over, "total_bases"]
    log.append(("clip dock_bikes to capacity", len(df), n_clip))

    df["dock_bikes"] = df["dock_bikes"].clip(lower=0)
    df["free_bases"] = df["free_bases"].clip(lower=0)

    lat_ok = df["lat"].between(MADRID_BBOX["lat_min"], MADRID_BBOX["lat_max"])
    lon_ok = df["lon"].between(MADRID_BBOX["lon_min"], MADRID_BBOX["lon_max"])
    geo_ok = lat_ok & lon_ok
    log.append(("drop coordinates outside Madrid bbox", len(df), int((~geo_ok.fillna(False)).sum())))
    df = df.loc[geo_ok.fillna(False)]

    remaining, rem = [], n0
    dropped = [r[2] for r in log]
    for i, d in enumerate(dropped):
        if i == 0:
            remaining.append(n0)
        else:
            rem = rem - (d if "clip" not in log[i][0] else 0)
            remaining.append(rem)
    impact = pd.DataFrame(log, columns=["rule", "n_before_or_current", "n_dropped"])
    impact["n_remaining"] = remaining
    return df.reset_index(drop=True), impact


def gini_coefficient(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def univariate_profile(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Series({"variable": name, "n": 0})
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return pd.Series(
        {
            "variable": name,
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p01": float(q.loc[0.01]),
            "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "median": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "max": float(s.max()),
            "skew": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
        }
    )
