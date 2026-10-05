"""Mass-event features from `data/raw/mass_events/madrid_mass_events.csv`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MADRID_TZ, ProjectConfig

PRE_HOURS = 2
POST_HOURS = 2
RADIUS_KM = 2.0
VERY_NEAR_KM = 0.8

TYPE_GROUP = {
    "Football match": "football",
    "Football final": "football",
    "Rugby match": "football",
    "Concert": "concert",
    "Multi-artist concert": "concert",
    "Trade fair": "fair",
    "Art fair": "fair",
    "Political rally": "other",
    "Sports celebration": "other",
    "Religious assembly": "other",
    "Motorsport / stunt show": "other",
}

EVENT_FEATURE_COLUMNS = [
    "evt_n",
    "evt_attendance",
    "evt_attendance_log",
    "evt_min_dist_km",
    "evt_nearby",
    "evt_very_near",
    "evt_pre",
    "evt_during",
    "evt_post",
    "evt_football",
    "evt_concert",
    "evt_fair",
    "evt_other",
    "evt_city_n",
    "evt_city_attendance",
]


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _parse_attendance(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace('"', "").replace(",", "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_mass_events(cfg: ProjectConfig) -> pd.DataFrame:
    path = cfg.raw_dir / "mass_events" / "madrid_mass_events.csv"
    if not path.exists():
        return pd.DataFrame()
    last_err = None
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
            df = None
    if df is None:
        raise last_err or FileNotFoundError(path)
    df.columns = [c.strip() for c in df.columns]
    df["start"] = pd.to_datetime(df["start_datetime_local"], errors="coerce")
    df["end"] = pd.to_datetime(df["end_datetime_local"], errors="coerce")
    df["attendance_num"] = df["attendance"].map(_parse_attendance)
    venue_med = df.groupby("venue_group")["attendance_num"].median()
    df["attendance_filled"] = df["attendance_num"].fillna(df["venue_group"].map(venue_med)).fillna(0.0)
    df["type_group"] = df["event_type"].map(TYPE_GROUP).fillna("other")
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["start", "end", "lat", "lon"])
    return df


def _localize_hour_index(start, end) -> pd.DatetimeIndex:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize(MADRID_TZ, ambiguous=True, nonexistent="shift_forward")
        end = end.tz_localize(MADRID_TZ, ambiguous=True, nonexistent="shift_forward")
    else:
        start = start.tz_convert(MADRID_TZ)
        end = end.tz_convert(MADRID_TZ)
    t0 = (start - pd.Timedelta(hours=PRE_HOURS)).floor("h")
    t1 = end.floor("h") + pd.Timedelta(hours=POST_HOURS)
    if t1 < t0:
        t1 = t0
    return pd.date_range(t0, t1, freq="h", tz=MADRID_TZ)


def expand_event_hours(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, local hour) covering pre / during / post buffers."""
    rows: list[dict] = []
    for ev in events.itertuples(index=False):
        hours = _localize_hour_index(ev.start, ev.end)
        raw_start = pd.Timestamp(ev.start)
        raw_end = pd.Timestamp(ev.end)
        if raw_start.tzinfo is None:
            raw_start = raw_start.tz_localize(MADRID_TZ, ambiguous=True, nonexistent="shift_forward")
            raw_end = raw_end.tz_localize(MADRID_TZ, ambiguous=True, nonexistent="shift_forward")
        for h in hours:
            hour_end = h + pd.Timedelta(hours=1)
            if hour_end <= raw_start:
                phase = "pre"
            elif h >= raw_end:
                phase = "post"
            else:
                phase = "during"
            rows.append(
                {
                    "event_id": ev.event_id,
                    "hour": h,
                    "phase": phase,
                    "type_group": ev.type_group,
                    "attendance": float(ev.attendance_filled),
                    "lat": float(ev.lat),
                    "lon": float(ev.lon),
                    "venue_group": ev.venue_group,
                    "event_name": ev.event_name,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["event_id", "hour", "phase", "type_group", "attendance", "lat", "lon", "venue_group", "event_name"]
        )
    return pd.DataFrame(rows)


def add_event_features(
    table: pd.DataFrame,
    cfg: ProjectConfig,
    radius_km: float = RADIUS_KM,
    very_near_km: float = VERY_NEAR_KM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join proximity event features onto the station-hour table.

    Returns (table, event_hour panel used for the join).
    """
    events = load_mass_events(cfg)
    out = table.copy()
    empty = {c: 0.0 for c in EVENT_FEATURE_COLUMNS}
    empty["evt_min_dist_km"] = 99.0
    if events.empty or out.empty:
        for c, v in empty.items():
            if c not in out.columns:
                out[c] = v
        return out, pd.DataFrame()

    eh = expand_event_hours(events)
    tmin, tmax = out["hour"].min(), out["hour"].max()
    eh = eh[(eh["hour"] >= tmin) & (eh["hour"] <= tmax)].copy()

    city = (
        eh.groupby("hour", as_index=False)
        .agg(evt_city_n=("event_id", "nunique"), evt_city_attendance=("attendance", "sum"))
    )

    stations = out[["station_id", "lat", "lon"]].drop_duplicates()
    stations = stations.dropna(subset=["lat", "lon"])
    if eh.empty or stations.empty:
        for c, v in empty.items():
            if c not in out.columns:
                out[c] = v
        out = out.drop(columns=[c for c in ("evt_city_n", "evt_city_attendance") if c in out.columns], errors="ignore")
        out = out.merge(city, on="hour", how="left")
        out["evt_city_n"] = out["evt_city_n"].fillna(0)
        out["evt_city_attendance"] = out["evt_city_attendance"].fillna(0)
        return out, eh

    st = stations.rename(columns={"lat": "s_lat", "lon": "s_lon"})
    cross = st.merge(eh, how="cross")
    cross["dist_km"] = _haversine_km(
        cross["s_lat"].to_numpy(),
        cross["s_lon"].to_numpy(),
        cross["lat"].to_numpy(),
        cross["lon"].to_numpy(),
    )
    near = cross.loc[cross["dist_km"] <= radius_km].copy()
    if near.empty:
        local = pd.DataFrame(columns=["station_id", "hour"] + [c for c in EVENT_FEATURE_COLUMNS if not c.startswith("evt_city")])
    else:
        near["is_pre"] = (near["phase"] == "pre").astype(int)
        near["is_during"] = (near["phase"] == "during").astype(int)
        near["is_post"] = (near["phase"] == "post").astype(int)
        near["is_football"] = (near["type_group"] == "football").astype(int)
        near["is_concert"] = (near["type_group"] == "concert").astype(int)
        near["is_fair"] = (near["type_group"] == "fair").astype(int)
        near["is_other"] = (near["type_group"] == "other").astype(int)
        near["is_very_near"] = (near["dist_km"] <= very_near_km).astype(int)
        local = (
            near.groupby(["station_id", "hour"], as_index=False)
            .agg(
                evt_n=("event_id", "nunique"),
                evt_attendance=("attendance", "sum"),
                evt_min_dist_km=("dist_km", "min"),
                evt_very_near=("is_very_near", "max"),
                evt_pre=("is_pre", "max"),
                evt_during=("is_during", "max"),
                evt_post=("is_post", "max"),
                evt_football=("is_football", "max"),
                evt_concert=("is_concert", "max"),
                evt_fair=("is_fair", "max"),
                evt_other=("is_other", "max"),
            )
        )
        local["evt_nearby"] = (local["evt_n"] > 0).astype(int)
        local["evt_attendance_log"] = np.log1p(local["evt_attendance"])

    drop_existing = [c for c in EVENT_FEATURE_COLUMNS if c in out.columns]
    if drop_existing:
        out = out.drop(columns=drop_existing)
    out = out.merge(local, on=["station_id", "hour"], how="left")
    out = out.merge(city, on="hour", how="left")

    out["evt_n"] = out["evt_n"].fillna(0)
    out["evt_attendance"] = out["evt_attendance"].fillna(0)
    out["evt_attendance_log"] = out["evt_attendance_log"].fillna(0)
    out["evt_min_dist_km"] = out["evt_min_dist_km"].fillna(99.0)
    for c in (
        "evt_nearby",
        "evt_very_near",
        "evt_pre",
        "evt_during",
        "evt_post",
        "evt_football",
        "evt_concert",
        "evt_fair",
        "evt_other",
        "evt_city_n",
        "evt_city_attendance",
    ):
        out[c] = out[c].fillna(0)
    return out, eh
