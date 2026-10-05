"""District-level hourly weather for Madrid (Open-Meteo ERA5 at district centroids)."""

from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MADRID_TZ, ProjectConfig
from .ingest import hourly_bin

# Official 21 distritos of the City of Madrid. Centroids are representative
# points used both to assign BiciMAD stations and to query weather.
MADRID_DISTRICTS: dict[int, tuple[str, float, float]] = {
    1: ("Centro", 40.4153, -3.7074),
    2: ("Arganzuela", 40.3987, -3.6938),
    3: ("Retiro", 40.4114, -3.6766),
    4: ("Salamanca", 40.4300, -3.6778),
    5: ("Chamartín", 40.4584, -3.6766),
    6: ("Tetuán", 40.4606, -3.6984),
    7: ("Chamberí", 40.4362, -3.7038),
    8: ("Fuencarral-El Pardo", 40.4980, -3.7314),
    9: ("Moncloa-Aravaca", 40.4350, -3.7450),
    10: ("Latina", 40.3889, -3.7456),
    11: ("Carabanchel", 40.3837, -3.7282),
    12: ("Usera", 40.3815, -3.7060),
    13: ("Puente de Vallecas", 40.3870, -3.6570),
    14: ("Moratalaz", 40.4070, -3.6450),
    15: ("Ciudad Lineal", 40.4500, -3.6510),
    16: ("Hortaleza", 40.4724, -3.6406),
    17: ("Villaverde", 40.3450, -3.7090),
    18: ("Villa de Vallecas", 40.3670, -3.6210),
    19: ("Vicálvaro", 40.4040, -3.6080),
    20: ("San Blas-Canillejas", 40.4360, -3.6150),
    21: ("Barajas", 40.4733, -3.5778),
}

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = (
    "temperature_2m",
    "precipitation",
    "rain",
    "windspeed_10m",
    "relativehumidity_2m",
    "cloudcover",
)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def district_frame() -> pd.DataFrame:
    rows = [
        {"district_id": i, "district_name": name, "district_lat": lat, "district_lon": lon}
        for i, (name, lat, lon) in MADRID_DISTRICTS.items()
    ]
    return pd.DataFrame(rows)


def assign_districts(stations: pd.DataFrame) -> pd.DataFrame:
    """Nearest-centroid assignment of stations to the 21 Madrid districts."""
    out = stations.copy()
    districts = district_frame()
    if out.empty or "lat" not in out.columns:
        out["district_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["district_name"] = pd.NA
        return out
    dlat = districts["district_lat"].to_numpy()[:, None]
    dlon = districts["district_lon"].to_numpy()[:, None]
    slat = out["lat"].to_numpy(dtype=float)[None, :]
    slon = out["lon"].to_numpy(dtype=float)[None, :]
    dist = _haversine_km(dlat, dlon, slat, slon)
    nearest = np.nanargmin(np.where(np.isfinite(slat), dist, np.inf), axis=0)
    missing = ~np.isfinite(out["lat"].to_numpy(dtype=float))
    ids = districts["district_id"].to_numpy()[nearest]
    names = districts["district_name"].to_numpy()[nearest]
    out["district_id"] = pd.Series(ids, index=out.index, dtype="Int64")
    out["district_name"] = names
    out.loc[missing, "district_id"] = pd.NA
    out.loc[missing, "district_name"] = pd.NA
    return out


def _http_get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "BiciMAD-TFG/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except ssl.SSLError:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        if "CERTIFICATE" in str(exc).upper() or "SSL" in str(exc).upper():
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        raise


def _fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Madrid",
    }
    url = OPEN_METEO_ARCHIVE + "?" + urllib.parse.urlencode(params)
    payload = _http_get_json(url)
    hourly = payload.get("hourly") or {}
    if not hourly or "time" not in hourly:
        raise RuntimeError(f"Open-Meteo returned no hourly data for {lat},{lon} {start}..{end}")
    df = pd.DataFrame(hourly)
    df = df.rename(
        columns={
            "temperature_2m": "wx_temp_c",
            "precipitation": "wx_precip_mm",
            "rain": "wx_rain_mm",
            "windspeed_10m": "wx_wind_kmh",
            "relativehumidity_2m": "wx_humidity_pct",
            "cloudcover": "wx_cloud_pct",
        }
    )
    ts = pd.to_datetime(df["time"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(MADRID_TZ, ambiguous=True, nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(MADRID_TZ)
    df["hour"] = ts
    return df.drop(columns=["time"])


def weather_cache_dir(cfg: ProjectConfig) -> Path:
    d = cfg.processed_dir / "weather"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_district_weather(
    cfg: ProjectConfig,
    district_ids: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Hourly weather at each requested district centroid, cached as parquet."""
    start_d = pd.Timestamp(start).tz_convert(MADRID_TZ).date() if getattr(start, "tzinfo", None) else pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).tz_convert(MADRID_TZ).date() if getattr(end, "tzinfo", None) else pd.Timestamp(end).date()
    start_s, end_s = start_d.isoformat(), end_d.isoformat()
    frames = []
    cache = weather_cache_dir(cfg)
    for did in sorted(set(int(i) for i in district_ids if pd.notna(i))):
        if did not in MADRID_DISTRICTS:
            continue
        path = cache / f"openmeteo_d{did:02d}_{start_s}_{end_s}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        name, lat, lon = MADRID_DISTRICTS[did]
        raw = _fetch_open_meteo(lat, lon, start_s, end_s)
        raw["district_id"] = did
        raw["district_name"] = name
        raw.to_parquet(path, index=False)
        frames.append(raw)
        time.sleep(0.15)
    if not frames:
        return pd.DataFrame()
    wx = pd.concat(frames, ignore_index=True)
    wx["district_id"] = wx["district_id"].astype("Int64")
    return wx


def _add_weather_lags(wx: pd.DataFrame) -> pd.DataFrame:
    df = wx.sort_values(["district_id", "hour"]).copy()
    g = df.groupby("district_id", sort=False)
    df["wx_temp_lag_1h"] = g["wx_temp_c"].shift(1)
    df["wx_precip_lag_1h"] = g["wx_precip_mm"].shift(1)
    lagged_p = g["wx_precip_mm"].shift(1)
    df["wx_precip_3h"] = (
        lagged_p.groupby(df["district_id"]).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df["wx_precip_24h"] = (
        lagged_p.groupby(df["district_id"]).rolling(24, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df["wx_is_rain"] = (df["wx_precip_mm"].fillna(0) >= 0.2).astype(int)
    return df


def add_weather_features(
    table: pd.DataFrame,
    catalog: pd.DataFrame,
    cfg: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign districts, fetch weather, join on (district_id, hour).

    Returns (table, catalog_with_district, weather_frame).
    """
    cat = catalog if "district_id" in catalog.columns else assign_districts(catalog)
    table = table.copy()
    if "district_id" not in table.columns:
        table = table.merge(cat[["station_id", "district_id", "district_name"]], on="station_id", how="left")
    hours = pd.to_datetime(table["hour"])
    wx = load_district_weather(
        cfg,
        district_ids=cat["district_id"].dropna().astype(int).unique().tolist(),
        start=hours.min(),
        end=hours.max(),
    )
    if wx.empty:
        return table, cat, wx
    wx = _add_weather_lags(wx)
    # align join key tz
    wx["hour"] = hourly_bin(wx["hour"]) if str(wx["hour"].dtype).endswith("UTC") else wx["hour"]
    cols = [
        "district_id",
        "hour",
        "wx_temp_c",
        "wx_precip_mm",
        "wx_rain_mm",
        "wx_wind_kmh",
        "wx_humidity_pct",
        "wx_cloud_pct",
        "wx_is_rain",
        "wx_temp_lag_1h",
        "wx_precip_lag_1h",
        "wx_precip_3h",
        "wx_precip_24h",
    ]
    cols = [c for c in cols if c in wx.columns]
    drop_existing = [c for c in cols if c in table.columns and c not in ("district_id", "hour")]
    if drop_existing:
        table = table.drop(columns=drop_existing)
    table = table.merge(wx[cols], on=["district_id", "hour"], how="left")
    return table, cat, wx


WEATHER_FEATURE_COLUMNS = [
    "district_id",
    "wx_temp_c",
    "wx_precip_mm",
    "wx_wind_kmh",
    "wx_humidity_pct",
    "wx_cloud_pct",
    "wx_is_rain",
    "wx_temp_lag_1h",
    "wx_precip_lag_1h",
    "wx_precip_3h",
    "wx_precip_24h",
]
