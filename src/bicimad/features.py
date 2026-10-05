"""Hourly station-time feature store and leakage-safe target engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .config import RANDOM_SEED
from .ingest import hourly_bin, to_madrid
from .events import EVENT_FEATURE_COLUMNS
from .weather import WEATHER_FEATURE_COLUMNS

LAGS_H = (1, 2, 3, 24, 48, 168)
ROLL_WINDOWS = (3, 24)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_station_hour_table(
    trips: pd.DataFrame,
    status: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Regular station × hour grid with starts, ends, occupancy, capacity."""
    status = status.copy()
    status["hour"] = hourly_bin(status["ts"])
    status_h = (
        status.sort_values("ts")
        .groupby(["station_id", "hour"], as_index=False)
        .last()
    )
    status_h["occupancy"] = np.where(
        status_h["total_bases"] > 0,
        status_h["dock_bikes"] / status_h["total_bases"],
        np.nan,
    )
    status_h["is_empty"] = (status_h["dock_bikes"] == 0).astype(int)
    status_h["is_full"] = (status_h["free_bases"] == 0).astype(int)

    trips = trips.copy()
    trips["start_hour"] = hourly_bin(trips["start_ts"])
    trips["end_hour"] = hourly_bin(trips["end_ts"])
    starts = (
        trips.groupby(["start_station", "start_hour"], as_index=False)
        .size()
        .rename(columns={"start_station": "station_id", "start_hour": "hour", "size": "starts"})
    )
    ends = (
        trips.groupby(["end_station", "end_hour"], as_index=False)
        .size()
        .rename(columns={"end_station": "station_id", "end_hour": "hour", "size": "ends"})
    )
    ride_min = (
        trips.groupby(["start_station", "start_hour"], as_index=False)["travel_time_min"]
        .sum()
        .rename(columns={"start_station": "station_id", "start_hour": "hour", "travel_time_min": "trip_minutes"})
    )

    hours = pd.Index(status_h["hour"].dropna().unique()).sort_values()
    stations = catalog["station_id"].dropna().unique()
    grid = pd.MultiIndex.from_product([stations, hours], names=["station_id", "hour"]).to_frame(index=False)
    grid["station_id"] = grid["station_id"].astype("Int64")

    table = grid.merge(
        status_h[
            [
                "station_id",
                "hour",
                "dock_bikes",
                "free_bases",
                "total_bases",
                "occupancy",
                "is_empty",
                "is_full",
                "activate",
                "no_available",
            ]
        ],
        on=["station_id", "hour"],
        how="left",
    )
    table = table.merge(starts, on=["station_id", "hour"], how="left")
    table = table.merge(ends, on=["station_id", "hour"], how="left")
    table = table.merge(ride_min, on=["station_id", "hour"], how="left")
    table["starts"] = table["starts"].fillna(0).astype(float)
    table["ends"] = table["ends"].fillna(0).astype(float)
    table["trip_minutes"] = table["trip_minutes"].fillna(0).astype(float)
    table["net_flow"] = table["ends"] - table["starts"]

    table = table.sort_values(["station_id", "hour"])
    for col in ("dock_bikes", "free_bases", "total_bases", "occupancy", "activate"):
        table[col] = table.groupby("station_id")[col].ffill()
    table["is_empty"] = (table["dock_bikes"].fillna(-1) == 0).astype(int)
    table["is_full"] = (table["free_bases"].fillna(-1) == 0).astype(int)

    meta = catalog[["station_id", "name", "lat", "lon", "total_bases"]].rename(
        columns={"total_bases": "capacity_static"}
    )
    table = table.merge(meta, on="station_id", how="left")
    table["capacity"] = table["total_bases"].fillna(table["capacity_static"])
    return table.reset_index(drop=True)


def add_calendar_features(table: pd.DataFrame) -> pd.DataFrame:
    df = table.copy()
    hour = to_madrid(df["hour"]) if not str(df["hour"].dtype).endswith("Europe/Madrid") else df["hour"]
    df["hour_of_day"] = hour.dt.hour
    df["dow"] = hour.dt.dayofweek
    df["month"] = hour.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    try:
        import holidays as hol

        md_holidays = hol.country_holidays("ES", subdiv="MD")
        dates = hour.dt.date
        df["is_holiday"] = dates.map(lambda d: int(d in md_holidays))
    except Exception:
        df["is_holiday"] = 0
    return df


def add_lags_and_rolling(table: pd.DataFrame) -> pd.DataFrame:
    df = table.sort_values(["station_id", "hour"]).copy()
    grouped = df.groupby("station_id", sort=False)
    for col in ("starts", "ends", "net_flow", "occupancy", "dock_bikes"):
        for lag in LAGS_H:
            df[f"lag_{col}_{lag}h"] = grouped[col].shift(lag)
        lag1 = df[f"lag_{col}_1h"]
        g1 = lag1.groupby(df["station_id"], sort=False)
        for win in ROLL_WINDOWS:
            rolled = g1.rolling(win, min_periods=1)
            df[f"roll_mean_{col}_{win}h"] = rolled.mean().reset_index(level=0, drop=True)
            df[f"roll_std_{col}_{win}h"] = rolled.std().reset_index(level=0, drop=True)
        df[f"ewm_{col}"] = g1.ewm(span=12, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def add_targets(table: pd.DataFrame) -> pd.DataFrame:
    df = table.sort_values(["station_id", "hour"]).copy()
    g = df.groupby("station_id", sort=False)
    df["starts_next"] = g["starts"].shift(-1)
    df["ends_next"] = g["ends"].shift(-1)
    df["net_flow_next"] = g["net_flow"].shift(-1)
    df["occupancy_next"] = g["occupancy"].shift(-1)
    df["dock_bikes_next"] = g["dock_bikes"].shift(-1)
    df["is_empty_next"] = g["is_empty"].shift(-1)
    df["is_full_next"] = g["is_full"].shift(-1)
    cap = df["capacity"].fillna(df["total_bases"]).clip(lower=1)
    # Oracle stock: enough bikes for next-hour starts, enough docks for next-hour ends
    desired = df["starts_next"].clip(lower=0)
    max_for_returns = (cap - df["ends_next"].clip(lower=0)).clip(lower=0)
    df["target_stock"] = np.minimum(desired, max_for_returns).clip(lower=0, upper=cap)
    crit = np.select(
        [df["is_empty_next"] == 1, df["is_full_next"] == 1],
        [1, 2],
        default=0,
    )
    df["critical_next"] = pd.Series(crit, index=df.index).astype("Int64")
    df["is_critical_next"] = (
        (df["is_empty_next"] == 1) | (df["is_full_next"] == 1)
    ).astype("Int64")
    return df


def cluster_stations(table: pd.DataFrame, n_clusters: int = 4, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Cluster stations by 24-h mean start/end profile."""
    tmp = table.copy()
    hour = to_madrid(tmp["hour"]) if not str(tmp["hour"].dtype).endswith("Europe/Madrid") else tmp["hour"]
    tmp["hod"] = hour.dt.hour
    starts = tmp.pivot_table(index="station_id", columns="hod", values="starts", aggfunc="mean").fillna(0)
    ends = tmp.pivot_table(index="station_id", columns="hod", values="ends", aggfunc="mean").fillna(0)
    starts.columns = [f"s{c}" for c in starts.columns]
    ends.columns = [f"e{c}" for c in ends.columns]
    X = starts.join(ends, how="inner")
    if len(X) < n_clusters:
        n_clusters = max(1, len(X))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X.values)
    return pd.DataFrame({"station_id": X.index.astype("Int64"), "cluster": labels})


def add_neighbor_features(table: pd.DataFrame, catalog: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    cat = catalog.dropna(subset=["lat", "lon", "station_id"]).copy()
    if len(cat) < 3:
        table["nbr_occ_mean"] = np.nan
        table["nbr_net_mean"] = np.nan
        table["sys_occ_mean"] = table.groupby("hour")["occupancy"].transform("mean")
        return table
    coords = cat[["lat", "lon"]].to_numpy()
    ids = cat["station_id"].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(cat)), metric="haversine")
    nn.fit(np.radians(coords))
    dist, idx = nn.kneighbors(np.radians(coords))
    neighbor_map = {}
    for i, station in enumerate(ids):
        nbrs = [int(ids[j]) for j in idx[i] if int(ids[j]) != int(station)][:k]
        neighbor_map[int(station)] = nbrs

    occ = table.pivot_table(index="hour", columns="station_id", values="occupancy")
    net = table.pivot_table(index="hour", columns="station_id", values="net_flow")
    nbr_occ_rows = []
    nbr_net_rows = []
    for sid, nbrs in neighbor_map.items():
        cols = [c for c in nbrs if c in occ.columns]
        if not cols:
            continue
        nbr_occ_rows.append(occ[cols].mean(axis=1).rename(sid))
        nbr_net_rows.append(net[cols].mean(axis=1).rename(sid))
    if nbr_occ_rows:
        nbr_occ = pd.concat(nbr_occ_rows, axis=1).melt(ignore_index=False, var_name="station_id", value_name="nbr_occ_mean")
        nbr_occ = nbr_occ.reset_index()
        nbr_net = pd.concat(nbr_net_rows, axis=1).melt(ignore_index=False, var_name="station_id", value_name="nbr_net_mean")
        nbr_net = nbr_net.reset_index()
        table = table.merge(nbr_occ, on=["hour", "station_id"], how="left")
        table = table.merge(nbr_net, on=["hour", "station_id"], how="left")
    else:
        table["nbr_occ_mean"] = np.nan
        table["nbr_net_mean"] = np.nan
    table["sys_occ_mean"] = table.groupby("hour")["occupancy"].transform("mean")
    return table


def add_train_only_hour_of_week_means(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Historical hour-of-week means computed on train only, then joined to all splits."""
    t = train.copy()
    hour = to_madrid(t["hour"]) if not str(t["hour"].dtype).endswith("Europe/Madrid") else t["hour"]
    t["hod"] = hour.dt.hour
    t["dow"] = hour.dt.dayofweek
    stats = (
        t.groupby(["station_id", "dow", "hod"], as_index=False)
        .agg(how_starts=("starts", "mean"), how_occ=("occupancy", "mean"))
    )

    def _join(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        h = to_madrid(out["hour"]) if not str(out["hour"].dtype).endswith("Europe/Madrid") else out["hour"]
        out["hod"] = h.dt.hour
        out["dow"] = h.dt.dayofweek
        out = out.merge(stats, on=["station_id", "dow", "hod"], how="left")
        return out

    return tuple(_join(df) for df in (train, *others))


FEATURE_COLUMNS = [
    "hour_of_day",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "capacity",
    "dock_bikes",
    "occupancy",
    "starts",
    "ends",
    "net_flow",
    "is_empty",
    "is_full",
    "lat",
    "lon",
    "cluster",
    "nbr_occ_mean",
    "nbr_net_mean",
    "sys_occ_mean",
    "how_starts",
    "how_occ",
] + WEATHER_FEATURE_COLUMNS + EVENT_FEATURE_COLUMNS


def feature_matrix(df: pd.DataFrame) -> list[str]:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    extra = [
        c
        for c in df.columns
        if c.startswith(("lag_", "roll_", "ewm_")) and df[c].dtype != "O"
    ]
    return cols + extra
