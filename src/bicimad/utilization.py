"""Bike utilization rate and lost-demand simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fleet_size_from_trips(trips: pd.DataFrame) -> int | None:
    if "bike_id" not in trips.columns:
        return None
    n = trips["bike_id"].nunique(dropna=True)
    return int(n) if n else None


def fleet_size_from_status(status: pd.DataFrame) -> float:
    """Upper bound: max observed bikes sitting at stations in a single snapshot."""
    if status.empty:
        return np.nan
    per_snap = status.groupby("ts")["dock_bikes"].sum()
    return float(per_snap.mean())


def fleet_utilization(
    trips: pd.DataFrame,
    n_bikes: float,
    window_hours: float | None = None,
) -> dict:
    """U = sum(travel_time) / (N_bikes * T). T inferred from trip span if omitted."""
    tt_hours = pd.to_numeric(trips["travel_time_min"], errors="coerce").sum() / 60.0
    if window_hours is None:
        t0 = trips["start_ts"].min()
        t1 = trips["end_ts"].max()
        window_hours = max((t1 - t0).total_seconds() / 3600.0, 1.0)
    denom = float(n_bikes) * float(window_hours)
    u = float(tt_hours / denom) if denom else np.nan
    return {
        "trip_hours": float(tt_hours),
        "n_bikes": float(n_bikes),
        "window_hours": float(window_hours),
        "utilization": u,
        "utilization_pct": 100.0 * u if np.isfinite(u) else np.nan,
    }


def availability_metrics(station_hour: pd.DataFrame) -> dict:
    n = len(station_hour)
    empty = int(station_hour["is_empty"].fillna(0).sum())
    full = int(station_hour["is_full"].fillna(0).sum())
    ok = n - empty - full + int(((station_hour["is_empty"] == 1) & (station_hour["is_full"] == 1)).sum())
    return {
        "station_hours": n,
        "empty_hours": empty,
        "full_hours": full,
        "empty_rate": empty / n if n else np.nan,
        "full_rate": full / n if n else np.nan,
        "neither_empty_nor_full_rate": ok / n if n else np.nan,
    }


def lost_demand(
    station_hour: pd.DataFrame,
    stock_col: str = "dock_bikes",
    start_col: str = "starts_next",
    end_col: str = "ends_next",
) -> pd.DataFrame:
    """Lost starts/ends for the *upcoming* hour given stock at the end of the current hour.

    `target_stock` is built from `starts_next` / `ends_next`, so evaluation must use
    those same columns. Falls back to contemporaneous starts/ends if the next-hour
    labels are absent.
    """
    df = station_hour.copy()
    stock = pd.to_numeric(df[stock_col], errors="coerce").fillna(0).clip(lower=0)
    cap = pd.to_numeric(df["capacity"], errors="coerce")
    cap = cap.fillna(pd.to_numeric(df.get("total_bases"), errors="coerce")).fillna(stock)
    free = (cap - stock).clip(lower=0)
    if start_col not in df.columns:
        start_col = "starts"
    if end_col not in df.columns:
        end_col = "ends"
    starts = pd.to_numeric(df[start_col], errors="coerce").fillna(0)
    ends = pd.to_numeric(df[end_col], errors="coerce").fillna(0)
    df["lost_starts"] = np.maximum(starts - stock, 0)
    df["lost_ends"] = np.maximum(ends - free, 0)
    df["served_starts"] = starts - df["lost_starts"]
    df["served_ends"] = ends - df["lost_ends"]
    df["_demand_starts"] = starts
    df["_demand_ends"] = ends
    return df


def summarize_lost(df: pd.DataFrame) -> dict:
    starts = df["_demand_starts"].sum() if "_demand_starts" in df.columns else df["starts"].sum()
    ends = df["_demand_ends"].sum() if "_demand_ends" in df.columns else df["ends"].sum()
    return {
        "lost_starts": float(df["lost_starts"].sum()),
        "lost_ends": float(df["lost_ends"].sum()),
        "served_starts": float(df["served_starts"].sum()),
        "start_service_rate": float(1 - df["lost_starts"].sum() / starts) if starts else np.nan,
        "end_service_rate": float(1 - df["lost_ends"].sum() / ends) if ends else np.nan,
    }


def simulate_policies(
    station_hour: pd.DataFrame,
    predicted_stock: pd.Series | None = None,
    lp_stock: pd.Series | None = None,
) -> pd.DataFrame:
    """Compare observed stock vs predicted target vs LP-rebalanced stock."""
    rows = []
    base = lost_demand(station_hour, "dock_bikes")
    rows.append({"policy": "observed (no extra rebalancing)", **summarize_lost(base)})

    oracle = station_hour.copy()
    if "target_stock" in oracle.columns:
        oracle["dock_bikes"] = oracle["target_stock"]
        rows.append({"policy": "oracle target stock (perfect foresight + perfect rebalancing)", **summarize_lost(lost_demand(oracle))})

    if predicted_stock is not None:
        pred = station_hour.copy()
        pred["dock_bikes"] = predicted_stock.to_numpy()
        rows.append({"policy": "predicted target stock (perfect rebalancing)", **summarize_lost(lost_demand(pred))})

    if lp_stock is not None:
        lp = station_hour.copy()
        lp["dock_bikes"] = lp_stock.to_numpy()
        rows.append({"policy": "LP rebalancing (truck capacity)", **summarize_lost(lost_demand(lp))})

    return pd.DataFrame(rows)
