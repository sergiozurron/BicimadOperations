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


def daily_presence_utilization(
    snapshots: pd.DataFrame,
    *,
    tz: str = "Europe/Madrid",
    max_gap: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Time in use over time present, for each calendar day in ``tz``.

    ``R(t)`` is ``n_riding`` and ``B(t)`` is ``n_bikes``. Between consecutive
    snapshots both counts stay at the earlier snapshot, and an interval that
    crosses midnight is split between the two days. A gap longer than
    ``max_gap`` (6 hours when omitted) is left out of both integrals. The
    frame has one row for every day that receives a positive length of
    bracketed time: ``date``, ``used_hours``, ``present_hours``,
    ``covered_hours``, and ``utilization``.
    """
    if max_gap is None:
        max_gap = pd.Timedelta(hours=6)
    missing = {"ts", "n_riding", "n_bikes"} - set(snapshots.columns)
    if missing:
        raise ValueError(f"snapshots is missing columns: {sorted(missing)}")
    frame = snapshots.loc[:, ["ts", "n_riding", "n_bikes"]].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    if frame.empty or frame["ts"].isna().any():
        raise ValueError("snapshots need a timestamp on every row")
    frame = frame.sort_values("ts", kind="mergesort")
    local = frame["ts"].dt.tz_convert(tz)
    gap = local.shift(-1) - local
    keep = gap.gt(pd.Timedelta(0)) & gap.le(max_gap)
    columns = ["date", "used_hours", "present_hours", "covered_hours", "utilization"]
    empty = pd.DataFrame(columns=columns)
    if not keep.any():
        return empty

    left = local[keep]
    right = local.shift(-1)[keep]
    hours = gap[keep].dt.total_seconds().to_numpy() / 3600.0
    riding = pd.to_numeric(frame.loc[keep, "n_riding"], errors="coerce").fillna(0).to_numpy()
    bikes = pd.to_numeric(frame.loc[keep, "n_bikes"], errors="coerce").fillna(0).to_numpy()
    day_left = left.dt.floor("D")
    day_right = right.dt.floor("D")
    same_day = day_left.eq(day_right).to_numpy()
    midnight = day_left + pd.offsets.Day(1)
    if bool(day_right.gt(midnight).any()):
        raise RuntimeError("a kept snapshot gap crosses more than one midnight")
    to_midnight = (midnight - left).dt.total_seconds().to_numpy() / 3600.0
    used_left = np.where(same_day, hours * riding, to_midnight * riding)
    present_left = np.where(same_day, hours * bikes, to_midnight * bikes)
    covered_left = np.where(same_day, hours, to_midnight)
    used_right = np.where(same_day, 0.0, (hours - to_midnight) * riding)
    present_right = np.where(same_day, 0.0, (hours - to_midnight) * bikes)
    covered_right = np.where(same_day, 0.0, hours - to_midnight)
    parts = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": day_left.to_numpy(),
                    "used_hours": used_left,
                    "present_hours": present_left,
                    "covered_hours": covered_left,
                }
            ),
            pd.DataFrame(
                {
                    "date": day_right.to_numpy(),
                    "used_hours": used_right,
                    "present_hours": present_right,
                    "covered_hours": covered_right,
                }
            ),
        ],
        ignore_index=True,
    )
    parts = parts.loc[parts["covered_hours"] > 0]
    daily = parts.groupby("date", as_index=False).sum(numeric_only=True)
    present = daily["present_hours"].to_numpy()
    daily["utilization"] = np.divide(
        daily["used_hours"].to_numpy(),
        present,
        out=np.full(len(daily), np.nan),
        where=present > 0,
    )
    daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.tz_convert(tz)
    return daily.loc[:, columns].sort_values("date").reset_index(drop=True)


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
