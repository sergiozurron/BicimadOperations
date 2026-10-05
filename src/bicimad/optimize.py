"""Linear program: move bikes between stations to cut empty/full risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def redistribution_lp(
    snapshot: pd.DataFrame,
    truck_capacity: int = 40,
    empty_penalty: float = 8.0,
    full_penalty: float = 6.0,
    move_cost_per_km: float = 0.15,
    time_limit_s: int = 30,
) -> dict:
    """Single-window pooled-truck LP.

    snapshot columns required:
      station_id, dock_bikes, capacity, lat, lon, starts_hat, ends_hat
    """
    if pulp is None:
        raise ImportError("pulp is required for the redistribution LP")

    df = snapshot.dropna(subset=["station_id", "dock_bikes", "capacity", "lat", "lon"]).copy()
    df["station_id"] = df["station_id"].astype(int)
    ids = df["station_id"].tolist()
    if len(ids) < 2:
        raise ValueError("Need at least two stations")

    bikes = dict(zip(ids, df["dock_bikes"].astype(float)))
    cap = dict(zip(ids, df["capacity"].astype(float)))
    starts = dict(zip(ids, df["starts_hat"].fillna(0).astype(float)))
    ends = dict(zip(ids, df["ends_hat"].fillna(0).astype(float)))
    lat = dict(zip(ids, df["lat"].astype(float)))
    lon = dict(zip(ids, df["lon"].astype(float)))

    dist = {(i, j): _haversine_km(lat[i], lon[i], lat[j], lon[j]) for i in ids for j in ids if i != j}

    prob = pulp.LpProblem("bicimad_redistribute", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("move", dist.keys(), lowBound=0, cat="Continuous")
    stock = pulp.LpVariable.dicts("stock", ids, lowBound=0, cat="Continuous")
    lost_s = pulp.LpVariable.dicts("lost_start", ids, lowBound=0, cat="Continuous")
    lost_e = pulp.LpVariable.dicts("lost_end", ids, lowBound=0, cat="Continuous")

    for i in ids:
        outgoing = pulp.lpSum(x[i, j] for j in ids if j != i)
        incoming = pulp.lpSum(x[j, i] for j in ids if j != i)
        prob += stock[i] == bikes[i] - outgoing + incoming
        prob += stock[i] <= cap[i]
        prob += lost_s[i] >= starts[i] - stock[i]
        prob += lost_e[i] >= ends[i] - (cap[i] - stock[i])

    prob += pulp.lpSum(x[k] for k in dist) <= truck_capacity

    prob += (
        empty_penalty * pulp.lpSum(lost_s[i] for i in ids)
        + full_penalty * pulp.lpSum(lost_e[i] for i in ids)
        + move_cost_per_km * pulp.lpSum(dist[i, j] * x[i, j] for i, j in dist)
    )

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    status = prob.solve(solver)

    moves = []
    for i, j in dist:
        val = pulp.value(x[i, j]) or 0.0
        if val > 0.05:
            moves.append(
                {
                    "from_id": i,
                    "to_id": j,
                    "bikes": round(val, 2),
                    "km": round(dist[i, j], 2),
                }
            )
    stock_after = {i: pulp.value(stock[i]) for i in ids}
    return {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(prob.objective),
        "bikes_moved": float(sum(m["bikes"] for m in moves)),
        "n_arcs": len(moves),
        "lost_starts": float(sum(pulp.value(lost_s[i]) or 0 for i in ids)),
        "lost_ends": float(sum(pulp.value(lost_e[i]) or 0 for i in ids)),
        "moves": pd.DataFrame(moves),
        "stock_after": pd.Series(stock_after, name="lp_stock"),
    }
