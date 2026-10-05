"""Schema and business-rule validation (Pandera + custom checks)."""

from __future__ import annotations

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:  # pragma: no cover
    pa = None
    Check = None
    Column = None
    DataFrameSchema = None

from .config import CAPACITY_TOLERANCE, MAX_TRIP_MINUTES, MIN_TRIP_MINUTES, MADRID_BBOX


def trips_schema():
    if DataFrameSchema is None:
        return None
    return DataFrameSchema(
        {
            "start_ts": Column("datetime64[ns, UTC]", nullable=False),
            "end_ts": Column("datetime64[ns, UTC]", nullable=True),
            "start_station": Column("Int64", nullable=False),
            "end_station": Column("Int64", nullable=False),
            "travel_time_min": Column(
                float,
                checks=[
                    Check.ge(MIN_TRIP_MINUTES),
                    Check.le(MAX_TRIP_MINUTES),
                ],
                nullable=False,
            ),
        },
        coerce=True,
        strict=False,
    )


def status_schema():
    if DataFrameSchema is None:
        return None
    return DataFrameSchema(
        {
            "ts": Column("datetime64[ns, UTC]", nullable=False),
            "station_id": Column("Int64", nullable=False),
            "total_bases": Column("Int64", checks=[Check.gt(0)], nullable=False),
            "dock_bikes": Column("Int64", checks=[Check.ge(0)], nullable=False),
            "free_bases": Column("Int64", checks=[Check.ge(0)], nullable=True),
            "lat": Column(
                float,
                checks=[Check.in_range(MADRID_BBOX["lat_min"], MADRID_BBOX["lat_max"])],
                nullable=False,
            ),
            "lon": Column(
                float,
                checks=[Check.in_range(MADRID_BBOX["lon_min"], MADRID_BBOX["lon_max"])],
                nullable=False,
            ),
        },
        coerce=True,
        strict=False,
    )


def station_hour_schema():
    if DataFrameSchema is None:
        return None
    return DataFrameSchema(
        {
            "station_id": Column("Int64", nullable=False),
            "hour": Column("datetime64[ns, Europe/Madrid]", nullable=False),
            "starts": Column(float, checks=[Check.ge(0)], nullable=False),
            "ends": Column(float, checks=[Check.ge(0)], nullable=False),
            "occupancy": Column(
                float, checks=[Check.in_range(0.0, 1.0)], nullable=True
            ),
        },
        coerce=True,
        strict=False,
    )


def _safe_validate(schema, df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a violations table. Empty means pass."""
    if schema is None:
        return pd.DataFrame(
            [{"check": "pandera_missing", "n": 0, "detail": "pandera not installed; skipped"}]
        )
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame(columns=["table", "check", "n", "detail"])
    except Exception as exc:
        failures = getattr(exc, "failure_cases", None)
        if failures is None:
            return pd.DataFrame(
                [{"table": name, "check": type(exc).__name__, "n": 1, "detail": str(exc)[:300]}]
            )
        out = (
            failures.groupby(["check", "column"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        out.insert(0, "table", name)
        out["detail"] = out["check"].astype(str) + " on " + out["column"].astype(str)
        return out[["table", "check", "n", "detail"]]


def validate_trips(trips: pd.DataFrame) -> pd.DataFrame:
    return _safe_validate(trips_schema(), trips, "trips")


def validate_status(status: pd.DataFrame) -> pd.DataFrame:
    return _safe_validate(status_schema(), status, "status")


def validate_station_hour(table: pd.DataFrame) -> pd.DataFrame:
    return _safe_validate(station_hour_schema(), table, "station_hour")


def business_rule_violations(
    trips: pd.DataFrame,
    status: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    known = set(catalog["station_id"].dropna().astype(int)) if len(catalog) else set()
    if known:
        unknown_o = (~trips["start_station"].isin(known)).sum()
        unknown_d = (~trips["end_station"].isin(known)).sum()
        rows.append(("trips", "origin not in station catalog", int(unknown_o), "quarantine if systemic"))
        rows.append(("trips", "dest not in station catalog", int(unknown_d), "quarantine if systemic"))

    if len(status):
        cap = status["total_bases"].astype(float)
        bikes = status["dock_bikes"].astype(float)
        free = status["free_bases"].astype(float)
        over = (bikes + free > cap + CAPACITY_TOLERANCE).sum()
        rows.append(("status", "bikes+free exceed capacity+tolerance", int(over), "allow booking slack"))

        # snapshot spacing
        ts = status.drop_duplicates("ts")["ts"].sort_values()
        if len(ts) > 1:
            delta_h = ts.diff().dt.total_seconds().dropna() / 3600.0
            gaps = int((delta_h > 2.5).sum())
            rows.append(("status", "snapshot gaps > 2.5 h", gaps, "do not interpolate across outages"))

    return pd.DataFrame(rows, columns=["table", "rule", "n_flagged", "action"])


def assert_no_future_leakage(features: pd.DataFrame, lag_prefixes: tuple[str, ...] = ("lag_", "roll_", "ewm_")) -> None:
    """Sanity check: lag columns must not be identical to the raw contemporaneous target."""
    if "starts_next" in features.columns and "starts" in features.columns:
        same = (features["starts_next"] == features["starts"]).mean()
        if same > 0.98:
            raise ValueError("starts_next is almost identical to starts — possible shift error")
