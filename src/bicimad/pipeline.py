"""End-to-end pipeline: ingest → clean → station-hour feature store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig, discover_months
from .features import (
    add_calendar_features,
    add_lags_and_rolling,
    add_neighbor_features,
    add_targets,
    build_station_hour_table,
    cluster_stations,
)
from .ingest import (
    build_station_catalog,
    load_status_for_months,
    load_trips_for_months,
)
from .perf import PerfLogger
from .quality import clean_status, clean_trips
from .validate import business_rule_violations, validate_status, validate_trips
from .events import add_event_features
from .weather import add_weather_features


def parquet_path(cfg: ProjectConfig, kind: str) -> Path:
    cfg.ensure_dirs()
    scope = cfg.data_scope
    months = "-".join(cfg.months_for_scope()[:1] + cfg.months_for_scope()[-1:])
    return cfg.processed_dir / f"{kind}_{scope}_{months}.parquet"


def run_pipeline(cfg: ProjectConfig, perf: PerfLogger | None = None) -> dict:
    """Build the modeling table for the configured scope.

    Returns dict with trips, status, catalog, station_hour, quality reports, perf.
    """
    perf = perf or PerfLogger()
    cfg.ensure_dirs()
    months = cfg.months_for_scope()
    if not months:
        raise RuntimeError(f"No months resolved for DATA_SCOPE={cfg.data_scope!r}")

    feat_path = cfg.feature_dir / f"station_hour_{cfg.data_scope}.parquet"
    if feat_path.exists() and not cfg.rebuild_parquet:
        with perf.stage("load_cached_features", rows=None) as info:
            table = pd.read_parquet(feat_path)
            info["rows"] = len(table)
        catalog = pd.read_parquet(cfg.processed_dir / f"stations_{cfg.data_scope}.parquet")
        trips = pd.read_parquet(cfg.processed_dir / f"trips_{cfg.data_scope}.parquet")
        status = pd.read_parquet(cfg.processed_dir / f"status_{cfg.data_scope}.parquet")
        dirty = False
        if "wx_temp_c" not in table.columns:
            with perf.stage("weather_features") as info:
                table, catalog, _wx = add_weather_features(table, catalog, cfg)
                info["rows"] = int(table["wx_temp_c"].notna().sum()) if "wx_temp_c" in table.columns else 0
            dirty = True
        if "evt_n" not in table.columns:
            with perf.stage("event_features") as info:
                table, _eh = add_event_features(table, cfg)
                info["rows"] = int((table["evt_n"] > 0).sum()) if "evt_n" in table.columns else 0
            dirty = True
        if dirty:
            table.to_parquet(feat_path, index=False)
            catalog.to_parquet(cfg.processed_dir / f"stations_{cfg.data_scope}.parquet", index=False)
        return {
            "trips": trips,
            "status": status,
            "catalog": catalog,
            "station_hour": table,
            "months": months,
            "perf": perf,
            "cached": True,
        }

    with perf.stage("ingest_trips", months=len(months)) as info:
        trips_raw = load_trips_for_months(cfg, months)
        info["rows"] = len(trips_raw)

    with perf.stage("ingest_status", months=len(months)) as info:
        status_raw = load_status_for_months(cfg, months)
        info["rows"] = len(status_raw)

    with perf.stage("clean_trips") as info:
        trips, trip_impact = clean_trips(trips_raw)
        info["rows"] = len(trips)

    with perf.stage("clean_status") as info:
        status, status_impact = clean_status(status_raw)
        info["rows"] = len(status)

    with perf.stage("station_catalog") as info:
        catalog = build_station_catalog(status)
        info["rows"] = len(catalog)

    with perf.stage("validate") as info:
        v_trips = validate_trips(trips)
        v_status = validate_status(status)
        v_biz = business_rule_violations(trips, status, catalog)
        info["rows"] = len(v_trips) + len(v_status) + len(v_biz)

    with perf.stage("station_hour_grid") as info:
        table = build_station_hour_table(trips, status, catalog)
        info["rows"] = len(table)

    with perf.stage("calendar_features") as info:
        table = add_calendar_features(table)
        info["rows"] = len(table)

    with perf.stage("weather_features") as info:
        table, catalog, wx = add_weather_features(table, catalog, cfg)
        info["rows"] = int(table["wx_temp_c"].notna().sum()) if "wx_temp_c" in table.columns else 0

    with perf.stage("event_features") as info:
        table, _eh = add_event_features(table, cfg)
        info["rows"] = int((table["evt_n"] > 0).sum()) if "evt_n" in table.columns else 0

    with perf.stage("cluster_stations") as info:
        clusters = cluster_stations(table)
        table = table.merge(clusters, on="station_id", how="left")
        info["rows"] = clusters["cluster"].nunique()

    with perf.stage("neighbor_features") as info:
        table = add_neighbor_features(table, catalog, k=5)
        info["rows"] = len(table)

    with perf.stage("lags_rolling") as info:
        table = add_lags_and_rolling(table)
        info["rows"] = len(table)

    with perf.stage("targets") as info:
        table = add_targets(table)
        info["rows"] = int(table["target_stock"].notna().sum())

    with perf.stage("write_parquet") as info:
        cfg.ensure_dirs()
        trips.to_parquet(cfg.processed_dir / f"trips_{cfg.data_scope}.parquet", index=False)
        status.to_parquet(cfg.processed_dir / f"status_{cfg.data_scope}.parquet", index=False)
        catalog.to_parquet(cfg.processed_dir / f"stations_{cfg.data_scope}.parquet", index=False)
        table.to_parquet(feat_path, index=False)
        info["rows"] = len(table)

    return {
        "trips_raw": trips_raw,
        "status_raw": status_raw,
        "trips": trips,
        "status": status,
        "catalog": catalog,
        "station_hour": table,
        "trip_impact": trip_impact,
        "status_impact": status_impact,
        "violations_trips": v_trips,
        "violations_status": v_status,
        "violations_business": v_biz,
        "months": months,
        "inventory": discover_months(cfg),
        "perf": perf,
        "cached": False,
    }
