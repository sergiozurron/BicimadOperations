"""Pipeline performance logger: wall time, peak RSS, row counts."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class PerfLogger:
    rows: list[dict] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, **extra) -> Iterator[dict]:
        start = time.perf_counter()
        rss0 = _rss_mb()
        info: dict = {"stage": name, **extra}
        try:
            yield info
        finally:
            elapsed = time.perf_counter() - start
            rss1 = _rss_mb()
            rec = {
                "stage": name,
                "seconds": round(elapsed, 3),
                "rss_start_mb": None if rss0 is None else round(rss0, 1),
                "rss_end_mb": None if rss1 is None else round(rss1, 1),
                "rss_delta_mb": None
                if rss0 is None or rss1 is None
                else round(rss1 - rss0, 1),
                **{k: v for k, v in info.items() if k != "stage"},
            }
            self.rows.append(rec)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)
