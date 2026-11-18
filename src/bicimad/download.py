"""Download EMT historical trips and station status into ``data/raw``.

The CKAN package ``historicos-de-bicimad-2017_2023`` publishes one ZIP per
year (2017–2023). Inside each ZIP the months are nested ZIP or RAR archives,
plus a ``__MACOSX`` tree that is discarded. Trip files are written to
``data/raw/trips`` and station snapshots to ``data/raw/status``, which is the
layout ``discover_trip_files`` / ``discover_status_files`` already read.

``DATA_SCOPE`` chooses the months:

- ``demo`` — ``ProjectConfig.demo_months`` (default October 2022)
- ``2022`` — every month of 2022
- ``all`` — every month the seven yearly archives contain

A month that is already on disk is left untouched.

``ensure_status_files`` is the status-only counterpart used by the
infrastructure notebook. It fetches station snapshots and leaves trip
archives on the server.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd

from .config import (
    ProjectConfig,
    _month_from_stem,
    discover_status_files,
    discover_trip_files,
)

DATASET_PAGE = "https://datos.emtmadrid.es/dataset/historicos-de-bicimad-2017_2023"
CKAN_PACKAGE = "historicos-de-bicimad-2017_2023"
CKAN_URL = f"https://datos.emtmadrid.es/api/3/action/package_show?id={CKAN_PACKAGE}"

# Direct file URLs published on the dataset page. Used when the CKAN API
# is unreachable. Confirmed against the live package (filename bicimad_YYYY.zip).
FALLBACK_YEAR_URLS: dict[int, str] = {
    2017: "https://media.emtmadrid.es/-HoGTStPZeC",
    2018: "https://media.emtmadrid.es/-9iZNUjrjri",
    2019: "https://media.emtmadrid.es/-WNgfSj2ZvC",
    2020: "https://media.emtmadrid.es/-rkuBymuFJX",
    2021: "https://media.emtmadrid.es/-BRk4rTaAdV",
    2022: "https://media.emtmadrid.es/-uHaW6iZkhG",
    2023: "https://media.emtmadrid.es/-9Nii5DXo4x",
}

USER_AGENT = "bicimad-operations-optimization/0.1 (research)"
_REPORT_COLUMNS = ["year", "month", "kind", "file", "bytes", "action"]
_BUNDLE_RE = re.compile(r"(20\d{2})[_-](\d{1,2})[_-](\d{1,2})")
_PREDICT_RE = re.compile(r"(.+)-(csv|json)\.(zip|rar)$", re.IGNORECASE)


def months_for_download(cfg: ProjectConfig) -> set[str] | None:
    """Months this scope should materialise. ``None`` means every month in the archives."""
    if cfg.data_scope == "all":
        return None
    if cfg.data_scope == "2022":
        return {f"2022-{month:02d}" for month in range(1, 13)}
    if cfg.data_scope == "demo":
        return set(cfg.demo_months)
    raise ValueError(f"Unknown DATA_SCOPE {cfg.data_scope!r}")


def ensure_raw_data(cfg: ProjectConfig) -> pd.DataFrame:
    """Download and unpack the EMT archives required by ``cfg``.

    Returns one row per trip/status file: ``already present``, ``downloaded``,
    or ``not published`` when that archive has no file of that kind.
    """
    cfg.trips_raw.mkdir(parents=True, exist_ok=True)
    cfg.status_raw.mkdir(parents=True, exist_ok=True)
    wanted = months_for_download(cfg)
    if wanted is not None and _months_fully_present(cfg, wanted):
        print(
            "EMT raw files already on disk for "
            + ", ".join(sorted(wanted))
            + " — skipping download."
        )
        return _frame(_present_rows(cfg, wanted))

    resources = fetch_year_resources()
    rows: list[dict] = []
    seen: dict[str, set[str]] = {"trips": set(), "status": set()}
    visited: set[int] = set()

    def _visit(year: int) -> None:
        if year in visited:
            return
        if year not in resources:
            raise RuntimeError(
                f"The EMT package has no archive for {year}. Dataset page: {DATASET_PAGE}"
            )
        visited.add(year)
        rows.extend(_acquire_year(cfg, year, resources[year], wanted, seen))

    if wanted is None:
        for year in sorted(resources):
            _visit(year)
    else:
        for year in sorted({int(month[:4]) for month in wanted}):
            _visit(year)
        # December of year Y is sometimes filed in the Y+1 ZIP (2017-12 lives in bicimad_2018).
        still_missing = [
            month
            for month in sorted(wanted)
            if _month_file(cfg, month, "trips") is None and _month_file(cfg, month, "status") is None
        ]
        for year in sorted({int(month[:4]) + 1 for month in still_missing}):
            if year in resources:
                _visit(year)
        rows.extend(_gap_rows(cfg, wanted, seen))

    if not rows:
        return _frame([])
    return _frame(rows)


def ensure_status_files(cfg: ProjectConfig) -> pd.DataFrame:
    """Download every station-status member of the EMT yearly archives.

    Trip members are not fetched. A status file already on disk is left as it
    is. A year whose archive contains no status member is one row with action
    ``not published``.
    """
    cfg.status_raw.mkdir(parents=True, exist_ok=True)
    resources = fetch_year_resources()
    rows: list[dict] = []
    seen: dict[str, set[str]] = {"trips": set(), "status": set()}
    unpublished: list[dict] = []
    for year in sorted(resources):
        got = _acquire_year(cfg, year, resources[year], None, seen, kinds={"status"})
        if not got:
            print(f"EMT {year}: archive has no station-status files.", flush=True)
            unpublished.append(
                {
                    "year": year,
                    "month": str(year),
                    "kind": "status",
                    "file": "",
                    "bytes": 0,
                    "action": "not published",
                }
            )
            continue
        rows.extend(got)
    frame = _frame(rows)
    if not unpublished:
        return frame
    extra = pd.DataFrame(unpublished, columns=_REPORT_COLUMNS)
    return pd.concat([frame, extra], ignore_index=True)


def fetch_year_resources() -> dict[int, str]:
    """Map calendar year → ZIP URL from the CKAN package, with a fixed fallback."""
    try:
        payload = _read_json(CKAN_URL)
        found: dict[int, str] = {}
        for resource in payload["result"]["resources"]:
            if (resource.get("format") or "").upper() == "PDF":
                continue
            name = resource.get("name") or ""
            match = re.search(r"\b(20\d{2})\b", name)
            if not match or "bicimad" not in name.lower():
                continue
            url = resource.get("url") or ""
            if not url.startswith("https://"):
                continue
            found[int(match.group(1))] = url
        if found:
            return found
        print(f"CKAN package {CKAN_PACKAGE} listed no yearly ZIPs; using the dataset-page URLs.")
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as exc:
        print(f"CKAN listing failed ({exc}); using the dataset-page URLs at {DATASET_PAGE}.")
    return dict(FALLBACK_YEAR_URLS)


def is_junk(name: str) -> bool:
    """AppleDouble / Finder metadata shipped inside the EMT ZIPs."""
    if not name or name.endswith("/"):
        return True
    parts = PurePosixPath(name).parts
    base = parts[-1]
    if base.startswith("._") or base == ".DS_Store":
        return True
    return "__MACOSX" in parts


def kind_of(name: str) -> str | None:
    """``trips`` or ``status`` from an archive member name. ``None`` if it is neither."""
    base = PurePosixPath(name).name.lower()
    if "usage_bicimad" in base or "movement" in base or base.startswith("trips_"):
        return "trips"
    if "station" in base or "estacion" in base:
        return "status"
    stem = re.sub(r"\.(zip|rar|json|csv)$", "", base, flags=re.IGNORECASE)
    stem = re.sub(r"-(csv|json)$", "", stem, flags=re.IGNORECASE)
    # Bare YYYYMM.json / YYYYMM.zip / YYYYMM-json.zip are station snapshots
    # (the trip files always carry Usage, movements, or trips_ in the name).
    if re.fullmatch(r"\d{6}", stem):
        return "status"
    return None


def month_of(name: str) -> str | None:
    """YYYY-MM from a member name, ignoring the ``-csv`` / ``-json`` packaging suffix."""
    base = PurePosixPath(name).name
    stem = re.sub(r"\.(zip|rar|json|csv)$", "", base, flags=re.IGNORECASE)
    stem = re.sub(r"-(csv|json)$", "", stem, flags=re.IGNORECASE)
    return _month_from_stem(stem)


def bundle_months(name: str) -> set[str] | None:
    """Months covered by a multi-month bundle such as ``BiciMAD_stations_2019_7_12``."""
    match = _BUNDLE_RE.search(PurePosixPath(name).name)
    if not match:
        return None
    year = int(match.group(1))
    start, end = int(match.group(2)), int(match.group(3))
    if not (1 <= start <= 12 and 1 <= end <= 12):
        return None
    lo, hi = min(start, end), max(start, end)
    return {f"{year:04d}-{month:02d}" for month in range(lo, hi + 1)}


def predicted_output_name(name: str) -> str | None:
    """Filename a monthly member unpacks to, when the archive name determines it."""
    base = PurePosixPath(name).name
    lower = base.lower()
    if lower.endswith(".csv") or lower.endswith(".json"):
        return base
    match = _PREDICT_RE.fullmatch(base)
    if match:
        return f"{match.group(1)}.{match.group(2).lower()}"
    packaged = re.fullmatch(r"(.+)\.(zip|rar)", base, flags=re.IGNORECASE)
    if packaged and bundle_months(base) is None:
        return f"{packaged.group(1)}.json"
    return None


def _months_fully_present(cfg: ProjectConfig, months: set[str]) -> bool:
    return all(
        _month_file(cfg, month, "trips") is not None and _month_file(cfg, month, "status") is not None
        for month in months
    )


def _month_file(cfg: ProjectConfig, month: str, kind: str) -> Path | None:
    files = discover_trip_files(cfg) if kind == "trips" else discover_status_files(cfg)
    path = files.get(month)
    if path is not None and path.is_file() and path.stat().st_size > 0:
        return path
    return None


def _present_rows(cfg: ProjectConfig, months: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for month in sorted(months):
        for kind in ("trips", "status"):
            path = _month_file(cfg, month, kind)
            if path is None:
                continue
            rows.append(_row(month, kind, path.name, path.stat().st_size, "already present"))
    return rows


def _gap_rows(cfg: ProjectConfig, wanted: set[str], seen: dict[str, set[str]]) -> list[dict]:
    """Months the visited archives do not publish and that are not on disk."""
    rows: list[dict] = []
    for month in sorted(wanted):
        for kind in ("trips", "status"):
            if _month_file(cfg, month, kind) is not None:
                continue
            if month not in seen[kind]:
                rows.append(_row(month, kind, "", 0, "not published"))
    return rows


def _row(month: str, kind: str, file: str, size: int, action: str) -> dict:
    return {
        "year": int(month[:4]),
        "month": month,
        "kind": kind,
        "file": file,
        "bytes": size,
        "action": action,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=_REPORT_COLUMNS)
    if frame.empty:
        return frame
    # A month can be reported twice when the ZIP holds both a loose file and a nested copy.
    preference = {"downloaded": 0, "already present": 1, "not published": 2}
    frame["_pref"] = frame["action"].map(preference).fillna(9)
    frame = frame.sort_values(["month", "kind", "_pref", "file"], kind="mergesort")
    frame = frame.drop_duplicates(["month", "kind", "file"], keep="first")
    return frame.drop(columns="_pref").reset_index(drop=True)


def _acquire_year(
    cfg: ProjectConfig,
    year: int,
    url: str,
    wanted: set[str] | None,
    seen: dict[str, set[str]],
    kinds: set[str] | None = None,
) -> list[dict]:
    print(f"EMT {year}: listing {url}", flush=True)
    entries = _remote_zip_entries(url)
    rows: list[dict] = []
    for entry in entries:
        name = entry["name"]
        if is_junk(name) or entry["csize"] <= 0:
            continue
        kind, months = _covered_months(name)
        if kind is None or not months:
            continue
        if kinds is not None and kind not in kinds:
            continue
        if wanted is not None:
            months = {month for month in months if month in wanted}
        if not months:
            continue
        seen[kind].update(months)
        if _already_materialised(cfg, name, kind, months):
            for month in sorted(months):
                path = _month_file(cfg, month, kind)
                if path is not None:
                    rows.append(_row(month, kind, path.name, path.stat().st_size, "already present"))
            continue
        print(
            f"  downloading {PurePosixPath(name).name} ({entry['csize'] / 1e6:.1f} MB compressed)",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="bicimad-member-") as tmp:
            payload = Path(tmp) / "payload"
            inflated = Path(tmp) / "inflated"
            _download_member(url, entry, payload)
            _inflate_member(payload, inflated, entry)
            placed = _place_file(inflated, name, cfg, wanted, depth=0, kinds=kinds)
            if not placed:
                raise RuntimeError(f"Unpacked {name} but it contained no trip or status file.")
            rows.extend(placed)
    return rows


def _covered_months(name: str) -> tuple[str | None, set[str]]:
    kind = kind_of(name)
    if kind is None or is_junk(name):
        return None, set()
    month = month_of(name)
    if month is not None:
        return kind, {month}
    span = bundle_months(name)
    if span:
        return kind, span
    return kind, set()


def _already_materialised(cfg: ProjectConfig, name: str, kind: str, months: set[str]) -> bool:
    """True when every month this member would produce is already on disk.

    June 2021 is both a JSON and a CSV. A CSV still downloads when only the
    JSON is present, and the other way around. ``Bicimad_Estacions_201809``
    and ``Bicimad_Stations_201809`` are the same snapshot under two spellings,
    so a same-suffix file for that month counts as present.
    """
    if bundle_months(name) is not None:
        return all(_month_file(cfg, month, kind) is not None for month in months)
    predicted = predicted_output_name(name)
    if predicted is None:
        return False
    folder = cfg.trips_raw if kind == "trips" else cfg.status_raw
    dest = folder / predicted
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    if len(months) != 1:
        return False
    existing = _month_file(cfg, next(iter(months)), kind)
    return existing is not None and existing.suffix.lower() == Path(predicted).suffix.lower()


def _place_file(
    path: Path,
    entry_name: str,
    cfg: ProjectConfig,
    wanted: set[str] | None,
    depth: int,
    kinds: set[str] | None = None,
) -> list[dict]:
    if depth > 4:
        raise RuntimeError(f"Archive nesting is deeper than expected at {entry_name}")
    if is_junk(entry_name) or not path.is_file() or path.stat().st_size <= 0:
        return []
    suffix = PurePosixPath(entry_name).suffix.lower()
    if suffix == ".zip" or _looks_like_zip(path):
        return _place_zip(path, cfg, wanted, depth + 1, kinds)
    if suffix == ".rar" or _looks_like_rar(path):
        return _place_rar(path, entry_name, cfg, wanted, depth + 1, kinds)
    kind = kind_of(entry_name)
    month = month_of(entry_name)
    if kind is None or month is None:
        return []
    if kinds is not None and kind not in kinds:
        return []
    if wanted is not None and month not in wanted:
        return []
    folder = cfg.trips_raw if kind == "trips" else cfg.status_raw
    dest = folder / PurePosixPath(entry_name).name
    if dest.is_file() and dest.stat().st_size > 0:
        return [_row(month, kind, dest.name, dest.stat().st_size, "already present")]
    folder.mkdir(parents=True, exist_ok=True)
    _move_into_place(path, dest)
    print(f"    wrote {kind}/{dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return [_row(month, kind, dest.name, dest.stat().st_size, "downloaded")]


def _place_zip(
    path: Path,
    cfg: ProjectConfig,
    wanted: set[str] | None,
    depth: int,
    kinds: set[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or is_junk(info.filename):
                continue
            kind, months = _covered_months(info.filename)
            if kinds is not None and kind is not None and kind not in kinds:
                continue
            if (
                kind is not None
                and months
                and _already_materialised(cfg, info.filename, kind, months)
            ):
                for month in sorted(months):
                    existing = _month_file(cfg, month, kind)
                    if existing is not None:
                        rows.append(
                            _row(month, kind, existing.name, existing.stat().st_size, "already present")
                        )
                continue
            with tempfile.TemporaryDirectory(prefix="bicimad-inner-") as tmp:
                extracted = Path(tmp) / PurePosixPath(info.filename).name
                with archive.open(info) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                rows.extend(_place_file(extracted, info.filename, cfg, wanted, depth, kinds))
    return rows


def _place_rar(
    path: Path,
    entry_name: str,
    cfg: ProjectConfig,
    wanted: set[str] | None,
    depth: int,
    kinds: set[str] | None = None,
) -> list[dict]:
    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError(
            "EMT published "
            f"{PurePosixPath(entry_name).name} as a RAR archive. "
            "Unpacking it needs bsdtar (`tar` on PATH)."
        )
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="bicimad-rar-") as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        proc = subprocess.run(
            [tar, "-xf", str(path), "-C", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"tar could not unpack {PurePosixPath(entry_name).name}: {detail}")
        for file in sorted(p for p in out.rglob("*") if p.is_file()):
            rows.extend(_place_file(file, file.name, cfg, wanted, depth, kinds))
    return rows


def _looks_like_zip(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(4) == b"PK\x03\x04"


def _looks_like_rar(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(4) == b"Rar!"


def _move_into_place(src: Path, dest: Path) -> None:
    try:
        src.replace(dest)
    except OSError:
        shutil.copy2(src, dest)
        src.unlink(missing_ok=True)


def _remote_zip_entries(url: str) -> list[dict]:
    total = _remote_size(url)
    tail_len = min(total, 80_000)
    tail = _fetch_range(url, total - tail_len, total - 1)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0:
        raise RuntimeError(f"No ZIP end-of-central-directory in {url}")
    if marker + 22 > len(tail):
        raise RuntimeError(f"Truncated ZIP directory in {url}")
    cd_size = int.from_bytes(tail[marker + 12 : marker + 16], "little")
    cd_offset = int.from_bytes(tail[marker + 16 : marker + 20], "little")
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise RuntimeError(f"{url} is a Zip64 archive, which this downloader does not read.")
    cd = _fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    return _parse_central_directory(cd)


def _parse_central_directory(cd: bytes) -> list[dict]:
    entries: list[dict] = []
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        flags, method, _, _, crc, csize, usize, name_len, extra_len, comment_len = (
            int.from_bytes(cd[pos + 8 : pos + 10], "little"),
            int.from_bytes(cd[pos + 10 : pos + 12], "little"),
            int.from_bytes(cd[pos + 12 : pos + 14], "little"),
            int.from_bytes(cd[pos + 14 : pos + 16], "little"),
            int.from_bytes(cd[pos + 16 : pos + 20], "little"),
            int.from_bytes(cd[pos + 20 : pos + 24], "little"),
            int.from_bytes(cd[pos + 24 : pos + 28], "little"),
            int.from_bytes(cd[pos + 28 : pos + 30], "little"),
            int.from_bytes(cd[pos + 30 : pos + 32], "little"),
            int.from_bytes(cd[pos + 32 : pos + 34], "little"),
        )
        local_off = int.from_bytes(cd[pos + 42 : pos + 46], "little")
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        if flags & 0x1:
            raise RuntimeError(f"Encrypted ZIP member {name} is not supported.")
        entries.append(
            {
                "name": name,
                "method": method,
                "crc": crc,
                "csize": csize,
                "usize": usize,
                "local_off": local_off,
            }
        )
        pos += 46 + name_len + extra_len + comment_len
    return entries


def _download_member(url: str, entry: dict, dest: Path) -> None:
    header = _fetch_range(url, entry["local_off"], entry["local_off"] + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local header for {entry['name']}")
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    data_off = entry["local_off"] + 30 + name_len + extra_len
    _fetch_range_to(url, data_off, data_off + entry["csize"] - 1, dest, label=PurePosixPath(entry["name"]).name)


def _inflate_member(src: Path, dest: Path, entry: dict) -> None:
    method = entry["method"]
    if method == 0:
        shutil.copyfile(src, dest)
    elif method == 8:
        decoder = zlib.decompressobj(-15)
        with src.open("rb") as inp, dest.open("wb") as out:
            while True:
                chunk = inp.read(1 << 20)
                if not chunk:
                    break
                out.write(decoder.decompress(chunk))
            out.write(decoder.flush())
    else:
        raise RuntimeError(f"Unsupported ZIP method {method} for {entry['name']}")
    size = dest.stat().st_size
    expected = entry["usize"]
    if expected and size != expected:
        raise RuntimeError(
            f"{entry['name']} inflated to {size} bytes, archive directory says {expected}."
        )


def _remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 206:
            raise RuntimeError(f"{url} did not honour a range request (HTTP {resp.status}).")
        total = _total_from_content_range(resp.headers.get("Content-Range", ""))
    if total <= 0:
        raise RuntimeError(f"Could not read the size of {url}")
    return total


def _fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        if resp.status != 206:
            raise RuntimeError(f"Range request for {url} returned HTTP {resp.status}.")
        data = resp.read()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"Short read from {url}: {len(data)} bytes, expected {expected}.")
    return data


def _fetch_range_to(url: str, start: int, end: int, dest: Path, label: str) -> None:
    expected = end - start + 1
    last_error: Exception | None = None
    for _attempt in (1, 2):
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"Range request for {label} returned HTTP {resp.status}.")
                got = 0
                next_mark = 20 * 1024 * 1024
                with dest.open("wb") as fh:
                    while got < expected:
                        chunk = resp.read(min(1 << 20, expected - got))
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if got >= next_mark:
                            print(f"    {label}: {got / 1e6:.0f}/{expected / 1e6:.0f} MB", flush=True)
                            next_mark += 20 * 1024 * 1024
            if got != expected:
                raise RuntimeError(f"Short read of {label}: {got} bytes, expected {expected}.")
            return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {label} from {url}: {last_error}") from last_error


def _total_from_content_range(header: str) -> int:
    # bytes 0-0/315346750
    if "/" not in header:
        return -1
    try:
        return int(header.rsplit("/", 1)[-1])
    except ValueError:
        return -1


def _read_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))
