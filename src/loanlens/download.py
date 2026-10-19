"""Acquire raw HMDA LAR CSVs from the official Data Browser API, or register
files you downloaded manually. Raw files are written once and made read-only.

Usage:
    python -m loanlens.download                       # fetch both years (needs network access to ffiec.cfpb.gov)
    python -m loanlens.download --register 2023 PATH --retrieved 2026-10-16
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import stat
import urllib.request
from pathlib import Path

from .common import CFG, path, raw_file, read_json, sha256, write_json

PROVENANCE = lambda: path("raw_dir") / "provenance.json"  # noqa: E731


def build_url(year: int) -> str:
    s = CFG["source"]
    acts = ",".join(str(a) for a in s["actions_taken"])
    # Documented parameters: years (required), states (one geographic filter), actions_taken (HMDA filter)
    return f"{s['api_base']}?states={s['state']}&years={year}&actions_taken={acts}"


def _record(year: int, fp: Path, url: str, retrieved: str, method: str) -> None:
    prov = read_json(PROVENANCE()) if PROVENANCE().exists() else {"files": {}}
    prov["publisher"] = CFG["source"]["publisher"]
    prov["api_docs"] = CFG["source"]["api_docs"]
    prov["field_docs"] = CFG["source"]["field_docs"]
    prov["terms_note"] = (
        "Public HMDA data published by the CFPB/FFIEC under the Home Mortgage Disclosure Act. "
        "Records are modified for privacy before release. Cite as: FFIEC / CFPB, HMDA Loan Application "
        "Register (public), Data Browser query: <url>, retrieved <date>."
    )
    prov["files"][str(year)] = {
        "file": fp.name,
        "query_url": url,
        "retrieved": retrieved,
        "method": method,
        "sha256": sha256(fp),
        "bytes": fp.stat().st_size,
        "snapshot_note": "Data Browser serves the current published data, which can change on resubmission; "
                         "the checksum pins exactly what was used.",
    }
    write_json(prov, PROVENANCE())


def _lock(fp: Path) -> None:
    os.chmod(fp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def fetch(year: int) -> Path:
    fp = raw_file(year)
    if fp.exists():
        print(f"[skip] {fp.name} already present (raw files are immutable)")
        return fp
    url = build_url(year)
    tmp = fp.with_suffix(".part")
    print(f"[get] {url}")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    tmp.rename(fp)
    _lock(fp)
    _record(year, fp, url, dt.date.today().isoformat(), "automated download")
    return fp


def register(year: int, src: Path, retrieved: str) -> Path:
    fp = raw_file(year)
    if fp.exists():
        if sha256(fp) == sha256(src):
            print(f"[ok] {fp.name} already registered with identical checksum")
            return fp
        raise SystemExit(f"{fp} exists with a different checksum; raw files are immutable. Remove it deliberately first.")
    shutil.copyfile(src, fp)
    _lock(fp)
    _record(year, fp, build_url(year), retrieved, f"manual browser download of the query URL (source file: {src.name})")
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", nargs=2, metavar=("YEAR", "PATH"))
    ap.add_argument("--retrieved", default=dt.date.today().isoformat(), help="date the manual download happened")
    a = ap.parse_args()
    if a.register:
        fp = register(int(a.register[0]), Path(a.register[1]), a.retrieved)
        print(f"[registered] {fp}")
    else:
        for y in (CFG["source"]["train_year"], CFG["source"]["test_year"]):
            fetch(y)


if __name__ == "__main__":
    main()
