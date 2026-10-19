"""Shared helpers: config loading, paths, checksums, JSON I/O."""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


CFG = load_config()


def path(key: str) -> Path:
    p = ROOT / CFG["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_file(year: int) -> Path:
    s = CFG["source"]
    return path("raw_dir") / f"hmda_{s['state']}_{year}_actions123.csv"


def sha256(fp: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_json(obj, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(fp: Path):
    with open(fp) as f:
        return json.load(f)
