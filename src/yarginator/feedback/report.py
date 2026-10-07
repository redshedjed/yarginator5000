"""Reports written to ``<song>/reports/``.

- ``<part>.csv``: whatever rows a generator wants to expose (one per note, syllable, ...).
- ``review.csv``: flagged spots from every part. Re-running a part replaces only that part's rows.
  The REAPER step turns these into project markers so you can jump straight to them.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class ReviewItem:
    seconds: float
    part: str
    message: str
    severity: str = "warn"  # info | warn | error


def write_part_report(folder: Path, part: str, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    path = folder / f"{part}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def read_review(folder: Path) -> list[ReviewItem]:
    path = folder / "review.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [ReviewItem(float(r["seconds"]), r["part"], r["message"], r["severity"]) for r in csv.DictReader(f)]


def write_review(folder: Path, items: list[ReviewItem], replace_parts: set[str]) -> Path:
    """Merge ``items`` into review.csv, dropping old rows for ``replace_parts``."""
    keep = [i for i in read_review(folder) if i.part not in replace_parts]
    merged = sorted(keep + items, key=lambda i: (i.seconds, i.part))
    path = folder / "review.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(ReviewItem)])
        w.writeheader()
        w.writerows(asdict(i) | {"seconds": round(i.seconds, 3)} for i in merged)
    return path
