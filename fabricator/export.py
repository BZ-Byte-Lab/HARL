from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

from fabricator.fabricator import EventRow


def write_csv(rows: list[EventRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(EventRow)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def write_jsonl(rows: list[EventRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    return path
