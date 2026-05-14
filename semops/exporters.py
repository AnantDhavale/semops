from dataclasses import asdict, is_dataclass
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _to_plain_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_value(item) for item in value]
    return value


def _to_row(record: Any) -> Dict[str, Any]:
    plain = _to_plain_value(record)
    if isinstance(plain, dict):
        return plain
    raise TypeError("records must be dicts or dataclass instances")


def _flatten(prefix: str, value: Any, flat: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child_prefix, item, flat)
        return
    if isinstance(value, list):
        flat[prefix] = json.dumps(value)
        return
    flat[prefix] = value


def to_rows(records: Iterable[Any]) -> List[Dict[str, Any]]:
    """Normalize records into plain dictionaries."""
    return [_to_row(record) for record in records]


def export_json(records: Iterable[Any], path: str | Path, indent: int = 2) -> Path:
    """Write records to JSON."""
    target = Path(path)
    rows = to_rows(records)
    target.write_text(json.dumps(rows, indent=indent), encoding="utf-8")
    return target


def export_csv(records: Iterable[Any], path: str | Path) -> Path:
    """Write records to CSV, flattening nested metadata."""
    target = Path(path)
    rows = to_rows(records)

    flat_rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    for row in rows:
        flat: Dict[str, Any] = {}
        for key, value in row.items():
            _flatten(str(key), value, flat)
        flat_rows.append(flat)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    return target


def export_parquet(records: Iterable[Any], path: str | Path) -> Path:
    """Write records to Parquet when pyarrow is installed."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install semops[parquet] to enable Parquet export") from exc

    target = Path(path)
    rows = to_rows(records)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, target)
    return target
