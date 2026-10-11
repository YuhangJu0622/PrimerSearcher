"""Add VERL agent routing and judge context to OpenSeeker Parquet files."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def enrich(path: Path) -> None:
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        prompt = row.get("prompt") or []
        questions = [message.get("content", "") for message in prompt if message.get("role") == "user"]
        if not questions:
            raise ValueError(f"row in {path} has no user question")
        row["agent_name"] = "openseeker_agent"
        extra_info = dict(row.get("extra_info") or {})
        extra_info["question"] = questions[-1]
        row["extra_info"] = extra_info

    enriched = pa.Table.from_pylist(rows)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(enriched, temporary, compression="zstd")
        check = pq.read_table(temporary)
        if check.num_rows != table.num_rows or "agent_name" not in check.column_names:
            raise RuntimeError(f"failed to validate enriched dataset {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    default_dir = Path(__file__).resolve().parents[1] / "data" / "openseeker_qa"
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=[default_dir / "train_rl.parquet", default_dir / "eval_rl.parquet"])
    args = parser.parse_args()
    for path in args.paths:
        enrich(path.resolve())
        print(f"enriched {path}")


if __name__ == "__main__":
    main()

