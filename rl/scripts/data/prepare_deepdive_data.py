"""Convert DeepDive QA into the Parquet layout consumed by the RL agent loop."""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RL_ROOT / "data/deepdive_qa/deepdive_qa_rl.parquet"
DEFAULT_TRAIN = RL_ROOT / "data/deepdive_qa/train_rl.parquet"
DEFAULT_EVAL = RL_ROOT / "data/deepdive_qa/eval_rl.parquet"
DEFAULT_PROMPT_SOURCE = RL_ROOT / "data/openseeker_qa/train_rl.parquet"


def load_system_prompt(path: Path) -> str:
    """Reuse the production tool contract instead of maintaining a second copy."""
    table = pq.read_table(path, columns=["prompt"])
    if not table.num_rows:
        raise ValueError(f"system prompt source is empty: {path}")
    prompt = table.column("prompt")[0].as_py() or []
    if not prompt or prompt[0].get("role") != "system":
        raise ValueError(f"system prompt source has no initial system message: {path}")
    content = str(prompt[0].get("content", "")).strip()
    if not content:
        raise ValueError(f"system prompt source has an empty system message: {path}")
    return content


def convert_rows(rows: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for source_row in rows:
        question = str(source_row.get("question", "")).strip()
        answer = str(source_row.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"DeepDive row {source_row.get('id')!r} has an empty question or answer")
        source_id = source_row.get("id")
        if not isinstance(source_id, int):
            raise ValueError(f"DeepDive row has a non-integer id: {source_id!r}")
        converted.append(
            {
                "data_source": "deepdive_qa",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "reward_model": {"ground_truth": answer},
                "extra_info": {
                    "index": source_id,
                    "number_of_tool_calls": 0,
                    "question": question,
                },
                "agent_name": "openseeker_agent",
            }
        )
    return converted


def split_rows(rows: list[dict[str, Any]], eval_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between zero and one")
    if len(rows) < 2:
        raise ValueError("at least two rows are required to create train and eval splits")
    eval_count = max(1, min(len(rows) - 1, round(len(rows) * eval_fraction)))
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[eval_count:], shuffled[:eval_count]


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        if pq.read_table(temporary).num_rows != len(rows):
            raise RuntimeError(f"failed to validate generated dataset: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--system-prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    parser.add_argument("--eval-fraction", type=float, default=62 / (2744 + 62))
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    if args.train_output.resolve() == args.eval_output.resolve():
        raise ValueError("train and eval outputs must be different files")
    source_table = pq.read_table(args.input)
    converted = convert_rows(source_table.to_pylist(), load_system_prompt(args.system_prompt_source))
    train_rows, eval_rows = split_rows(converted, args.eval_fraction, args.seed)
    write_table(train_rows, args.train_output)
    write_table(eval_rows, args.eval_output)
    print(f"wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"wrote {len(eval_rows)} eval rows to {args.eval_output}")


if __name__ == "__main__":
    main()
