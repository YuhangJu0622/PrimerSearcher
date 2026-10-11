"""Fail-fast checks before an expensive OpenSeeker GRPO run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


RL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-30B-A3B-Thinking-2507"
)
DEFAULT_ADAPTER = Path(
    "/root/autodl-tmp/output/qwen3_30b_lora_2gpu_rtx6000_122k_exp6/"
    "v0-20260802-172409/checkpoint-440"
)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {description}: {path}")


def check_dataset(path: Path, expected_data_source: str) -> None:
    table = pq.read_table(
        path,
        columns=["data_source", "agent_name", "prompt", "reward_model", "extra_info"],
    )
    if not table.num_rows:
        raise SystemExit(f"dataset is empty: {path}")
    for index, row in enumerate(table.to_pylist()):
        if (
            row["data_source"] != expected_data_source
            or row["agent_name"] != "openseeker_agent"
        ):
            raise SystemExit(f"bad routing fields in {path} row {index}")
        if not row["reward_model"].get("ground_truth"):
            raise SystemExit(f"missing ground truth in {path} row {index}")
        prompt = row["prompt"]
        roles = [message.get("role") for message in prompt]
        if len(prompt) != 2 or roles != ["system", "user"]:
            raise SystemExit(
                f"bad initial role structure in {path} row {index}: {roles!r}"
            )
        if not all(
            isinstance(message.get("content"), str) and message["content"].strip()
            for message in prompt
        ):
            raise SystemExit(f"empty prompt content in {path} row {index}")
        question = row["extra_info"].get("question")
        if not question or question != row["prompt"][-1]["content"]:
            raise SystemExit(f"judge question mismatch in {path} row {index}")


def check_adapter(path: Path) -> None:
    config_path = path / "adapter_config.json"
    require_path(config_path, "LoRA adapter config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise SystemExit("LoRA adapter rank/alpha does not match the RL configuration")
    if set(config.get("target_modules", [])) != expected_modules:
        raise SystemExit(
            "LoRA adapter target_modules do not match the RL configuration"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(os.getenv("LORA_ADAPTER_PATH", DEFAULT_ADAPTER)),
    )
    parser.add_argument("--require-judge", action="store_true")
    parser.add_argument("--require-search", action="store_true")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=RL_ROOT / "data/deepdive_qa/train_rl.parquet",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=RL_ROOT / "data/deepdive_qa/eval_rl.parquet",
    )
    parser.add_argument("--data-source", default="deepdive_qa")
    args = parser.parse_args()

    require_path(args.model, "base model")
    require_path(args.adapter, "SFT adapter")
    check_adapter(args.adapter)
    check_dataset(args.train_file, args.data_source)
    check_dataset(args.val_file, args.data_source)

    judge_ready = all(
        (
            os.getenv("JUDGE_BASE_URL") or os.getenv("TOOL_BASE_URL"),
            os.getenv("JUDGE_API_KEY") or os.getenv("TOOL_API_KEY"),
            os.getenv("JUDGE_MODEL") or os.getenv("SUMMARY_MODEL"),
        )
    )
    judge_enabled = os.getenv("OPENSEEKER_JUDGE_ENABLED", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    search_ready = bool(os.getenv("SERPER_API_KEYS"))
    if args.require_judge and not (judge_ready and judge_enabled):
        raise SystemExit(
            "LLM judge is required but its enable flag or endpoint credentials are missing"
        )
    if args.require_search and not search_ready:
        raise SystemExit(
            "deep search is required but SERPER_API_KEYS is not set"
        )

    print("preflight ok")
    print(f"judge_ready={judge_ready and judge_enabled}")
    print(f"search_ready={search_ready}")


if __name__ == "__main__":
    main()
