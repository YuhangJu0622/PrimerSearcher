"""
Concurrent answer generation script for Research Agent.
Runs questions from a JSONL data file and writes answers to a results JSONL file.

Usage:
    python agent/scripts/run_agent.py --data agent/data/data_stage_1.jsonl
    python agent/scripts/run_agent.py --data agent/data/data_stage_1.jsonl --start 0 --end 30 --concurrency 3
    nohup python agent/scripts/run_agent.py --data agent/data/data_stage_1.jsonl --start 0 --end 50 --concurrency 5 > agent/output.log 2>&1 &

With --output-dir, logs and result.jsonl are written into
<output-dir>/logs/<run-name>/ (run name defaults to a timestamp; resume by
passing the same --output-dir and --run-name again). Without it, legacy behavior.
    python agent/scripts/run_agent.py --data agent/data/data_stage_1.jsonl --output-dir agent/runs --run-name test1
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    On Windows a *redirected* stream defaults to the ANSI codepage (GBK on
    Chinese locales).  Agent answers are scraped from the web and can contain
    CJK, emoji and other characters GBK cannot encode, so `print(...)` would
    raise UnicodeEncodeError and kill the whole `asyncio.gather` run.  UTF-8
    with errors="replace" makes every print safe.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # Not a reconfigurable TextIOWrapper (e.g. captured by a test
            # runner); leave it untouched.
            pass


_force_utf8_stdio()

from agent.agent_loop import react_agent
from agent.logger_config import current_qid, setup_eval_logging


class TrajectoryRecorder:
    """Atomically maintain one trajectory JSON file for a question."""

    def __init__(self, qid: int, question: str, path: Path):
        self.qid = qid
        self.question = question
        self.path = path
        self.started_at = time.time()
        self.messages = []

    def _write(
        self,
        *,
        status: str,
        answer: str = None,
        error: str = None,
    ) -> None:
        payload = {
            "id": self.qid,
            "question": self.question,
            "status": status,
            "answer": answer,
            "error": error,
            "elapsed": round(time.time() - self.started_at, 1),
            "messages": self.messages,
        }
        temp_path = self.path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, self.path)

    def checkpoint(self, messages: list) -> None:
        self.messages = messages
        self._write(status="running")

    def complete(self, answer: str) -> None:
        self._write(status="completed", answer=answer)

    def fail(self, error: str) -> None:
        self._write(status="error", error=error)


async def evaluate_single(
    qid: int,
    question: str,
    trajectory_path: Path,
) -> dict:
    """Run the agent on a single question, returning a result dict."""
    current_qid.set(qid)
    recorder = TrajectoryRecorder(qid, question, trajectory_path)
    try:
        answer = await react_agent(
            question,
            trajectory_callback=recorder.checkpoint,
        )
        elapsed = time.time() - recorder.started_at
        recorder.complete(answer)
        return {"id": qid, "answer": answer, "elapsed": round(elapsed, 1), "error": None}
    except Exception as e:
        elapsed = time.time() - recorder.started_at
        error = str(e)
        recorder.fail(error)
        return {"id": qid, "answer": "", "elapsed": round(elapsed, 1), "error": error}


async def main():
    parser = argparse.ArgumentParser(description="Run Research Agent on evaluation data")
    parser.add_argument("--data", required=True, help="Path to input JSONL (must have 'id' and 'question' fields)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder: <output-dir>/logs/<run-name>/ holds logs, trajectories/ and result.jsonl; resume reads result.jsonl from there. If omitted, legacy paths are used.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name under <output-dir>/logs/ (defaults to a timestamp). Reuse the same name to resume a run.",
    )
    parser.add_argument("--output", default=None, help="Path to output results JSONL (ignored when --output-dir is set)")
    parser.add_argument("--start", type=int, default=0, help="Start question ID (inclusive)")
    parser.add_argument("--end", type=int, default=100, help="End question ID (exclusive)")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent questions")
    args = parser.parse_args()

    if args.output_dir:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir).resolve() / "logs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(run_dir / "result.jsonl")
        log_dir = str(run_dir)
    else:
        if args.output is None:
            args.output = str(Path(__file__).resolve().parents[1] / "results.jsonl")
        agent_dir = Path(__file__).resolve().parents[1]
        log_dir = str(agent_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    # Setup per-question log files
    per_q_handler = setup_eval_logging(log_dir)
    trajectory_dir = Path(log_dir) / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs directory: {log_dir}")
    print(f"Trajectories directory: {trajectory_dir}")

    # Load questions
    data = {}
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item

    # Load existing results to support resume
    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    ids_to_process = [i for i in range(args.start, args.end) if i in data and i not in done_ids]
    skipped = len([i for i in range(args.start, args.end) if i in data and i in done_ids])

    print(f"Processing {len(ids_to_process)} questions (IDs {args.start}-{args.end - 1})")
    if skipped:
        print(f"Skipping {skipped} already completed")

    if not ids_to_process:
        print("No new questions to process.")
        per_q_handler.close()
        return

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    total = len(ids_to_process)

    async def run_one(qid: int) -> dict:
        nonlocal completed
        async with sem:
            result = await evaluate_single(
                qid,
                data[qid]["question"],
                trajectory_dir / f"question_{qid}.json",
            )
        async with write_lock:
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed += 1
            status = f"ERROR: {result['error']}" if result["error"] else result["answer"]
            print(f"[{completed}/{total}] Q{qid} ({result['elapsed']}s) -> {status}")
        return result

    tasks = [asyncio.create_task(run_one(qid)) for qid in ids_to_process]
    results = await asyncio.gather(*tasks)

    # Summary
    errors = sum(1 for r in results if r["error"])
    avg_elapsed = sum(r["elapsed"] for r in results) / len(results) if results else 0
    print(f"\nDone: {len(results)} questions, {errors} errors, avg {avg_elapsed:.1f}s per question")
    print(f"Results saved to: {args.output}")
    print(f"Logs saved to: {log_dir}/")

    per_q_handler.close()


if __name__ == "__main__":
    asyncio.run(main())
