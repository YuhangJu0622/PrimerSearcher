"""
Offline scoring script: compare agent results against reference answers.

Usage:
    python agent/scripts/score.py --results agent/results.jsonl --ref agent/data/data_stage_1.jsonl
    # 启用 LLM 含义匹配
    python agent/scripts/score.py --results agent/results.jsonl --ref agent/data/data_stage_1.jsonl --use-llm
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison: lowercase, strip, convert integer floats."""
    answer = str(answer).strip().lower()
    try:
        num = float(answer)
        if num == int(num):
            answer = str(int(num))
    except (ValueError, OverflowError):
        pass
    return answer


def _build_llm_client():
    """Build OpenAI client and extra_body once, reused across threads.

    The judge model goes directly through DMX (no NoDesk gateway).
    """
    from openai import OpenAI

    base_url = os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1")
    api_key = os.getenv("DMX_API_KEY", "")
    model = os.getenv("JUDGE_MODEL", "qwen-plus")

    if not api_key:
        print("警告: 未设置 DMX_API_KEY, LLM 判定可能失败。")

    client = OpenAI(base_url=base_url, api_key=api_key)

    extra_body = None
    return client, model, extra_body


def is_semantically_correct(predicted: str, expected: str, client, model: str, extra_body) -> bool:
    prompt = f"""你需要作为一个公正的裁判，评估模型针对问题给出的预测答案是否与标准参考答案含义相同。

[预测答案]: {predicted}
[标准答案]: {expected}

判断规则：
1. 如果预测答案与标准答案表达了相同的核心含义、人物、地点或事实，视为正确。
2. 如果预测答案包含了多余的废话，只要核心答案存在且正确，依然视为正确。
3. 语言不同但含义相同（例如："美国"和"United States"）视为正确。
4. 缩写和全称对应视为正确。
5. 如果预测答案缺少核心信息，或者表达了不同的含义，视为错误。

请仔细思考，并仅回复 "YES" 或 "NO"。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            extra_body=extra_body,
        )
        result = response.choices[0].message.content.strip().upper()
        return "YES" in result
    except Exception as e:
        print(f"LLM 判定失败 (predicted={predicted!r}): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Score agent results against reference answers")
    parser.add_argument("--results", required=True, help="Path to results JSONL (must have 'id' and 'answer')")
    parser.add_argument("--ref", required=True, help="Path to reference JSONL (must have 'id' and 'answer')")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to check semantic correctness for exact-match failures")
    args = parser.parse_args()

    # Load reference answers
    ref = {}
    with open(args.ref, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "answer" in item:
                ref[item["id"]] = item["answer"]

    if not ref:
        print(f"No reference answers found in {args.ref}")
        return

    # Load results
    results = {}
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                results[item["id"]] = item
            except Exception:
                pass

    if not results:
        print(f"No results found in {args.results}")
        return

    exact_correct = 0
    semantic_correct = 0
    total = 0

    strict_wrong = []
    semantic_wrong = []
    saved_by_llm = []

    errors = []
    elapsed_list = []

    for qid in sorted(results.keys()):
        r = results[qid]
        if qid not in ref:
            continue

        total += 1
        predicted = normalize_answer(r["answer"])
        expected = normalize_answer(ref[qid])
        elapsed = r.get("elapsed", 0)
        elapsed_list.append(elapsed)

        if r.get("error"):
            errors.append({"id": qid, "error": r["error"]})

        if predicted == expected:
            exact_correct += 1
            semantic_correct += 1
        else:
            wrong_item = {"id": qid, "predicted": r["answer"], "expected": ref[qid]}
            strict_wrong.append(wrong_item)

    if args.use_llm and strict_wrong:
        client, model, extra_body = _build_llm_client()
        print(f"并行 LLM 判断中 ({len(strict_wrong)} 道题)...")

        def _judge(item):
            qid = item["id"]
            ok = is_semantically_correct(item["predicted"], item["expected"], client, model, extra_body)
            return qid, item, ok

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_judge, w): w for w in strict_wrong}
            for future in as_completed(futures):
                qid, item, ok = future.result()
                if ok:
                    semantic_correct += 1
                    saved_by_llm.append(item)
                    print(f"  Q{qid}: ✓ LLM 判定正确")
                else:
                    semantic_wrong.append(item)
                    print(f"  Q{qid}: ✗ LLM 判定错误")

    # Print report
    print(f"\n{'=' * 60}")
    print(f"Results: {args.results}")
    print(f"Reference: {args.ref}")
    print(f"{'=' * 60}")

    if total > 0:
        print(f"\n[Exact Match] Accuracy: {exact_correct}/{total} = {exact_correct / total:.2%}")

        if args.use_llm:
            print(f"[Semantic]    Accuracy: {semantic_correct}/{total} = {semantic_correct / total:.2%}")
            saved_count = semantic_correct - exact_correct
            print(f"              (LLM saved {saved_count} answers that failed exact match)")
    else:
        print("\nNo matched questions to score.")
        return

    if elapsed_list:
        avg = sum(elapsed_list) / len(elapsed_list)
        print(f"\nAvg time: {avg:.1f}s  |  Total time: {sum(elapsed_list):.0f}s")

    if args.use_llm:
        if semantic_wrong:
            print(f"\n--- 真正的错题 (Semantic Wrong: {len(semantic_wrong)}) ---")
            for w in semantic_wrong:
                print(f"  Q{w['id']}: predicted={w['predicted']!r}  expected={w['expected']!r}")

        if saved_by_llm:
            print(f"\n--- 被 LLM 挽救的题目 (Strict Wrong but Semantic Correct: {len(saved_by_llm)}) ---")
            for w in saved_by_llm:
                print(f"  Q{w['id']}: predicted={w['predicted']!r}  expected={w['expected']!r}")
    else:
        if strict_wrong:
            print(f"\n--- Wrong answers ({len(strict_wrong)}) ---")
            for w in strict_wrong:
                print(f"  Q{w['id']}: predicted={w['predicted']!r}  expected={w['expected']!r}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  Q{e['id']}: {e['error']}")


if __name__ == "__main__":
    main()
