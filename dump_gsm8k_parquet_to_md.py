#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow.parquet as pq
from lighteval.tasks.registry import Registry


def extract_final_number(text: str) -> str | None:
    hash_match = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if hash_match:
        return hash_match[-1]

    number_match = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if number_match:
        return number_match[-1]

    return None


def build_markdown(parquet_path: Path, task_name: str) -> str:
    rows = pq.read_table(parquet_path).to_pylist()

    registry = Registry(tasks=task_name, custom_tasks=None, load_multilingual=False)
    task = next(iter(registry.load_tasks().values()))
    docs = task.eval_docs()

    lines: list[str] = []
    lines.append(f"# Task Dump")
    lines.append("")
    lines.append(f"- Parquet: `{parquet_path}`")
    lines.append(f"- Task: `{task_name}`")
    lines.append(f"- Samples: `{len(rows)}`")
    lines.append("")

    success_count = 0
    for i, row in enumerate(rows):
        doc = docs[int(row["sample_id"])]
        sample = row["sample"]
        texts = sample.get("text") or []
        pred = texts[0] if texts and texts[0] is not None else ""
        gold = doc.get_golds()[0]
        pred_num = extract_final_number(pred)
        gold_num = extract_final_number(gold)
        verdict = "OK" if pred_num is not None and pred_num == gold_num else "FAIL"
        if verdict == "OK":
            success_count += 1

        lines.append(f"## Sample {i:02d} - {verdict}")
        lines.append("")
        lines.append(f"- Predicted final number: `{pred_num}`")
        lines.append(f"- Gold final number: `{gold_num}`")
        lines.append("")
        lines.append("### Question")
        lines.append("")
        lines.append(doc.query.rstrip())
        lines.append("")
        lines.append("### Prediction")
        lines.append("")
        lines.append(pred.rstrip() if pred else "<EMPTY>")
        lines.append("")
        lines.append("### Gold")
        lines.append("")
        lines.append(gold.rstrip())
        lines.append("")

    lines.insert(4, f"- Exact final-number matches in dump: `{success_count}/{len(rows)}`")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump a lighteval parquet into readable Markdown.")
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument("--task", default="gsm8k|5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    markdown = build_markdown(args.parquet_path, args.task)
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
