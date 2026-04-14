#!/usr/bin/env python3
"""Summarize Jittor evaluation results into readable tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize Jittor evaluation results")
    parser.add_argument("--result-json", nargs="+", required=True, type=Path)
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--metrics", nargs="+", type=str)
    return parser.parse_args()


def infer_metric_order(data: dict, preferred_metrics: list[str] | None) -> list[str]:
    if preferred_metrics:
        return preferred_metrics
    metric_names = []
    for dataset_metrics in data.values():
        for key in dataset_metrics.keys():
            if key not in metric_names:
                metric_names.append(key)
    return metric_names


def compute_mean_row(data: dict, metric_names: list[str]) -> dict[str, float]:
    mean_row = {}
    num_rows = max(len(data), 1)
    for metric_name in metric_names:
        total = sum(float(dataset_metrics.get(metric_name, 0.0)) for dataset_metrics in data.values())
        mean_row[metric_name] = total / num_rows
    return mean_row


def render_text_table(title: str, rows: list[dict], metric_names: list[str]) -> str:
    headers = ["dataset", *metric_names]
    table_rows = [[row["dataset"], *[f"{float(row.get(metric, 0.0)):.3f}" for metric in metric_names]] for row in rows]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for idx, cell in enumerate(table_row):
            widths[idx] = max(widths[idx], len(cell))

    def _render_line(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    parts = [title, _render_line(headers), "-+-".join("-" * width for width in widths)]
    parts.extend(_render_line(row) for row in table_rows)
    return "\n".join(parts)


def render_markdown_table(title: str, rows: list[dict], metric_names: list[str]) -> str:
    headers = ["dataset", *metric_names]
    separator = ["---"] * len(headers)
    lines = [f"### {title}", "| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for row in rows:
        values = [row["dataset"], *[f"{float(row.get(metric, 0.0)):.3f}" for metric in metric_names]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_summary(result_path: Path, preferred_metrics: list[str] | None) -> dict:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    metric_names = infer_metric_order(data, preferred_metrics)
    mean_row = compute_mean_row(data, metric_names)
    rows = [{"dataset": dataset_name, **dataset_metrics} for dataset_name, dataset_metrics in data.items()]
    rows.append({"dataset": "mean", **mean_row})
    return {
        "run_name": result_path.parent.name,
        "result_json": str(result_path.resolve()),
        "metric_names": metric_names,
        "rows": rows,
    }


def render_output(summaries: list[dict], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(summaries, indent=2, ensure_ascii=False)

    blocks = []
    for summary in summaries:
        title = f"{summary['run_name']} ({summary['result_json']})"
        if fmt == "markdown":
            blocks.append(render_markdown_table(title, summary["rows"], summary["metric_names"]))
        else:
            blocks.append(render_text_table(title, summary["rows"], summary["metric_names"]))
    return "\n\n".join(blocks)


def main() -> int:
    args = parse_args()
    summaries = [build_summary(result_path, args.metrics) for result_path in args.result_json]
    output = render_output(summaries, args.format)
    print(output)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
