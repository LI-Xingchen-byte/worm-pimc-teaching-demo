"""Summarize the predeclared P1 runs; never discard a failed or missing seed.

python -m validation.report_p1 --root runs/validation/ideal
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .run_p1 import NAMES, analyze


SEEDS = (2718, 3141, 5772)
GROUPS = ("c0.5", "c1", "c2", "asym")


def build_report(root: Path) -> dict:
    report = {"runs": {}, "groups": {}, "missing": [], "incomplete": []}
    for group in GROUPS:
        collected = []
        for seed in SEEDS:
            name = f"{group}_s{seed}"
            path = root / name / "result.json"
            if not path.exists():
                report["missing"].append(name)
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            series = np.load(path.parent / "series.npy", allow_pickle=False)
            if hashlib.sha256(series.tobytes()).hexdigest() != result["series_sha256"]:
                raise ValueError(f"series hash mismatch: {name}")
            comparisons = analyze(series, result["reference"]["values"])
            if result["status"] != "COMPLETED":
                report["incomplete"].append(name)
            # n_effective in run_p1 is opportunity-equivalent. For an observable
            # on Z states, additionally show the usual Var_Z(f)/SE² definition.
            for column, observable in enumerate(NAMES, 1):
                row = comparisons[observable]
                values = series[series[:, 0] == 1, column]
                se = row.get("blocking", {}).get("standard_error")
                row["Z_effective_samples"] = (
                    min(float(len(values)), float(np.var(values, ddof=1)) / se**2)
                    if se and len(values) >= 2
                    else None
                )
            diagnostic = {
                "seed": seed,
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "attempts_per_second": result["attempts_per_second"],
                "opportunities": len(series),
                "Z_count": int(series[:, 0].sum()),
                "Z_fraction": result["Z_fraction"],
                "comparisons": comparisons,
                "half_chain_means": [
                    {
                        observable: (
                            float(part[:, i].sum() / part[:, 0].sum())
                            if part[:, 0].sum()
                            else None
                        )
                        for i, observable in enumerate(NAMES, 1)
                    }
                    for part in np.array_split(series, 2)
                ],
            }
            report["runs"][name] = diagnostic
            collected.append(diagnostic)
        aggregate = {}
        for observable in NAMES:
            rows = [run["comparisons"][observable] for run in collected]
            if len(rows) != len(SEEDS) or any(
                row.get("estimate") is None for row in rows
            ):
                aggregate[observable] = {"status": "INCOMPLETE"}
                continue
            values = np.array([row["estimate"] for row in rows])
            errors = [row.get("blocking", {}).get("standard_error") for row in rows]
            aggregate[observable] = {
                "equal_seed_mean": float(values.mean()),
                "reference": rows[0]["reference"],
                "within_chain_SE": (
                    math.sqrt(sum(se * se for se in errors)) / len(rows)
                    if all(se is not None for se in errors)
                    else None
                ),
                "between_seed_SE": float(values.std(ddof=1) / math.sqrt(len(rows))),
                "individual_statuses": [row["status"] for row in rows],
            }
        report["groups"][group] = aggregate
    report["comparison_counts"] = dict(
        Counter(
            row["status"]
            for run in report["runs"].values()
            for row in run["comparisons"].values()
        )
    )
    report["total_elapsed_seconds"] = sum(
        run["elapsed_seconds"] for run in report["runs"].values()
    )
    return report


def write_markdown(report: dict, path: Path):
    lines = [
        "# P1 数据汇总",
        "",
        "每条链采用预先规定的 4 SE 筛查。平台不足保持 INCONCLUSIVE。",
        "跨 seed 的均值仅作为描述，不能替代每条链的检验。SE 为 blocking 标准误差。",
        "",
        "| Run | Z 比例 | N ± SE | E ± SE | 交换粒子数 ± SE | 结果计数 |",
        "|---|---:|---:|---:|---:|---|",
    ]

    def show(row):
        value = row.get("estimate")
        se = row.get("blocking", {}).get("standard_error")
        if value is None:
            return "—"
        return (
            f"{value:.4f} ± {se:.4f}" if se is not None else f"{value:.4f} ± 未达平台"
        )

    for name, run in report["runs"].items():
        rows = run["comparisons"]
        counts = dict(Counter(row["status"] for row in rows.values()))
        lines.append(
            f"| {name} | {run['Z_fraction']:.3f} | {show(rows['N'])} | {show(rows['E'])} | "
            f"{show(rows['exchange_particles'])} | {counts} |"
        )
    lines.extend(
        [
            "",
            f"全部比较：{report['comparison_counts']}",
            f"缺失：{report['missing']}；未完成：{report['incomplete']}",
            f"各链耗时之和：{report['total_elapsed_seconds']:.1f} 秒（不是并行墙钟时间）。",
            "",
            "完整每个 observable、逐层 blocking、前后半链均值、有效 Z 样本数见 summary.json。",
            "原始 result.json 的 n_effective 是量测机会等效数量；summary.json 另给出",
            "Z_effective_samples=Var_Z(f)/SE²，便于与物理独立样本数比较。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    (args.root / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_markdown(report, args.root / "summary.md")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "comparison_counts",
                    "missing",
                    "incomplete",
                    "total_elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
