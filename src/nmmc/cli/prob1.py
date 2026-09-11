"""问题1 命令行入口。

用法：
    uv run prob1
    uv run prob1 --input docs/附件/附件1.xlsx --output-dir results/q1
    uv run prob1 --s0-fixed 6000        # 钉住 S_0 做 Z^6000 对照
"""
import argparse
from pathlib import Path

import numpy as np

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.model.lp import S_MAX, S_MIN, T
from nmmc.model.q1 import Solution, solve
from nmmc.report.plot import DataOverview, DualEconomics, S0Basin, SchedulePlot
from nmmc.report.q1 import (TEMPLATE, DetailsCsv, Extractor, Result1Xlsx,
                            SummaryMd, diagnostics, table1_rows, table2_rows)


def build_parser() -> argparse.ArgumentParser:
    """构建 prob1 命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="C 题问题1：典型日计划购电 LP")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径")
    parser.add_argument("--template", type=Path, default=TEMPLATE,
                        help="result1.xlsx 官方模板路径")
    parser.add_argument("--output-dir", type=Path, default=Path("results/q1"),
                        help="结果输出目录")
    parser.add_argument("--s0-fixed", type=float, default=None,
                        help="钉住 S_0 的取值（缺省则 S_0 自由决策）")
    return parser


def print_solution(sol: Solution) -> None:
    """向控制台打印论文表 1 / 表 2 结果。

    Args:
        sol: LP 最优解。
    """
    print("== 表1 指定时段购电量 (kWh) ==")
    for label, val in table1_rows(sol):
        print(f"  {label:>12}  {val:12.2f}")
    print("== 表2 充放电量 (kWh, 电池侧) ==")
    for label, ch, dis in table2_rows(sol):
        print(f"  {label:>10}  充 {ch:10.2f}  放 {dis:10.2f}")
    print(f"  {'0:00 储电量':>10}  {sol.S[0]:10.2f}")
    print(f"  {'24:00 储电量':>10}  {sol.S[T]:10.2f}")


def main(argv: list[str] | None = None) -> None:
    """prob1 主流程：加载 → 求解 → 打印诊断与对照 → 依次运行导出器。

    Args:
        argv: 命令行参数；缺省取 sys.argv。
    """
    args = build_parser().parse_args(argv)
    day = load_day1(args.input)
    sol = solve(day.p, day.ell, day.r, s0_fixed=args.s0_fixed)

    print_solution(sol)
    diag = diagnostics(sol, day)
    print("== 诊断 ==")
    for key, val in diag.items():
        print(f"  {key}: {val}")

    baselines: dict[str, float] = {}
    print("== 对照 ==")
    print(f"  Z* = {sol.cost:.2f} 元   S_0 = {sol.S[0]:.2f} kWh")
    if args.s0_fixed is None:
        sol6000 = solve(day.p, day.ell, day.r, s0_fixed=6000.0)
        low, high = (float(q) for q in np.quantile(day.p, (0.35, 0.65)))
        rule = solve(day.p, day.ell, day.r, rule_thresholds=(low, high))
        baselines = {
            "Z^6000": sol6000.cost,
            "S_0 决策化收益": sol6000.cost - sol.cost,
            "Z^rule": rule.cost,
            "规则节省比例": (rule.cost - sol.cost) / rule.cost * 100.0,
        }
        print(f"  Z^6000 = {sol6000.cost:.2f} 元   "
              f"S_0 决策化收益 = {baselines['S_0 决策化收益']:.2f} 元")
        print(f"  Z^rule = {rule.cost:.2f} 元   "
              f"优化相对规则节省 = {baselines['规则节省比例']:.2f}%")
        for delta in (-200.0, 200.0):  # S_0 平坦性探测
            probe = solve(day.p, day.ell, day.r, s0_fixed=sol.S[0] + delta)
            print(f"  S_0 钉在 {sol.S[0] + delta:.0f} 时 Z = {probe.cost:.2f} 元")
        # S_0 敏感性扫描（供盆地图）
        s0_grid = np.arange(S_MIN, S_MAX + 1.0, 200.0)
        if abs((sol.S[0] - S_MIN) % 200.0) > 1e-6:
            s0_grid = np.sort(np.append(s0_grid, sol.S[0]))
        s0_costs = [solve(day.p, day.ell, day.r, s0_fixed=float(v)).cost
                    for v in s0_grid]
        print(f"  S_0 敏感性扫描：{len(s0_grid)} 个点，"
              f"费用区间 [{min(s0_costs):.2f}, {max(s0_costs):.2f}] 元")
        s0_basin = S0Basin(s0_grid, s0_costs, float(sol.S[0]), sol.cost,
                           sol6000.cost)
    else:
        s0_basin = None

    extractors: list[Extractor] = [
        Result1Xlsx(args.template),
        DetailsCsv(),
        SchedulePlot(),
        DataOverview(),
        DualEconomics(),
        SummaryMd(diag, baselines),
    ]
    if s0_basin is not None:
        extractors.append(s0_basin)
    for ex in extractors:
        print(f"已导出 {ex.extract(day, sol, args.output_dir)}")


if __name__ == "__main__":
    main()
