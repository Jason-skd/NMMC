"""问题1 命令行入口。

用法：
    uv run prob1
    uv run prob1 --input docs/附件/附件1.xlsx --output-dir results/q1
    uv run prob1 --s0-fixed 6000        # 钉住 S_0 做 Z^6000 对照
"""
import argparse
from pathlib import Path

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.model.lp import T
from nmmc.model.q1 import Solution
from nmmc.report.plot.q1 import (DataOverview, DualEconomics, S0Basin,
                                 SchedulePlot)
from nmmc.report.q1 import (TEMPLATE, DetailsCsv, Exporter, Result1Xlsx,
                            SummaryMd, diagnostics, table1_rows, table2_rows)
from nmmc.run.q1 import run_q1


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
    """prob1 主流程：run_q1 求解与对照 → 打印诊断 → 依次运行导出器。

    Args:
        argv: 命令行参数；缺省取 sys.argv。
    """
    args = build_parser().parse_args(argv)
    day = load_day1(args.input)
    outcome = run_q1(day, s0_fixed=args.s0_fixed)
    sol = outcome.sol

    print_solution(sol)
    diag = diagnostics(sol, day)
    print("== 诊断 ==")
    for key, val in diag.items():
        print(f"  {key}: {val}")

    print("== 对照 ==")
    print(f"  Z* = {sol.cost:.2f} 元   S_0 = {sol.S[0]:.2f} kWh")
    s0_basin = None
    if args.s0_fixed is None:
        b = outcome.baselines
        print(f"  Z^6000 = {b['Z^6000']:.2f} 元   "
              f"S_0 决策化收益 = {b['S_0 决策化收益']:.2f} 元")
        print(f"  Z^rule = {b['Z^rule']:.2f} 元   "
              f"优化相对规则节省 = {b['规则节省比例']:.2f}%")
        for s0, cost in outcome.s0_probes:
            print(f"  S_0 钉在 {s0:.0f} 时 Z = {cost:.2f} 元")
        print(f"  S_0 敏感性扫描：{len(outcome.s0_grid)} 个点，"
              f"费用区间 [{min(outcome.s0_costs):.2f}, "
              f"{max(outcome.s0_costs):.2f}] 元")
        s0_basin = S0Basin(outcome.s0_grid, outcome.s0_costs,
                           float(sol.S[0]), sol.cost, b["Z^6000"])

    exporters: list[Exporter] = [
        Result1Xlsx(args.template),
        DetailsCsv(),
        SchedulePlot(),
        DataOverview(),
        DualEconomics(),
        SummaryMd(diag, outcome.baselines),
    ]
    if s0_basin is not None:
        exporters.append(s0_basin)
    for ex in exporters:
        print(f"已导出 {ex.export(day, sol, args.output_dir)}")


if __name__ == "__main__":
    main()
