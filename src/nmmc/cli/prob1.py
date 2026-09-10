"""问题1 命令行入口。

用法：
    uv run prob1
    uv run prob1 --input docs/附件/附件1.xlsx --output-dir results/q1
    uv run prob1 --s0-fixed 6000        # 钉住 S_0 做 Z^6000 对照
"""
import argparse
from pathlib import Path

from nmmc.data import ATTACH, load_day1
from nmmc.extract import TEMPLATE, Result1Xlsx, diagnostics, table1_rows, table2_rows
from nmmc.lp import T, Solution, solve


def build_parser() -> argparse.ArgumentParser:
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
    print("== 表1 指定时段购电量 (kWh) ==")
    for label, val in table1_rows(sol):
        print(f"  {label:>12}  {val:12.2f}")
    print("== 表2 充放电量 (kWh, 电池侧) ==")
    for label, ch, dis in table2_rows(sol):
        print(f"  {label:>10}  充 {ch:10.2f}  放 {dis:10.2f}")
    print(f"  {'0:00 储电量':>10}  {sol.S[0]:10.2f}")
    print(f"  {'24:00 储电量':>10}  {sol.S[T]:10.2f}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    day = load_day1(args.input)
    sol = solve(day.p, day.ell, day.r, s0_fixed=args.s0_fixed)

    print_solution(sol)
    print("== 诊断 ==")
    for key, val in diagnostics(sol, day).items():
        print(f"  {key}: {val}")

    print("== 对照 ==")
    print(f"  Z* = {sol.cost:.2f} 元   S_0 = {sol.S[0]:.2f} kWh")
    if args.s0_fixed is None:
        sol6000 = solve(day.p, day.ell, day.r, s0_fixed=6000.0)
        print(f"  Z^6000 = {sol6000.cost:.2f} 元   "
              f"S_0 决策化收益 = {sol6000.cost - sol.cost:.2f} 元")
        for delta in (-200.0, 200.0):  # S_0 平坦性探测
            probe = solve(day.p, day.ell, day.r, s0_fixed=sol.S[0] + delta)
            print(f"  S_0 钉在 {sol.S[0] + delta:.0f} 时 Z = {probe.cost:.2f} 元")

    path = Result1Xlsx(args.template).extract(day, sol, args.output_dir)
    print(f"已导出 {path}")


if __name__ == "__main__":
    main()
