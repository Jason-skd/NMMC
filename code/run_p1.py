"""题1 主流程：加载 → 求解 → 表1/表2 → 诊断与对照 → 导出 result1.xlsx。"""
import numpy as np

from data import load_day1
from model import T, Solution, solve
from report import diagnostics, table1_rows, table2_rows, write_result1


def print_solution(sol: Solution) -> None:
    print("== 表1 指定时段购电量 (kWh) ==")
    for label, val in table1_rows(sol):
        print(f"  {label:>12}  {val:12.2f}")
    print("== 表2 充放电量 (kWh, 电池侧) ==")
    for label, ch, dis in table2_rows(sol):
        print(f"  {label:>10}  充 {ch:10.2f}  放 {dis:10.2f}")
    print(f"  {'0:00 储电量':>10}  {sol.S[0]:10.2f}")
    print(f"  {'24:00 储电量':>10}  {sol.S[T]:10.2f}")


def main() -> None:
    day = load_day1()
    sol = solve(day.p, day.ell, day.r)
    print_solution(sol)

    print("== 诊断 ==")
    for key, val in diagnostics(sol, day.p, day.ell, day.r).items():
        print(f"  {key}: {val}")

    sol6000 = solve(day.p, day.ell, day.r, s0_fixed=6000.0)
    gain = sol6000.cost - sol.cost
    print(f"== 对照 ==")
    print(f"  Z* = {sol.cost:.2f} 元   S_0* = {sol.S[0]:.2f} kWh")
    print(f"  Z^6000 = {sol6000.cost:.2f} 元   S_0 决策化收益 = {gain:.2f} 元")

    for delta in (-200.0, 200.0):  # S_0 退化探测
        probe = solve(day.p, day.ell, day.r, s0_fixed=sol.S[0] + delta)
        print(f"  S_0 钉在 {sol.S[0] + delta:.0f} 时 Z = {probe.cost:.2f} 元")

    write_result1(sol)
    print("已导出 output/result1.xlsx")


if __name__ == "__main__":
    main()
