"""自检 C16 的可执行版：核验规划层 u 上界迭代是否**经济收敛**。

链条位置：v6 修正的收尾。§6.1 的序列线性化以 ‖Δg‖₁/Σg ≤ 1e-6 判不动点，
但 LP 存在等价最优（题 1 的 S₀ 扫描已证该模型有退化方向），少数日子会出现
**缓慢棘轮**：每轮 g 在"计划刚好盖住情景净负荷"的时段抬一点，cap 随之缩一点，
残差长期停在 1e-5 量级下不去。此时该问的不是"g 还在动吗"，而是"**还差多少钱**"。

本脚本对交付明细里 cap_resid > tol 的日子重放规划层迭代，打印每轮的
‖Δg‖₁/Σg 与 |ΔZ̃|/Z̃。判定：末轮 |ΔZ̃|/Z̃ ≤ 1e-4（等价于日费用量级 ≤ 数元）
即认定经济收敛，可交付；否则须加大轮数上限或改阻尼迭代。

⚠️ **不得按实测费用挑迭代轮次**。同一日不同轮次的 Z̃ 只差几元，但落到单条
真实路径上的现金费用可差上百元（2025-09-10：第 4 轮 44,468 元 vs 第 12 轮
44,595 元）。按实测费用选轮次 = 用真值选计划，属读法 C 明令禁止的信息泄漏。
唯一合法的取值点是迭代自身的收敛点。

用法：
    .venv/bin/python experiments/q2/check_c16.py
    .venv/bin/python experiments/q2/check_c16.py --details results/q2_v6/L1-P1-R2-smpc/q2_details.csv
"""
import argparse

import numpy as np
import pandas as pd

from nmmc.data.day import load_day1
from nmmc.data.year import load_year
from nmmc.model.forecast import (ForecastConfig, build_scenarios,
                                 forecast_load, forecast_pv)
from nmmc.model.q2 import KAPPA, NU_END, emergency_cap, solve_planning
from nmmc.run.q2 import CAP_MAX, CAP_TOL

DZ_GATE = 1e-4      # 经济收敛判据：末轮 |ΔZ̃|/Z̃


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="C16 经济收敛核验")
    ap.add_argument("--details", type=str,
                    default="results/q2_v6/L1-P1-R2-smpc/q2_details.csv",
                    help="交付明细 CSV（需含 cap_resid 列）")
    ap.add_argument("--tol", type=float, default=CAP_TOL,
                    help="决策空间不动点判据（默认与 run.q2.CAP_TOL 一致）")
    ap.add_argument("--passes", type=int, default=CAP_MAX,
                    help="重放轮数")
    return ap


def main(argv: list[str] | None = None) -> None:
    """重放未收敛日的规划层迭代并判定经济收敛。"""
    args = build_parser().parse_args(argv)
    det = pd.read_csv(args.details)
    if "cap_resid" not in det:
        raise SystemExit(f"{args.details} 无 cap_resid 列（需 v6 之后的产物）")
    flagged = det[det.cap_resid > args.tol]
    print(f"决策空间未达 {args.tol:g} 的日子：{len(flagged)} / {len(det)} 天")
    if flagged.empty:
        print("C16 过：全部日子在轮数上限内达到不动点。")
        return

    year = load_year()
    d1 = load_day1()
    p = d1.p
    cfg = ForecastConfig(load="L1", pv="P1", resid="R2", seed=0)
    dates = list(flagged.date)
    js = {d: int(np.where(year.dates == np.datetime64(d))[0][0]) for d in dates}
    # 情景抽样依赖 rng 状态，必须从 j=0 顺序重放（无 LP，秒级）
    rng = np.random.default_rng(cfg.seed)
    fcasts: list[tuple[np.ndarray, np.ndarray]] = []
    scns = {}
    for k in range(max(js.values()) + 1):
        ell_hat = forecast_load(cfg, year, k, d1.ell)
        r_hat = forecast_pv(cfg, year, k, d1.r)
        scn = build_scenarios(cfg, year, fcasts, k, ell_hat, r_hat, rng)
        if k in js.values():
            scns[k] = scn
        fcasts.append((ell_hat, r_hat))

    worst = 0.0
    for date in dates:
        j = js[date]
        scn = scns[j]
        probs = np.full(len(scn), 1.0 / len(scn))
        s0 = float(det.query("date == @date").S0.iloc[0])
        plan = solve_planning(p, scn, probs, s0, NU_END, KAPPA)
        print(f"\n{date}  S0 = {s0:,.1f} kWh")
        print("   轮   ‖Δg‖₁/Σg     Z̃（元）        |ΔZ̃|/Z̃")
        prev_z = None
        dz = float("nan")
        for k in range(1, args.passes + 1):
            prev_g = plan.g
            plan = solve_planning(p, scn, probs, s0, NU_END, KAPPA,
                                  emergency_cap(scn, prev_g))
            dg = float(np.abs(plan.g - prev_g).sum() / plan.g.sum())
            if prev_z is not None:
                dz = abs(plan.z_tilde - prev_z) / abs(prev_z)
            prev_z = plan.z_tilde
            print(f"  {k:3d}   {dg:.3e}   {plan.z_tilde:12,.4f}   {dz:.2e}")
        worst = max(worst, dz)

    print(f"\n末轮 |ΔZ̃|/Z̃ 最大 = {worst:.2e}，判据 {DZ_GATE:g}")
    print("C16 过：残余为等价最优间的缓慢棘轮，经济上已收敛。" if worst <= DZ_GATE
          else "C16 不过：须加大轮数上限或改阻尼迭代，不得交付。")


if __name__ == "__main__":
    main()
