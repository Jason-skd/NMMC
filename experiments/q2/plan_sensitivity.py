"""规划层敏感性：在**新口径**（u ≤ (n−g)⁺）下重扫裕度与末端估值。

链条位置：修正后的复核。变量档 §11.4 那张"加厚裕度只涨钱 / ν^end=0.472
最优"的表是在 v5 **无上界**口径下测的；上界关掉了日内 5 倍价补货通道，
日前多买的边际价值理论上上升，故两组结论都必须在新口径下重验。

口径与 §11.4 对齐：窗口 day 100–119（0 基 j=99..118），前 99 天照常滚动
以喂 SOC 链与预报历史，只统计窗口内。其余与交付一致（L1-P1-R2、seed 0、
在线随机 MPC）。基线不必重跑——直接取交付 run 的同窗口（见 --baseline）。

用法：
    # 裕度：情景净负荷整体上抬 50 kWh/时段（§11.4 原来的两点是 50 / 100）
    .venv/bin/python experiments/q2/plan_sensitivity.py --bump 50

    # 末端估值：ν^end 三点重扫（§11.4 原来的三点是 0.60 / 0.80 / 1.00）
    .venv/bin/python experiments/q2/plan_sensitivity.py --nu-end 0.60

    # v5 口径对照（复现 §11.4 原表）
    .venv/bin/python experiments/q2/plan_sensitivity.py --bump 50 --no-emerg-cap

每个点约 20 分钟（120 天 × 规划层迭代到不动点）。可并行跑多个点，
组合之间互不影响（跨日 SOC 链只在单个进程内串行）。

判读：只看窗口总费用。**紧急指标变好但总费用上升 = "以钱买心安"，
不构成改进**（这是 §11.4 的核心结论，v6 已用 bump=50 复核：费用
+4.79%、弃电 +65%、紧急电量 −71%）。
"""
import argparse

import numpy as np
import pandas as pd

from nmmc.data.day import load_day1
from nmmc.data.year import load_year
from nmmc.model.forecast import (ForecastConfig, build_scenarios,
                                 forecast_load, forecast_pv)
from nmmc.model.lp import T
from nmmc.model.q2 import KAPPA, NU_END
from nmmc.run.q2 import S0_INIT, cash_cost, run_smpc_day, solve_plan


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    ap = argparse.ArgumentParser(description="题2 规划层敏感性（新口径）")
    ap.add_argument("--bump", type=float, default=0.0,
                    help="情景净负荷整体上抬量，kWh/时段（裕度扫描）")
    ap.add_argument("--nu-end", type=float, default=NU_END,
                    help="日末储电量估值 ν^end")
    ap.add_argument("--days", type=int, default=120, help="滚动天数（含暖机）")
    ap.add_argument("--window", type=int, nargs=2, default=(99, 119),
                    metavar=("LO", "HI"), help="统计窗口（0 基，左闭右开）")
    ap.add_argument("--no-emerg-cap", action="store_true",
                    help="退回 v5 无上界口径（复现 §11.4 原表）")
    ap.add_argument("--baseline", type=str,
                    default="results/q2_v6/L1-P1-R2-smpc/q2_details.csv",
                    help="基线明细 CSV（同窗口对照，缺省取 v6 交付）")
    return ap


def main(argv: list[str] | None = None) -> None:
    """跑一个敏感性点并打印窗口汇总（与基线同窗口对照）。"""
    args = build_parser().parse_args(argv)
    emerg_cap = not args.no_emerg_cap
    lo, hi = args.window
    year = load_year()
    d1 = load_day1()
    p = d1.p
    cfg = ForecastConfig(load="L1", pv="P1", resid="R2", seed=0)
    rng = np.random.default_rng(cfg.seed)
    fcasts: list[tuple[np.ndarray, np.ndarray]] = []
    soc = S0_INIT
    rows = []
    for j in range(args.days):
        ell_hat = forecast_load(cfg, year, j, d1.ell)
        r_hat = forecast_pv(cfg, year, j, d1.r)
        scn, _ = build_scenarios(cfg, year, fcasts, j, ell_hat, r_hat,
                                 rng)
        scn = scn + args.bump
        probs = np.full(len(scn), 1.0 / len(scn))
        g, u_cap, _, _ = solve_plan(p, scn, probs, soc, args.nu_end, KAPPA,
                                    emerg_cap)
        net = year.ell[j] - year.r[j]
        c, d, u, w, s = run_smpc_day(p, net, g, scn, probs, soc, args.nu_end,
                                     KAPPA, u_cap)
        rows.append({"j": j, "z": cash_cost(p, g, u, KAPPA),
                     "u": float(u.sum()), "slots": int((u > 1e-6).sum()),
                     "w": float(w.sum()), "soc": float(s[1:].mean()),
                     "fee": KAPPA * float(p @ u)})
        soc = s[T]
        fcasts.append((ell_hat, r_hat))
        if (j + 1) % 30 == 0:
            print(f"  {j + 1}/{args.days} 天", flush=True)

    win = pd.DataFrame([r for r in rows if lo <= r["j"] < hi])
    z = win.z.sum()
    print(f"\n=== bump={args.bump:g} kWh/时段，ν^end={args.nu_end:g}，"
          f"口径 {'u ≤ (n−g)⁺' if emerg_cap else '无上界(v5)'}，"
          f"窗口 day {lo + 1}–{hi}（{len(win)} 天）")
    print(f"总费用      {z:,.0f} 元")
    print(f"紧急费      {win.fee.sum():,.0f} 元（占 {100 * win.fee.sum() / z:.2f}%）")
    print(f"紧急电量    {win.u.sum():,.0f} kWh；缺口时段 {win.slots.sum()}"
          f"（频率 {100 * win.slots.sum() / (len(win) * T):.2f}%）")
    print(f"弃电        {win.w.sum():,.0f} kWh；SOC 日内均值 {win.soc.mean():,.0f}")
    try:
        base = pd.read_csv(args.baseline).iloc[lo - 31:hi - 31]
        bz = base.z_online.sum()
        print(f"\n基线同窗口  {bz:,.0f} 元（{args.baseline}）")
        print(f"差额        {z - bz:+,.0f} 元（{100 * (z / bz - 1):+.2f}%）"
              f"  ← 上升即以钱买心安，不构成改进")
    except (FileNotFoundError, KeyError, IndexError) as exc:
        print(f"\n（基线未读取：{exc}）")


if __name__ == "__main__":
    main()
