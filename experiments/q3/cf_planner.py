"""题3 诊断反事实：把 0:00 规划层换成"非预期性"计划器，其余全不动。

交付臂的阶段0 LP（nmmc.model.q3.solve_stage, m=0）把 B_1..B_3 的注入写成
逐情景 wait-and-see 变量 a^(s)——即在选 g 的当口就假定"届时能按当时的
情景把注入重定到最优"。这是一个松弛：真实的 6/12/18 点只看得到已实现
前缀 + 更新预报，不是整条情景。本脚本换掉这一处，其它（预报层、情景器、
修订机制、内环随机 MPC、SOC 链、rng 消耗序）逐位不动：

  CF-A  plan=q2   g 由题2 两阶段 SAA（solve_plan，无 a^(s)、须自己扛全天）
                  给出；6/12/18 照常行使修订权（题3 solve_stage, m≥1）。
  CF-B  plan=q2   同上 g，但 a ≡ g 全天冻结（不行使修订权）——这是 C3-1
        + freeze 要求的"真 E0"对照（现仓库里的 E0 用的是交付臂那个
                  假定可改的规划器，属稻草人）。
  CF-C  plan=q3   plan 不变、只冻结 a ≡ g = 复现仓库里的 E0（回归核对）。
  CF-D  plan=fix  阶段0 LP 换成修正变量布局的同款 LP（experiments/q3/
                  probe_stage0.stage0_lp，o_s = n_hn；已用 N_s=1 与题2
                  solve_planning 逐位等价核验），6/12/18 照常改单——量化
                  别名缺陷的全年代价。
  CF-E  plan=fix  同 D 的 g，但 a ≡ g 冻结（修正后的"不改单"参照）。

用法：
  .venv/bin/python experiments/q3/cf_planner.py --arm A
  .venv/bin/python experiments/q3/cf_planner.py --arm D
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import DAYS, load_year
from nmmc.model.forecast import ForecastConfig, forecast_load
from nmmc.model.forecast3 import (Q3Config, build_scenarios, fusion_weights,
                                  hour_powers, load_point, pv_point)
from nmmc.model.lp import T
from nmmc.model.q2 import KAPPA, NU_END, solve_recourse
from nmmc.model.q3 import settle_cost, solve_stage
from nmmc.run.q2 import S0_INIT, solve_plan
from nmmc.run.q3 import run_smpc_segment

from probe_stage0 import stage0_lp

ROOT = Path(__file__).resolve().parents[2]
WARMUP = 31
REVISING = ("A", "C", "D")     # 行使 6/12/18 修订权的臂（B/E 冻结 a ≡ g）


def run(arm: str, cap: bool, n_days: int) -> pd.DataFrame:
    """跑一条反事实臂，返回逐日明细（含暖机，报告期在调用方切片）。"""
    d1 = load_day1(ATTACH / "附件1.xlsx")
    year = load_year(ATTACH / "附件2.xlsx")
    fc = load_forecast(ATTACH / "附件3.xlsx")
    cfg = Q3Config()
    p = d1.p
    rng = np.random.default_rng(cfg.seed)
    cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
    r_hour = hour_powers(year.r)
    hist: list[list[tuple[np.ndarray, np.ndarray]]] = []
    soc = S0_INIT
    rows = []
    for j in range(n_days):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, d1.ell)
        net = year.ell[j] - year.r[j]
        g = None
        committed = None
        c, d, u, w = (np.zeros(T) for _ in range(4))
        s = np.zeros(T + 1)
        s[0] = soc
        stage_fcasts = []
        for m in range(4):
            h0, h1 = 36 * m, 36 * (m + 1)
            r_hat = pv_point(cfg, year, fc, j, m, d1.r, weights)
            ell_hat = ell0 if m == 0 else load_point(cfg, ell0, year, j, m)
            stage_fcasts.append((ell_hat, r_hat))
            scn, _ = build_scenarios(cfg, year, hist, j, m, ell_hat, r_hat, rng)
            probs = np.full(len(scn), 1.0 / len(scn))
            if m == 0:
                if arm in ("A", "B"):
                    # 非预期性计划：题2 两阶段 SAA，g 须自己扛全天
                    g = solve_plan(p, scn, probs, s[0], NU_END, KAPPA, cap)[0]
                elif arm in ("D", "E"):
                    # 修正变量布局的阶段0 LP（别名缺陷已修）
                    g = stage0_lp(p, scn, probs, s[0], NU_END, KAPPA)["g"]
                else:
                    g = solve_stage(p, None, None, scn, probs, 0, s[0],
                                    NU_END, KAPPA).commit
                committed = g.copy()
            elif arm in REVISING:
                committed = solve_stage(p, g, committed, scn, probs, m,
                                        s[h0], NU_END, KAPPA).commit
            # arm B / E：a ≡ g，不重解承诺
            free_from = T          # 注入全程锁定（内环无权改 a）
            run_smpc_segment(p, net, committed, g, scn, probs, c, d, u, w,
                             s, h0, h1, NU_END, KAPPA, free_from=free_from)
        dev = np.abs(committed - g)
        u_exp = solve_recourse(p, net, committed, soc, NU_END, KAPPA).u
        rows.append({
            "date": str(year.dates[j].astype(object)),
            "z_online": settle_cost(p, g, committed, u, KAPPA),
            "z_expost": settle_cost(p, g, committed, u_exp, KAPPA),
            "g_sum": g.sum(), "a_sum": committed.sum(),
            "base_fee": float(p @ committed),
            "dev_fee": 0.5 * float(p @ dev),
            "dp_sum": float(np.maximum(committed - g, 0)[36:].sum()),
            "dm_sum": float(np.maximum(g - committed, 0)[36:].sum()),
            "u_sum": u.sum(), "emerg_fee": KAPPA * float(p @ u),
            "w_sum": w.sum(), "S0": s[0], "S144": s[T],
            "night_c": float(c[:24].sum()), "night_d": float(d[:24].sum()),
        })
        soc = s[T]
        hist.append(stage_fcasts)
        if (j + 1) % 31 == 0:
            print(f"  [{arm}{'-cap' if cap else ''}] {j + 1}/{n_days} 天，"
                  f"SOC={soc:.0f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "D", "E"], required=True)
    ap.add_argument("--cap", action="store_true",
                    help="0:00 计划用 v6 口径 u ≤ (n−g)⁺（内环仍无上界）")
    ap.add_argument("--days", type=int, default=DAYS)
    args = ap.parse_args()
    df = run(args.arm, args.cap, args.days)
    tag = f"cf{args.arm}{'-cap' if args.cap else ''}"
    out = ROOT / "results/q3-diag" / f"{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")
    rep = df.iloc[WARMUP:]
    print(f"\n== {tag}（报告期 {len(rep)} 天）==")
    for k in ("z_online", "z_expost", "base_fee", "dev_fee", "emerg_fee"):
        print(f"  {k:10s} {rep[k].sum():16,.2f} 元")
    print(f"  Σδ+ {rep.dp_sum.sum():12,.0f} / Σδ− {rep.dm_sum.sum():12,.0f} kWh"
          f"  → δ+/δ− = "
          f"{rep.dp_sum.sum() / max(rep.dm_sum.sum(), 1e-9):.4f}")
    print(f"  Σg {rep.g_sum.sum():,.0f} / Σa {rep.a_sum.sum():,.0f} kWh"
          f"  Σu {rep.u_sum.sum():,.0f}  Σw {rep.w_sum.sum():,.0f}")
    print(f"  日初 SOC 均值 {rep.S0.mean():,.0f}  日末均值 "
          f"{rep.S144.mean():,.0f}  末日末 {rep.S144.values[-1]:,.0f}")
    print(f"  0–4h 日均充 {rep.night_c.mean():,.0f} / 放 "
          f"{rep.night_d.mean():,.0f} kWh（电池母线侧）")
    print(f"  已落盘 {out}")


if __name__ == "__main__":
    main()
