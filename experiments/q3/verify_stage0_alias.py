"""题3 诊断：证实 nmmc.model.q3.solve_stage 在 m=0 的变量别名缺陷。

缺陷（源码 src/nmmc/model/q3.py 的变量布局）：

    o_dp = n_hn if m else 0                  # 共享 δ+/δ−（仅 m≥1）
    o_s  = o_dp + (2 * n_hn if m else 0)     # 首情景块起点

m ≥ 1 时 o_dp = n_hn、o_s = 3·n_hn，正确；**m = 0 时 o_dp = 0 ⇒ o_s = 0**，
于是 here-and-now 的 144 个计划变量 g 与"情景 0 块"的前 144 个变量占用同一
列，而向量末尾多出 144 个从未被引用的自由列（n 已按正确尺寸分配）。按
m=0 的块布局 [a^{(0)}(108), δ⁺^{(0)}(108), δ⁻^{(0)}(108), c, d, u, w, S(145)]：

    g[0:108]   与 a^{(0)}[36:144] 同列   → 计划的前 108 格 ≡ 情景0 对后 108
                                          格的试探注入（错位 36 格）
    g[108:144] 与 δ⁺^{(0)}[0:36] 同列    → 计划的末 36 格 ≡ 情景0 在 t=36..71
                                          的上调量

后果：0:00 计划 g 被一组与它毫无关系的等式绑死，解出来的既不是计划也不是
情景注入。这不是"松弛"而是**额外等式**，故 Z̃_bug ≥ Z̃_correct（可证伪预言）。
m ≥ 1（6/12/18 点改单）与内环 solve_step、CE 臂 solve_tail 的布局都没有这个
问题——所以病灶精确定位在"日前那一张 LP"。

用法：.venv/bin/python experiments/q3/verify_stage0_alias.py [--upto 120]
"""
import argparse

import numpy as np

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig, forecast_load
from nmmc.model.forecast3 import (Q3Config, build_scenarios, fusion_weights,
                                  hour_powers, pv_point)
from nmmc.model.lp import T
from nmmc.model.q2 import KAPPA, NU_END
from nmmc.model.q3 import solve_stage
from nmmc.run.q2 import solve_plan

from probe_stage0 import stage0_lp   # 修正布局的同款 LP（o_s = n_hn）

WARMUP = 31


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=120)
    ap.add_argument("--every", type=int, default=7)
    ap.add_argument("--s0", type=float, default=8000.0,
                    help="检验用固定日初 SOC（脱离 SOC 链，单点复算）")
    args = ap.parse_args()

    d1 = load_day1(ATTACH / "附件1.xlsx")
    year = load_year(ATTACH / "附件2.xlsx")
    fc = load_forecast(ATTACH / "附件3.xlsx")
    cfg = Q3Config()
    p = d1.p
    rng = np.random.default_rng(cfg.seed)
    cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
    r_hour = hour_powers(year.r)

    # 只重放预报层（无 LP）以复原 hist 与 rng 流
    hist: list[list[tuple[np.ndarray, np.ndarray]]] = []
    rec = []
    for j in range(args.upto):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, d1.ell)
        stage_fcasts = []
        scn0 = None
        for m in range(4):
            r_hat = pv_point(cfg, year, fc, j, m, d1.r, weights)
            ell_hat = ell0 if m == 0 else __import__(
                "nmmc.model.forecast3", fromlist=["load_point"]
            ).load_point(cfg, ell0, year, j, m)
            stage_fcasts.append((ell_hat, r_hat))
            s, _ = build_scenarios(cfg, year, hist, j, m, ell_hat, r_hat, rng)
            if m == 0:
                scn0 = s
        hist.append(stage_fcasts)
        if j < WARMUP or (j - WARMUP) % args.every:
            continue
        probs = np.full(len(scn0), 1.0 / len(scn0))
        bug = solve_stage(p, None, None, scn0, probs, 0, args.s0,
                          NU_END, KAPPA)
        ok = stage0_lp(p, scn0, probs, args.s0, NU_END, KAPPA)
        q2g = solve_plan(p, scn0, probs, args.s0, NU_END, KAPPA, True)[0]
        # 别名预言：g[0:108] ≡ a^(0)[36:144]，g[108:144] ≡ δ⁺^(0)[0:36]
        alias = float(np.abs(bug.commit[:108] - bug.a_scn[0, 36:]).max())
        rec.append({
            "j": j, "ns": len(scn0),
            "z_bug": bug.z_tilde, "z_ok": ok["z_tilde"],
            "g_bug": bug.commit.sum(), "g_ok": ok["g"].sum(),
            "g_q2": q2g.sum(),
            "b0_bug": bug.commit[:36].sum(), "b0_ok": ok["g"][:36].sum(),
            "b0_q2": q2g[:36].sum(),
            "free_bug": bug.commit[36:].sum(), "free_ok": ok["g"][36:].sum(),
            "free_q2": q2g[36:].sum(),
            "alias_resid": alias,
            "neg_bug": float(np.minimum(bug.commit, 0).sum()),
        })

    print(f"固定日初 SOC = {args.s0:,.0f} kWh，抽样 {len(rec)} 个报告期日\n")
    print("-- 别名预言核验：g[0:108] 与 a^(0)[36:144] 是否同一列 --")
    am = max(r["alias_resid"] for r in rec)
    print(f"   max|g[0:108] − a^(0)[36:144]| = {am:.3e} kWh"
          f"  → {'同列（缺陷证实）' if am < 1e-6 else '不同列'}")
    print("\n-- 可证伪预言：额外等式 ⇒ Z̃_bug ≥ Z̃_correct --")
    bad = sum(1 for r in rec if r["z_bug"] < r["z_ok"] - 1e-6)
    gap = np.array([r["z_bug"] - r["z_ok"] for r in rec])
    print(f"   Z̃_bug < Z̃_correct 的日子：{bad} / {len(rec)}（应为 0）")
    print(f"   Z̃_bug − Z̃_correct：均值 {gap.mean():+,.1f} 元/日"
          f"（{(gap / np.array([r['z_ok'] for r in rec])).mean():+.2%}）"
          f"  min {gap.min():+,.1f}  max {gap.max():+,.1f}")
    print("\n-- 计划形状：缺陷解 vs 修正解 vs 题2 SAA（非预期性对照）--")
    print(f"   {'':<14}{'Σg 全天':>12}{'Σg B_0(夜)':>13}{'Σg 可修订段':>13}")
    for lab, k in (("缺陷解(交付)", "bug"), ("修正解", "ok"),
                   ("题2 SAA", "q2")):
        print(f"   {lab:<14}{sum(r[f'g_{k}'] for r in rec) / len(rec):>12,.0f}"
              f"{sum(r[f'b0_{k}'] for r in rec) / len(rec):>13,.0f}"
              f"{sum(r[f'free_{k}'] for r in rec) / len(rec):>13,.0f}")
    b0b = sum(r["b0_bug"] for r in rec)
    b0o = sum(r["b0_ok"] for r in rec)
    fb = sum(r["free_bug"] for r in rec)
    fo = sum(r["free_ok"] for r in rec)
    print(f"\n   夜间 B_0：缺陷解比修正解少 {b0o - b0b:,.0f} kWh"
          f"（{b0b / b0o - 1:+.1%}）")
    print(f"   可修订段：缺陷解比修正解多 {fb - fo:,.0f} kWh"
          f"（{fb / fo - 1:+.1%}）")
    print(f"\n   缺陷解里 g < 0 的量（题3 摘要表 1 出现的 -0.00 即此）："
          f"日均 {sum(r['neg_bug'] for r in rec) / len(rec):,.3f} kWh")


if __name__ == "__main__":
    main()
