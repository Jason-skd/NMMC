"""题3 诊断：阶段0 规划 LP 的退化性与"想象中的紧急购电"探针。

两个待证命题：

P1（退化 / 无 tie-break）。阶段0 LP 里 t ≥ 36 的 g 在目标函数中**只**通过
    偏差项 ½p(δ⁺+δ⁻) 出现（solve_stage 的 cost_hn 只覆盖 p[:36]）。给定
    逐情景试探注入 a^(s)，最优 g_t 是 {a^(s)_t} 的加权中位数；情景等权且
    条数为偶数（交付 N_s = 12）时中位数是一个**区间**，Z̃ 在区间上恒定，
    HiGHS 返回其中某个顶点。若返回的是区间上端，g 就被系统性抬高，而
    Z̃ 完全看不出来——退化在规划层免费、在滚动执行里按 ½p 收费。
    检验：给 g_{t≥36} 加一个 ε·p 的极小 tie-break 再解，看 Σg 掉多少、
    Z̃ 变多少（真退化 ⇒ Σg 明显下降而 Z̃ 几乎不动）。

P2（缺 u 上界的影响面）。题3 三个 LP 入口都没有 u ≤ (n−a)⁺（题2 v6 §5.1）。
    实测轨迹自动满足（由 u·c = 0、u·w = 0），但**规划层情景块**里的
    u^(s) 无上界，可以扮演"5 倍价进货通道"，从而压低存量的影子价值。
    检验：直接取出阶段0 LP 的情景解，统计 u^(s) 超出 (n^(s)−a^(s))⁺ 的量
    与"u^(s) 与 c^(s) 同时为正"的时段数。

用法：.venv/bin/python experiments/q3/probe_stage0.py [--every 7]
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
from nmmc.model.lp import (E_MAX_PER, ETA, S_MAX, S_MIN, T, solve_eq_lp,
                           soc_recursion)
from nmmc.model.q2 import KAPPA, NU_END
from nmmc.model.q3 import solve_stage
from nmmc.run.q2 import S0_INIT, solve_plan
from nmmc.run.q3 import run_smpc_segment

ROOT = Path(__file__).resolve().parents[2]
WARMUP = 31


def stage0_lp(p: np.ndarray, scn: np.ndarray, probs: np.ndarray,
              s_anchor: float, nu_end: float, kappa: float,
              eps_g: float = 0.0, u_cap: np.ndarray | None = None) -> dict:
    """阶段0 LP（nmmc.model.q3.solve_stage 的 m=0 分支逐行复刻），额外支持：

    eps_g: 给 g_{t≥36} 加 eps_g·p_t 的 tie-break 系数（P1 探针）。
    u_cap: 情景紧急购电上界 (N_s, 144)（P2 反事实）。

    返回 dict：g、a_scn、z_tilde（含 tie-break 项）、z_true（剔除 tie-break
    的原目标值，跨 eps 可比）、以及逐情景 (c, d, u, w, S)。
    """
    m, h0 = 0, 0
    n_s = len(scn)
    he, fe = T, 36
    n_hn, n_free, nh = T, T - fe, T
    n_blk = 3 * n_free + 5 * nh + 1
    n = n_hn + n_s * n_blk
    o_hn, o_s = 0, n_hn

    obj = np.zeros(n)
    obj_tb = np.zeros(n)                      # 只含 tie-break 项，便于剔除
    lo, hi = np.zeros(n), np.full(n, np.inf)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    b: list[float] = []
    row = 0
    obj[o_hn:o_hn + 36] = p[:36]              # m=0 只计不可调整段的计划费
    if eps_g:
        obj_tb[o_hn + 36:o_hn + T] = eps_g * p[36:]
    for s in range(n_s):
        o = o_s + s * n_blk
        o_af, o_dpf = o, o + n_free
        o_dmf = o_dpf + n_free
        o_c = o_dmf + n_free
        o_d, o_u = o_c + nh, o_c + 2 * nh
        o_w, o_st = o_c + 3 * nh, o_c + 4 * nh
        pi = probs[s]
        tf = np.arange(n_free)
        rf = row + tf
        rows += [rf, rf, rf, rf]
        cols += [o_af + tf, o_dpf + tf, o_dmf + tf, o_hn + fe + tf]
        vals += [np.ones(n_free), -np.ones(n_free), np.ones(n_free),
                 -np.ones(n_free)]
        b += [0.0] * n_free
        obj[o_af:o_af + n_free] = pi * p[fe:]
        obj[o_dpf:o_dpf + n_free] = pi * 0.5 * p[fe:]
        obj[o_dmf:o_dmf + n_free] = pi * 0.5 * p[fe:]
        row += n_free
        tt = np.arange(nh)
        rb = row + tt
        for arr in (o_d + tt, o_u + tt):
            rows.append(rb)
            cols.append(arr)
            vals.append(np.ones(nh))
        for arr in (o_c + tt, o_w + tt):
            rows.append(rb)
            cols.append(arr)
            vals.append(-np.ones(nh))
        rows.append(rb[:fe])
        cols.append(o_hn + tt[:fe])
        vals.append(np.ones(fe))
        rows.append(rb[fe:])
        cols.append(o_af + tt[fe:] - fe)
        vals.append(np.ones(n_free))
        b += list(scn[s, h0:])
        row += nh
        sr, sc, sv = soc_recursion(nh, row, o_st, o_c, o_d)
        rows += sr
        cols += sc
        vals += sv
        b += [0.0] * nh
        row += nh
        obj[o_u:o_u + nh] = pi * kappa * p
        obj[o_st + nh] = -pi * nu_end
        hi[o_c:o_c + nh] = E_MAX_PER
        hi[o_d:o_d + nh] = E_MAX_PER
        lo[o_st:o_st + nh + 1] = S_MIN
        hi[o_st:o_st + nh + 1] = S_MAX
        lo[o_st] = hi[o_st] = s_anchor
        if u_cap is not None:
            hi[o_u:o_u + nh] = u_cap[s]

    x, _, _ = solve_eq_lp(obj + obj_tb, rows, cols, vals, np.array(b), lo, hi)
    g = x[o_hn:o_hn + n_hn].copy()
    a_scn = np.repeat(g[None, :], n_s, axis=0)
    sc_c, sc_d, sc_u, sc_w = (np.zeros((n_s, T)) for _ in range(4))
    sc_S = np.zeros((n_s, T + 1))
    for s in range(n_s):
        o = o_s + s * n_blk
        o_c = o + 3 * n_free
        o_d, o_u = o_c + nh, o_c + 2 * nh
        o_w, o_st = o_c + 3 * nh, o_c + 4 * nh
        a_scn[s, fe:] = x[o:o + n_free]
        sc_c[s], sc_d[s] = x[o_c:o_c + nh], x[o_d:o_d + nh]
        sc_u[s], sc_w[s] = x[o_u:o_u + nh], x[o_w:o_w + nh]
        sc_S[s] = x[o_st:o_st + nh + 1]
    return {"g": g, "a_scn": a_scn, "z_tilde": float((obj + obj_tb) @ x),
            "z_true": float(obj @ x), "c": sc_c, "d": sc_d, "u": sc_u,
            "w": sc_w, "S": sc_S}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=7,
                    help="每隔几天做一次探针（探针本身不入 SOC 链）")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--days", type=int, default=DAYS)
    args = ap.parse_args()

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
    for j in range(args.days):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, d1.ell)
        net = year.ell[j] - year.r[j]
        g = None
        committed = None
        c, d, u, w = (np.zeros(T) for _ in range(4))
        s = np.zeros(T + 1)
        s[0] = soc
        stage_fcasts = []
        probe = j >= WARMUP and (j - WARMUP) % args.every == 0
        for m in range(4):
            h0, h1 = 36 * m, 36 * (m + 1)
            r_hat = pv_point(cfg, year, fc, j, m, d1.r, weights)
            ell_hat = ell0 if m == 0 else load_point(cfg, ell0, year, j, m)
            stage_fcasts.append((ell_hat, r_hat))
            scn, _ = build_scenarios(cfg, year, hist, j, m, ell_hat, r_hat, rng)
            probs = np.full(len(scn), 1.0 / len(scn))
            res = solve_stage(p, g, committed, scn, probs, m, s[h0],
                              NU_END, KAPPA)
            if m == 0:
                g = res.commit
                if probe:
                    base = stage0_lp(p, scn, probs, s[0], NU_END, KAPPA)
                    tb = stage0_lp(p, scn, probs, s[0], NU_END, KAPPA,
                                   eps_g=args.eps)
                    cap = np.maximum(scn - base["a_scn"], 0.0)
                    capped = stage0_lp(p, scn, probs, s[0], NU_END, KAPPA,
                                       u_cap=np.maximum(
                                           scn - np.median(base["a_scn"],
                                                           axis=0), 0.0))
                    q2g = solve_plan(p, scn, probs, s[0], NU_END, KAPPA,
                                     True)[0]
                    asc = base["a_scn"]
                    free = slice(36, T)
                    # g 在 {a^(s)} 经验分布中的位置（0=全在其上，1=全在其下）
                    pos = (asc[:, free] < base["g"][free] - 1e-6).mean(0)
                    su, sc_ = base["u"], base["c"]
                    rows.append({
                        "date": str(year.dates[j].astype(object)), "ns": len(scn),
                        "g_solve_stage": res.commit.sum(),
                        "g_probe": base["g"].sum(),
                        "g_probe_free": base["g"][free].sum(),
                        "g_tb_free": tb["g"][free].sum(),
                        "g_tb": tb["g"].sum(),
                        "g_b0": base["g"][:36].sum(),
                        "g_tb_b0": tb["g"][:36].sum(),
                        "g_q2_b0": q2g[:36].sum(),
                        "g_q2": q2g.sum(),
                        "z_true": base["z_true"], "z_tb_true": tb["z_true"],
                        "z_capped_true": capped["z_true"],
                        "g_capped": capped["g"].sum(),
                        "g_capped_b0": capped["g"][:36].sum(),
                        "pos_mean": float(pos.mean()),
                        "pos_ge_half": float((pos >= 0.5 - 1e-9).mean()),
                        "pos_eq_one": float((pos >= 1.0 - 1e-9).mean()),
                        "scn_u": float(su.sum() / len(scn)),
                        "scn_u_over_cap": float(
                            np.maximum(su - np.maximum(
                                scn - base["a_scn"], 0.0), 0.0).sum()
                            / len(scn)),
                        "scn_uc_slots": int(((su > 1e-6) & (sc_ > 1e-6)).sum()),
                        "scn_slots": int(su.size),
                        "cap_unused": float(cap.sum()),
                    })
            committed = res.commit
            run_smpc_segment(p, net, committed, g, scn, probs, c, d, u, w,
                             s, h0, h1, NU_END, KAPPA, free_from=h1)
        soc = s[T]
        hist.append(stage_fcasts)
        if (j + 1) % 62 == 0:
            print(f"  {j + 1}/{args.days} 天", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results/q3-diag/probe_stage0.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"\n== 阶段0 探针（{len(df)} 个抽样日，每 {args.every} 天一个）==")
    print(f"  探针复算 g 与 solve_stage 一致性：max|Δ| = "
          f"{(df.g_probe - df.g_solve_stage).abs().max():.3e} kWh")
    print("\n-- P1 退化 / tie-break --")
    print(f"  Σg（t≥36，可修订段）  原解 {df.g_probe_free.sum():,.0f} → "
          f"加 ε={args.eps:g} tie-break 后 {df.g_tb_free.sum():,.0f} kWh"
          f"（{df.g_tb_free.sum() / df.g_probe_free.sum() - 1:+.2%}）")
    print(f"  Σg（B_0，有真实费用）  原解 {df.g_b0.sum():,.0f} → "
          f"{df.g_tb_b0.sum():,.0f} kWh"
          f"（{df.g_tb_b0.sum() / df.g_b0.sum() - 1:+.2%}）")
    print(f"  规划目标 Z̃（剔除 tie-break 项）原 {df.z_true.sum():,.2f} → "
          f"{df.z_tb_true.sum():,.2f} 元"
          f"（{df.z_tb_true.sum() / df.z_true.sum() - 1:+.3%}）")
    print(f"  逐日 |ΔZ̃|/Z̃ 中位 "
          f"{((df.z_tb_true - df.z_true).abs() / df.z_true).median():.2e}，"
          f"最大 {((df.z_tb_true - df.z_true).abs() / df.z_true).max():.2e}")
    print(f"  → g 的可修订段在规划层几乎无成本信号："
          f"Σg 掉 {1 - df.g_tb_free.sum() / df.g_probe_free.sum():.1%} 而 Z̃ 只动 "
          f"{abs(df.z_tb_true.sum() / df.z_true.sum() - 1):.3%}")
    print(f"\n  g 在 {{a^(s)}} 经验分布中的位置（0=最低，1=最高）：")
    print(f"    日均分位 {df.pos_mean.mean():.3f}；"
          f"≥中位数的时段占比 {df.pos_ge_half.mean():.1%}；"
          f"= 全情景最高的时段占比 {df.pos_eq_one.mean():.1%}")
    print("\n-- 非预期性对照（题2 SAA 计划器，同情景集）--")
    print(f"  Σg  题3 阶段0 {df.g_probe.sum():,.0f} vs 题2 SAA "
          f"{df.g_q2.sum():,.0f} kWh"
          f"（{df.g_q2.sum() / df.g_probe.sum() - 1:+.1%}）")
    print(f"  ΣgB_0 题3 {df.g_b0.sum():,.0f} vs 题2 SAA {df.g_q2_b0.sum():,.0f}"
          f" kWh（{df.g_q2_b0.sum() / df.g_b0.sum() - 1:+.1%}）")
    print("\n-- P2 规划层情景块里的紧急购电 --")
    print(f"  每情景日均 Σu^(s) = {df.scn_u.mean():,.1f} kWh；"
          f"超出 (n−a)⁺ 的部分 {df.scn_u_over_cap.mean():,.1f} kWh")
    print(f"  情景块 u^(s)∧c^(s) 同时为正的时段 {df.scn_uc_slots.sum():,} / "
          f"{df.scn_slots.sum():,}（{df.scn_uc_slots.sum() / df.scn_slots.sum():.2%}）")
    print(f"  加 u 上界后 Σg {df.g_capped.sum():,.0f}（原 {df.g_probe.sum():,.0f}），"
          f"Z̃ {df.z_capped_true.sum():,.2f}（原 {df.z_true.sum():,.2f}）")
    print(f"  已落盘 {out}")


if __name__ == "__main__":
    main()
