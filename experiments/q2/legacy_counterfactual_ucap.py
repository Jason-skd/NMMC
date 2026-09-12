"""临时实验：给紧急购电加"只补缺口"上界 u_t ≤ (n_t − g_t)^+，看费用与指标。

对照口径：同预报（L1-P1-R2、seed 0）、同 ν^end、同 κ、同 100 天，
唯一差别是 u 的上界。基线数字取自 scratch_soc_diag.py 的同口径运行。
"""
import numpy as np

from nmmc.data.day import load_day1
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig, build_scenarios, forecast_load, forecast_pv
from nmmc.model.lp import E_MAX_PER, ETA, S_MAX, S_MIN, T, solve_eq_lp, soc_recursion
from nmmc.model.q2 import KAPPA, NU_END

DAYS = 100


def plan_capped(p, scn, probs, s0, nu_end, kappa, cap):
    """题2 §6 的 SAA LP + u 上界（cap 为 (N_s,T) 或 None）。"""
    n_s, blk = len(scn), 5 * T + 1
    n = T + n_s * blk
    obj = np.zeros(n)
    obj[:T] = p
    lo, hi = np.zeros(n), np.full(n, np.inf)
    b = np.zeros(2 * T * n_s)
    rows, cols, vals = [], [], []
    tt = np.arange(T)
    for s in range(n_s):
        o_c = T + s * blk
        o_d, o_u, o_w, o_s = o_c + T, o_c + 2 * T, o_c + 3 * T, o_c + 4 * T
        r = s * 2 * T + tt
        rows += [r, r, r, r, r]
        cols += [tt, o_c + tt, o_d + tt, o_u + tt, o_w + tt]
        vals += [np.ones(T), -np.ones(T), np.ones(T), np.ones(T), -np.ones(T)]
        b[r] = scn[s]
        sr, sc, sv = soc_recursion(T, s * 2 * T + T, o_s, o_c, o_d)
        rows += sr
        cols += sc
        vals += sv
        pi = probs[s]
        obj[o_u:o_u + T] = pi * kappa * p
        obj[o_s + T] = -pi * nu_end
        hi[o_c:o_c + T] = E_MAX_PER
        hi[o_d:o_d + T] = E_MAX_PER
        lo[o_s:o_s + T + 1] = S_MIN
        hi[o_s:o_s + T + 1] = S_MAX
        lo[o_s] = hi[o_s] = s0
        if cap is not None:
            hi[o_u:o_u + T] = cap[s]
    x, _, _ = solve_eq_lp(obj, rows, cols, vals, b, lo, hi)
    return x[:T]


def step_capped(p, net, g, scn, probs, t, s_prev, nu_end, kappa, cap_u):
    """§7.2 单步两阶段 LP + u 上界（当步 cap_u[0]，未来 cap_u[1] 为 (N_s,m)）。"""
    m = T - 1 - t
    n_s = len(scn)
    blk = 5 * m + 1
    n = 5 + n_s * blk
    obj = np.zeros(n)
    obj[2] = kappa * p[t]
    lo, hi = np.zeros(n), np.full(n, np.inf)
    hi[0] = hi[1] = E_MAX_PER
    lo[4], hi[4] = S_MIN, S_MAX
    hi[2] = cap_u[0]
    rows = [np.array([0, 0, 0, 0]), np.array([1, 1, 1])]
    cols = [np.array([1, 2, 3, 0]), np.array([4, 0, 1])]
    vals = [np.array([1.0, 1.0, -1.0, -1.0]),
            np.array([1.0, -ETA, 1.0 / ETA])]
    b = [net[t] - g[t], s_prev]
    if m == 0:
        obj = np.array([0.0, 0.0, kappa * p[t], 0.0, -nu_end])
        x, _, _ = solve_eq_lp(obj, rows, cols, vals, np.asarray(b),
                              np.array([0.0, 0.0, 0.0, 0.0, S_MIN]),
                              np.array([E_MAX_PER, E_MAX_PER, cap_u[0],
                                        np.inf, S_MAX]))
        return float(x[0]), float(x[1]), float(x[2]), float(x[3])
    tt = np.arange(m)
    for s in range(n_s):
        o_c = 5 + s * blk
        o_d, o_u, o_w, o_s = o_c + m, o_c + 2 * m, o_c + 3 * m, o_c + 4 * m
        pi = float(probs[s])
        obj[o_u:o_u + m] = pi * kappa * p[t + 1:]
        obj[o_s + m] = -pi * nu_end
        hi[o_c:o_c + m] = E_MAX_PER
        hi[o_d:o_d + m] = E_MAX_PER
        hi[o_u:o_u + m] = cap_u[1][s]
        lo[o_s:o_s + m + 1] = S_MIN
        hi[o_s:o_s + m + 1] = S_MAX
        r0 = 2 + s * (2 * m + 1)
        rows.append(np.array([r0, r0]))
        cols.append(np.array([o_s, 4]))
        vals.append(np.array([1.0, -1.0]))
        b.append(0.0)
        r = r0 + 1 + tt
        rows += [r, r, r, r]
        cols += [o_c + tt, o_d + tt, o_u + tt, o_w + tt]
        vals += [-np.ones(m), np.ones(m), np.ones(m), -np.ones(m)]
        b.extend((scn[s, t + 1:] - g[t + 1:]).tolist())
        sr, sc, sv = soc_recursion(m, r0 + 1 + m, o_s, o_c, o_d)
        rows += sr
        cols += sc
        vals += sv
        b.extend([0.0] * m)
    x, _, _ = solve_eq_lp(obj, rows, cols, vals, np.asarray(b), lo, hi)
    return float(x[0]), float(x[1]), float(x[2]), float(x[3])


year = load_year()
d1 = load_day1()
p = d1.p
cfg = ForecastConfig(load="L1", pv="P1", resid="R2", seed=0)
rng = np.random.default_rng(cfg.seed)
fcasts = []
soc = 6000.0
rows_out = []
for j in range(DAYS):
    ell_hat = forecast_load(cfg, year, j, d1.ell)
    r_hat = forecast_pv(cfg, year, j, d1.r)
    scn = build_scenarios(cfg, year, fcasts, j, ell_hat, r_hat, rng)
    probs = np.full(len(scn), 1.0 / len(scn))
    # 规划层：情景内 u 只能补该情景的计划缺口
    g = plan_capped(p, scn, probs, soc, NU_END, KAPPA,
                    np.maximum(scn - np.zeros_like(scn), 0.0) * 0 + np.inf)
    # 两遍：g 依赖 cap，cap 依赖 g ⇒ 用无界 g 起步再迭代一次
    for _ in range(2):
        cap = np.maximum(scn - g, 0.0)
        g = plan_capped(p, scn, probs, soc, NU_END, KAPPA, cap)
    net = year.ell[j] - year.r[j]
    c = np.zeros(T); d = np.zeros(T); u = np.zeros(T); w = np.zeros(T)
    s = np.empty(T + 1); s[0] = soc
    for t in range(T):
        cap_now = max(net[t] - g[t], 0.0)
        cap_fut = np.maximum(scn[:, t + 1:] - g[t + 1:], 0.0)
        c_t, d_t, u_t, w_t = step_capped(p, net, g, scn, probs, t, s[t],
                                        NU_END, KAPPA, (cap_now, cap_fut))
        if c_t > 0 and d_t > 0:
            x = min(c_t, d_t / ETA**2)
            c_t -= x; d_t -= ETA**2 * x; w_t += x * (1 - ETA**2)
        c[t], d[t], u[t], w[t] = c_t, d_t, u_t, w_t
        s[t + 1] = s[t] + ETA * c_t - d_t / ETA
    rows_out.append((j, g.copy(), c, d, u, w, s.copy(),
                     float(p @ g + KAPPA * (p @ u))))
    soc = s[T]
    fcasts.append((ell_hat, r_hat))
    if (j + 1) % 20 == 0:
        print("day %d done, SOC %.0f" % (j + 1, soc), flush=True)

rep = [r for r in rows_out if r[0] >= 31]
U = np.array([r[4] for r in rep]); W = np.array([r[5] for r in rep])
C = np.array([r[2] for r in rep]); S = np.array([r[6] for r in rep])
D = np.array([r[3] for r in rep]); G = np.array([r[1] for r in rep])
Z = sum(r[7] for r in rep)
P = np.tile(p, (len(rep), 1))
mask = U > 1e-6
r_sum = np.array([year.r[r[0]] for r in rep]).sum()
print("\n=== u 只补缺口（%d 天）" % len(rep))
print("费用 %.0f 元" % Z)
print("紧急电量 %.0f kWh，费用 %.0f 元（占 %.2f%%）"
      % (U.sum(), KAPPA * (U * P).sum(), 100 * KAPPA * (U * P).sum() / Z))
print("缺口频率 %.2f%%   u∧c 时段 %d" % (100 * mask.mean(),
                                        (mask & (C > 1e-6)).sum()))
print("SOC 日内均值 %.1f   S0 均值 %.1f   S144 均值 %.1f"
      % (S[:, 1:].mean(), S[:, 0].mean(), S[:, 144].mean()))
print("弃电 %.0f kWh（率 %.2f%%）" % (W.sum(), 100 * W.sum() / r_sum))
band = [("谷 p<0.5", P < 0.5), ("平", (P >= 0.5) & (P <= 1.0)),
        ("峰 p>1.0", P > 1.0)]
for nm, mb in band:
    print("  %-10s 电量占比 %.1f%%" % (nm, 100 * U[mask & mb].sum() / U.sum()))
print("  盈余时段(n<0) 占比 %.1f%%"
      % (100 * U[mask & (np.array([year.ell[r[0]] - year.r[r[0]]
                                  for r in rep]) < 0)].sum() / U.sum()))
soc_prev = S[:, :-1]
print("  SOC<2000 电量占比 %.1f%%；SOC>8000 %.1f%%"
      % (100 * U[mask & (soc_prev < 2000)].sum() / U.sum(),
         100 * U[mask & (soc_prev > 8000)].sum() / U.sum()))

NETa = np.array([year.ell[r[0]] - year.r[r[0]] for r in rep])
hi_soc = mask & (soc_prev > 8000)
print("\n--- 加上界后 SOC>8000 仍紧急购电的时段解剖")
print("  时段 %d，电量 %.0f kWh，均价 %.4f" % (hi_soc.sum(), U[hi_soc].sum(),
                                             (P[hi_soc] * U[hi_soc]).sum()
                                             / U[hi_soc].sum()))
print("  其中放电顶功率上限 %.1f%%（按电量），放电=0 %.1f%%，g=0 %.1f%%"
      % (100 * U[hi_soc & (D > E_MAX_PER - 1e-3)].sum() / U[hi_soc].sum(),
         100 * U[hi_soc & (D < 1e-6)].sum() / U[hi_soc].sum(),
         100 * U[hi_soc & (G < 1e-6)].sum() / U[hi_soc].sum()))
rem_max = np.array([p[t:].max() for t in range(T)])
tie = hi_soc & (P >= np.tile(rem_max, (len(rep), 1)) - 1e-9)
print("  当步已是当日剩余最高价（放电时机无差异/退化）的电量占比 %.1f%%"
      % (100 * U[tie].sum() / U[hi_soc].sum()))
np.savez("scratch_q2_ucap_slots.npz", U=U, W=W, C=C, D=D, G=G, S=S, NET=NETa)
