"""题3 执行层：嵌套双环（外环锁块、内环 MPC）+ 全年滚动 + PI/事后诊断。

文档 §8 与 §10：外环每日 4 次在 τ_m 解阶段 LP 并锁定本块承诺 a(B_m)
（偏差费即时入账）；内环每 10 分钟以真实 SOC 重锚、当前时段真值 +
参考路径解剩余时域 LP（未锁定段按偏差费内部决策）、只执行当步。
跨日唯一反馈是真实路径 S_144 与当日实际值；诚实回测纪律同题2 档 §9。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.lp import ETA, T
from nmmc.q2.data import ALPHA_RT, KAPPA, NU_END, S0_INIT, YearData
from nmmc.q2.forecast import ForecastConfig, forecast_load, reference_path
from nmmc.q2.lp import solve_planning, solve_recourse
from nmmc.q3.data import YearForecast
from nmmc.q3.forecast import (Q3Config, build_scenarios, fusion_weights,
                              hour_powers, load_point, pv_point)
from nmmc.q3.lp import settle_cost, solve_stage, solve_tail


@dataclass
class DayResult:
    """单日回测结果。

    Attributes:
        j: 0 基日索引。
        g: 0:00 计划购电量（计费基准），shape (144,)。
        a: 全天最终注入（B_0 段 = g，B_1..3 段为各阶段锁定值），shape (144,)。
        ell_hat / r_hat: 0:00 点预报（诊断用）。
        ns: 阶段 0 情景数。
        c / d / u / w: MPC 真实路径补救量（母线侧），shape (144,)。
        S: 真实路径储电量，shape (145,)。
        z: 当日现金费用 Σ[p·a + ½p|a−g|] + κΣp·u，元。
        z_expost: 事后最优补救的现金费用（同 g、a，同真路径），元。
        z_pi: 完美信息（0:00 已知真值）的现金费用，元。
    """

    j: int
    g: np.ndarray
    a: np.ndarray
    ell_hat: np.ndarray
    r_hat: np.ndarray
    ns: int
    c: np.ndarray
    d: np.ndarray
    u: np.ndarray
    w: np.ndarray
    S: np.ndarray
    z: float
    z_expost: float
    z_pi: float


def run_mpc_segment(p: np.ndarray, net: np.ndarray, committed: np.ndarray,
                    g: np.ndarray, ref: np.ndarray, c: np.ndarray,
                    d: np.ndarray, u: np.ndarray, w: np.ndarray,
                    s: np.ndarray, h0: int, h1: int, nu_end: float,
                    kappa: float) -> None:
    """内环 MPC：就地执行本块时段 [h0, h1)，[0, h0) 段保留前段结果。

    当前时段用真值 net，未来用参考路径 ref；本块已锁定（committed 的
    [h0, h1) 段），更晚块为内部变量（文档 §8 供给三层）。

    Args:
        p: 逐时段电价，shape (144,)。
        net: 当日真实净负荷，shape (144,)。
        committed: 截至本块的已承诺注入（[h1, 144) 段取值未定、被掩码）。
        g: 0:00 计划（未锁定段的偏差基准）。
        ref: α^rt 分位参考路径，shape (144,)。
        c / d / u / w / s: 全日数组（就地写入 [h0, h1) 与 s[h0+1..h1]，
            s[h0] 为本块起点真实储电量）。
        h0 / h1: 本块起止时段（0 基，左闭右开）。
        nu_end / kappa: 末端估值 / 紧急倍率。
    """
    for t in range(h0, h1):
        slots = np.arange(t, T)
        free = slots >= h1
        comm = np.where(free, 0.0, committed[slots])
        tail_net = np.concatenate([net[t:t + 1], ref[t + 1:]])
        c_t, d_t, u_t, w_t, _ = solve_tail(p[t:], tail_net, comm, free,
                                           g[t:], s[t], nu_end, kappa)
        if c_t[0] > 0.0 and d_t[0] > 0.0:
            # 性质 3 同款：HiGHS 可能返回 c·d>0 的等价最优顶点，按非等量
            # 缩减归一化，SOC 轨迹与费用严格不变（继承 q2 执行层处理）。
            x = min(c_t[0], d_t[0] / ETA**2)
            c_t[0] -= x
            d_t[0] -= ETA**2 * x
            w_t[0] += x * (1 - ETA**2)
        c[t], d[t], u[t], w[t] = c_t[0], d_t[0], u_t[0], w_t[0]
        s[t + 1] = s[t] + ETA * c[t] - d[t] / ETA


def rollout(year: YearData, fc: YearForecast, p: np.ndarray,
            typ_ell: np.ndarray, typ_r: np.ndarray, cfg: Q3Config,
            nu_end: float = NU_END, kappa: float = KAPPA,
            alpha_rt: float = ALPHA_RT, days: int | None = None
            ) -> list[DayResult]:
    """全年诚实回测（文档 §10）：外环四阶段 → 内环 MPC → SOC 链推进。

    Args:
        year: 全年数据（按日解锁使用）。
        fc: 附件3 官方光伏预报。
        p: 逐时段电价（附件1，全年相同）。
        typ_ell / typ_r: 附件1 典型日曲线（冷启动先验）。
        cfg: 题3 预报层配置。
        nu_end / kappa / alpha_rt: 末端估值 / 紧急倍率 / MPC 参考分位。
        days: 只滚动前 N 天（含暖机）；缺省全年 365。

    Returns:
        逐日结果列表，暖机期完整包含（报告期切片在 report 层）。
    """
    rng = np.random.default_rng(cfg.seed)
    cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
    r_hour = hour_powers(year.r)
    hist: list[list[tuple[np.ndarray, np.ndarray]]] = []
    soc = S0_INIT
    out: list[DayResult] = []
    n_days = days or len(year.dates)
    for j in range(n_days):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, typ_ell)
        net = year.ell[j] - year.r[j]
        g = None
        committed = None
        stage_fcasts: list[tuple[np.ndarray, np.ndarray]] = []
        z_pi = np.inf
        ns0 = 1
        c, d, u, w = (np.zeros(T) for _ in range(4))
        s = np.zeros(T + 1)
        s[0] = soc
        for m in range(4):
            h0, h1 = 36 * m, 36 * (m + 1)
            r_hat = pv_point(cfg, year, fc, j, m, typ_r, weights)
            ell_hat = ell0 if m == 0 else load_point(cfg, ell0, year, j, m)
            stage_fcasts.append((ell_hat, r_hat))
            scn = build_scenarios(cfg, year, hist, j, m, ell_hat, r_hat, rng)
            probs = np.full(len(scn), 1.0 / len(scn))
            res = solve_stage(p, g, committed, scn, probs, m, s[h0],
                              nu_end, kappa)
            if m == 0:
                g = res.commit
                ns0 = len(scn)
                # PI 诊断：0:00 已知真值 ⇒ 确定性全天 LP（题2 模板，无偏差
                # 机制——完美信息下 a≡g，一次定准）
                pi = solve_planning(p, net[None, :], np.ones(1), s[0],
                                    nu_end, kappa)
                u_pi = solve_recourse(p, net, pi.g, s[0], nu_end, kappa).u
                z_pi = settle_cost(p, pi.g, pi.g, u_pi, kappa)
            committed = res.commit
            ref = reference_path(scn, alpha_rt)
            run_mpc_segment(p, net, committed, g, ref, c, d, u, w, s,
                            h0, h1, nu_end, kappa)
        u_exp = solve_recourse(p, net, committed, soc, nu_end, kappa).u
        out.append(DayResult(
            j=j, g=g, a=committed, ell_hat=ell0, r_hat=stage_fcasts[0][1],
            ns=ns0, c=c, d=d, u=u, w=w, S=s,
            z=settle_cost(p, g, committed, u, kappa),
            z_expost=settle_cost(p, g, committed, u_exp, kappa),
            z_pi=z_pi))
        soc = s[T]
        hist.append(stage_fcasts)
        if (j + 1) % 31 == 0:
            print(f"已回测 {j + 1}/{n_days} 天，年末 SOC = {soc:.1f} kWh",
                  flush=True)
    return out
