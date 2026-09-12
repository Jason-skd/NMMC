"""题3 规划层与 MPC 尾部 LP（稀疏 HiGHS）。

两个入口共用题2 档 §5 的物理模板（nmmc.model.q2 的布局约定，g → a）：
    solve_stage  阶段 m 两阶段场景 LP（文档 §7.1）——本块承诺 a(B_m)
                 （m=0 为全天计划 g）是 here-and-now 共享变量，更晚块的
                 试探 a^(s) 是 wait-and-see（偏差费计入期望，正确预期
                 "届时会重定"）；δ± 自动互斥（性质 5），纯 LP。
    solve_tail   MPC 剩余时域 LP（文档 §8）——已锁定段注入为参数，
                 未锁定段 a 为内部变量并按偏差费计（与 solve_stage 的
                 wait-and-see 同一记账）。

变量布局（题2 档约定）：每情景块 [c(T'), d(T'), u(T'), w(T'), S(T'+1)]；
偏差量 δ+/δ− 仅对可调整时段存在。S 初值由 bounds 钉在锚点。

题3 档常数（docs/题3变量定义.md §1 集合与 §2.3 倍率）：修订时刻 τ_m、
块结构 B_m 与结算倍率 β± 定义于此。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.model.lp import (E_MAX_PER, S_MAX, S_MIN, T, solve_eq_lp,
                           soc_recursion)

BETA_MINUS = 0.5        # 违约倍率 β⁻（计划高于调整的部分，文档 §2.3）
BETA_PLUS = 1.5         # 加价倍率 β⁺（调整高于计划的部分）
VINTAGE_HOURS = (0, 6, 12, 18)   # 修订时刻 τ_m（小时），m = 0..3
BLOCK_LEN = 36                    # 修订块 B_m 的时段数（6 小时 × 6 时段/h）
BLOCK_STARTS = (0, 36, 72, 108)   # B_m 的 0 基起始时段 = 6·τ_m


@dataclass
class StageResult:
    """阶段 m 子问题最优解。

    Attributes:
        commit: 本阶段后的已承诺注入，shape (144,)。m=0 为计划 g（B_0 段
            即最终量）；m≥1 为"前缀已定 + 本块 here-and-now"。
        a_scn: 逐情景试探注入，shape (N_s, 144)——前缀与本块为 commit，
            更晚块为 wait-and-see 解（PI 诊断取 a_scn[0]）。
        z_tilde: 规划目标值 Z̃（期望口径，含末端估值，不是现金费用）。
    """

    commit: np.ndarray
    a_scn: np.ndarray
    z_tilde: float


def solve_stage(p: np.ndarray, g: np.ndarray | None, committed: np.ndarray | None,
                scn: np.ndarray, probs: np.ndarray, m: int, s_anchor: float,
                nu_end: float, kappa: float) -> StageResult:
    """解阶段 m 两阶段场景 LP（文档 §7.1）。

    Args:
        p: 逐时段电价，元/kWh，shape (144,)。
        g: 0:00 计划购电量（计费基准）；m=0 时为 None（g 是本阶段决策）。
        committed: m≥1 时的已承诺注入前缀（[0, 6τ_m) 段有效）；m=0 为 None。
        scn: 净负荷场景 n^(s)，kWh，shape (N_s, 144)（已实现前缀为真值）。
        probs: 情景概率 π_s。
        m: 修订阶段 0..3。
        s_anchor: 当前真实储电量（SOC 重锚，文档 §5.4）。
        nu_end: 日末储电量估值 ν^end。
        kappa: 紧急购电倍率 κ。

    Returns:
        StageResult: 本块承诺、逐情景试探注入与规划目标值。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    h0 = 36 * m
    n_s = len(scn)
    # here-and-now 段 [h0, he) 与可调整段 [fe, T)：m=0 的 hn 是全天计划 g
    #（费用只计不可调整的 [0,36) 段，其余段 g 仅为偏差基准）；m≥1 的 hn
    # 是本块 36 个承诺，与 g 单次比对（共享 δ±）。
    he = T if m == 0 else h0 + 36
    fe = 36 if m == 0 else h0 + 36
    n_hn = he - h0
    n_free = T - fe
    nh = T - h0
    n_blk = 3 * n_free + 5 * nh + 1
    n = n_hn + (2 * n_hn if m else 0) + n_s * n_blk
    o_hn = 0
    o_dp = n_hn if m else 0                  # 共享 δ+/δ−（仅 m≥1）
    o_s = o_dp + (2 * n_hn if m else 0)      # 首情景块起点
    obj = np.zeros(n)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    b: list[float] = []
    row = 0

    # here-and-now：基础购电费（m=0 只计 [0,36) 不可调整段；m≥1 计全块）
    cost_hn = p[:36] if m == 0 else p[h0:he]
    obj[o_hn:o_hn + len(cost_hn)] = cost_hn
    if m:
        # 共享偏差：hn − δ+ + δ− = g（B_m 段，单次比对）
        tt = np.arange(n_hn)
        r0 = row + np.arange(n_hn)
        rows += [r0, r0, r0]
        cols += [o_hn + tt, o_dp + tt, o_dp + n_hn + tt]
        vals += [np.ones(n_hn), -np.ones(n_hn), np.ones(n_hn)]
        b += list(g[h0:he])
        obj[o_dp:o_dp + n_hn] = 0.5 * p[h0:he]
        obj[o_dp + n_hn:o_dp + 2 * n_hn] = 0.5 * p[h0:he]
        row += n_hn

    for s in range(n_s):
        o = o_s + s * n_blk
        o_af, o_dpf = o, o + n_free
        o_dmf = o_dpf + n_free
        o_c = o_dmf + n_free
        o_d = o_c + nh
        o_u = o_d + nh
        o_w = o_u + nh
        o_st = o_w + nh
        pi = probs[s]
        # 试探偏差：af − δ+ + δ− = g。m=0 的基准是正在决策的 g 变量
        #（与 hn 的变量间等式）；m≥1 的基准是已锁定的常数计划。
        if n_free:
            tf = np.arange(n_free)
            rf = row + np.arange(n_free)
            if m == 0:
                rows += [rf, rf, rf, rf]
                cols += [o_af + tf, o_dpf + tf, o_dmf + tf,
                         o_hn + fe + tf]
                vals += [np.ones(n_free), -np.ones(n_free), np.ones(n_free),
                         -np.ones(n_free)]
                b += [0.0] * n_free
            else:
                rows += [rf, rf, rf]
                cols += [o_af + tf, o_dpf + tf, o_dmf + tf]
                vals += [np.ones(n_free), -np.ones(n_free), np.ones(n_free)]
                b += list(g[fe:])
            obj[o_af:o_af + n_free] = pi * p[fe:]
            obj[o_dpf:o_dpf + n_free] = pi * 0.5 * p[fe:]
            obj[o_dmf:o_dmf + n_free] = pi * 0.5 * p[fe:]
            row += n_free
        # 供需平衡：注入 + d + u − c − w = n（注入 = hn 或 af）
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
        n_lock = fe - h0                                 # 平衡式取 hn 的时段数
        if n_lock:
            rows.append(rb[:n_lock])
            cols.append(o_hn + tt[:n_lock])
            vals.append(np.ones(n_lock))
        if n_free:
            rows.append(rb[n_lock:])
            cols.append(o_af + tt[n_lock:] - n_lock)
            vals.append(np.ones(n_free))
        b += list(scn[s, h0:])
        row += nh
        # SOC 递推：S_{τ+1} − S_τ − ηc + d/η = 0
        soc_rows, soc_cols, soc_vals = soc_recursion(nh, row, o_st, o_c, o_d)
        rows += soc_rows
        cols += soc_cols
        vals += soc_vals
        b += [0.0] * nh
        row += nh
        # 物理边界与目标（紧急 + 末端估值）
        obj[o_u:o_u + nh] = pi * kappa * p[h0:]
        obj[o_st + nh] = -pi * nu_end
        hi[o_c:o_c + nh] = E_MAX_PER
        hi[o_d:o_d + nh] = E_MAX_PER
        lo[o_st:o_st + nh + 1] = S_MIN
        hi[o_st:o_st + nh + 1] = S_MAX
        lo[o_st] = hi[o_st] = s_anchor

    x, _, _ = solve_eq_lp(obj, rows, cols, vals, np.array(b), lo, hi)
    commit = (committed.copy() if committed is not None
              else np.zeros(T))
    commit[h0:he] = x[o_hn:o_hn + n_hn]
    a_scn = np.repeat(commit[None, :], n_s, axis=0)
    for s in range(n_s):
        o = o_s + s * n_blk
        if n_free:
            a_scn[s, fe:] = x[o:o + n_free]
    return StageResult(commit=commit, a_scn=a_scn,
                       z_tilde=float(obj @ x))


def solve_tail(p: np.ndarray, net: np.ndarray, committed: np.ndarray,
               free_mask: np.ndarray, g: np.ndarray, s_start: float,
               nu_end: float, kappa: float) -> tuple[np.ndarray, ...]:
    """解 MPC 剩余时域 LP（文档 §8 供给分层）。

    已锁定段注入为参数（移平衡式右端）；未锁定段 a 为变量，按
    p·a + ½p|a−g| 计费（与规划层 wait-and-see 同一记账）。

    Args:
        p: 电价，与 net 等长的尾部切片。
        net: 净负荷路径（首时段真值 + 参考路径）。
        committed: 各时段注入参数（free_mask 为 True 处取值被忽略）。
        free_mask: 未锁定时段掩码，与 net 等长。
        g: 计划购电量切片（未锁定段的偏差基准）。
        s_start: 锚定储电量。
        nu_end / kappa: 末端估值 / 紧急倍率。

    Returns:
        (c, d, u, w, S)：尾部补救量与储电量轨迹，S shape (n+1,)。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    n_t = len(net)
    free = np.flatnonzero(free_mask)
    n_f = len(free)
    o_c, o_d = 0, n_t
    o_u, o_w = 2 * n_t, 3 * n_t
    o_st = 4 * n_t
    o_af = o_st + n_t + 1
    n = o_af + 3 * n_f
    obj = np.zeros(n)
    obj[o_u:o_u + n_t] = kappa * p
    obj[o_st + n_t] = -nu_end
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    hi[o_c:o_c + n_t] = E_MAX_PER
    hi[o_d:o_d + n_t] = E_MAX_PER
    lo[o_st:o_st + n_t + 1] = S_MIN
    hi[o_st:o_st + n_t + 1] = S_MAX
    lo[o_st] = hi[o_st] = s_start
    tt = np.arange(n_t)
    rows = [tt, tt, tt, tt]
    cols = [o_c + tt, o_d + tt, o_u + tt, o_w + tt]
    vals = [-np.ones(n_t), np.ones(n_t), np.ones(n_t), -np.ones(n_t)]
    soc_rows, soc_cols, soc_vals = soc_recursion(n_t, n_t, o_st, o_c, o_d)
    rows += soc_rows
    cols += soc_cols
    vals += soc_vals
    b = list(np.where(free_mask, net, net - committed)) + [0.0] * n_t
    if n_f:
        # 未锁定段：注入变量 af 进平衡式；偏差 af − δ+ + δ− = g（新行，
        # 排在 SOC 递推之后）
        rows.append(tt[free])
        cols.append(o_af + np.arange(n_f))
        vals.append(np.ones(n_f))
        rf = 2 * n_t + np.arange(n_f)
        rows += [rf, rf, rf]
        cols += [o_af + np.arange(n_f), o_af + n_f + np.arange(n_f),
                 o_af + 2 * n_f + np.arange(n_f)]
        vals += [np.ones(n_f), -np.ones(n_f), np.ones(n_f)]
        b += list(g[free])
        obj[o_af:o_af + n_f] = p[free]
        obj[o_af + n_f:o_af + 2 * n_f] = 0.5 * p[free]
        obj[o_af + 2 * n_f:] = 0.5 * p[free]
    x, _, _ = solve_eq_lp(obj, rows, cols, vals, np.array(b), lo, hi)
    return (x[o_c:o_c + n_t], x[o_d:o_d + n_t], x[o_u:o_u + n_t],
            x[o_w:o_w + n_t], x[o_st:o_st + n_t + 1])


def settle_cost(p: np.ndarray, g: np.ndarray, a: np.ndarray,
                u: np.ndarray, kappa: float = 5.0) -> float:
    """现金费用 Z = Σ[p·a + ½p|a−g|] + κΣp·u（唯一上报口径，文档 §9.1）。"""
    return float(p @ a + 0.5 * (p @ np.abs(a - g)) + kappa * (p @ u))


def beta_check() -> float:
    """结算式恒等自检（文档 C3-3）：三段计价与 |·| 形式在随机 (g, a) 上
    数值一致的最大残差（应 < 1e-9 元）。"""
    rng = np.random.default_rng(0)
    gg, aa = rng.random(50) * 200, rng.random(50) * 200
    p = rng.random(50) + 0.1
    three = (p * np.minimum(gg, aa)
             + BETA_MINUS * p * np.maximum(gg - aa, 0)
             + BETA_PLUS * p * np.maximum(aa - gg, 0))
    return float(np.abs(three - p * aa - 0.5 * p * np.abs(aa - gg)).max())
