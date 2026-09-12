"""题2 规划层与补救 LP（稀疏 HiGHS）。

三个入口共用物理模板 X(g, S_0, n)（文档 §5）：
    solve_planning  日前两阶段场景 LP（SAA）——g 为 here-and-now 共享变量，
                    情景块 (c,d,u,w,S) 独立；计划购电费不进情景平均。
    solve_recourse  单路径补救 LP——g 为参数（平衡式右端移项），
                    事后最优（全天）与 MPC 剩余时域（尾部切片）共用。
    solve_stochastic_step  在线随机 MPC 的单步两阶段 LP——当步决策
                    here-and-now 全情景共享，剩余时域按情景分支（§7.2）。

变量布局（每情景块）：[c(T'), d(T'), u(T'), w(T'), S(T'+1)]。
等式约束（每块）：T' 供需平衡 g+d+u−c−w = n 与 T' SOC 递推
S_t − S_{t−1} − ηc_t + d_t/η = 0。S_0 由 bounds 钉在给定初值。
物理常数、SOC 递推块与稀疏求解封装沿用 nmmc.model.lp。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.model.lp import (E_MAX_PER, ETA, S_MAX, S_MIN, T, solve_eq_lp,
                           soc_recursion)

KAPPA = 5.0    # 紧急购电倍率
NU_END = 0.472  # 日末储电量估值锚，元/kWh（文档 §2.3）


def emergency_cap(net: np.ndarray, g: np.ndarray) -> np.ndarray:
    """紧急购电"只补缺口"上界 u_t ≤ (n_t − g_t)⁺（文档 §5.1，假设 A6′）。

    题面语义：紧急购电是断供兜底，按实际缺口计量，不是 5 倍价的采购
    窗口。给定已锁定的 g，该上界是常数，模型仍为纯 LP；并且它**等价于**
    禁止"紧急购电同时充电"（性质 8′）：由平衡式 u = (n−g) + c + w − d 与
    充放互斥 c·d = 0，u ≤ (n−g)⁺ 蕴含 u·c = 0。恒可行性不变（性质 1 的
    构造 c = d = 0、u = (n−g)⁺、w = (g−n)⁺ 恰好落在上界上）。

    Args:
        net: 净负荷路径 n（真值或情景），kWh。
        g: 已锁定的计划购电量，与 net 同形。

    Returns:
        逐时段上界 (n−g)⁺，与 net 同形。
    """
    return np.maximum(net - g, 0.0)


@dataclass
class PlanResult:
    """日前两阶段场景 LP 最优解。

    Attributes:
        g: 计划购电量，kWh，shape (144,)。
        z_tilde: 规划目标值 Z̃（含末端估值虚拟记账项，不是现金费用，文档 §6）。
    """

    g: np.ndarray
    z_tilde: float


@dataclass
class PathResult:
    """单路径补救 LP 最优解。

    Attributes:
        c / d / u / w: 补救量（母线侧），kWh，shape (T',)。
        S: 储电量，kWh，shape (T'+1)，S[0] 为给定锚。
        obj: LP 目标值 κΣp·u − ν·S_end（诊断用，不是现金费用）。
    """

    c: np.ndarray
    d: np.ndarray
    u: np.ndarray
    w: np.ndarray
    S: np.ndarray
    obj: float


def solve_planning(p: np.ndarray, scn: np.ndarray, probs: np.ndarray,
                   s0: float, nu_end: float, kappa: float,
                   u_cap: np.ndarray | None = None,
                   p_scn: np.ndarray | None = None) -> PlanResult:
    """解日前两阶段场景 LP（文档 §6）。

    Args:
        p: 一阶段计价电价（题4 为情景均值折叠价 p̂ = Σπ_s p^(s)，性质 2
            使期望计划费精确折叠），元/kWh，shape (144,)。
        scn: 净负荷场景 n^(s)，kWh，shape (N_s, 144)。
        probs: 情景概率 π_s（等权 1/N_s）。
        s0: 当日 0:00 储电量（给定状态）。
        nu_end: 日末储电量估值 ν^end。
        kappa: 紧急购电倍率 κ。
        u_cap: 情景紧急购电上界，shape (N_s, 144)；None 为 v5 旧口径
            （无上界）。上界依赖 g，由调用方按序列线性化给定（§6.1）。
        p_scn: 情景电价 p^(s)（题4 读法 R），shape (N_s, 144)——紧急项
            不可折叠、逐情景计价（性质 2 推论 2）；None 即常数价口径
            （题2 / 读法 D / 附件1 退化，各情景用 p，逐位不变）。

    Returns:
        PlanResult: 计划购电量 g 与规划目标值 Z̃。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    n_s = len(scn)
    blk = 5 * T + 1
    n = T + n_s * blk

    # 目标：Z̃ = Σp·g（计划费，不进情景平均）+ Σ_s π_s[κΣp·u^(s) − ν·S_144^(s)]
    obj = np.zeros(n)
    obj[:T] = p

    # 变量域：g ≥ 0；情景块内 0 ≤ c,d ≤ E_MAX_PER、u,w ≥ 0、S ∈ [S_MIN, S_MAX]
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    b = np.zeros(2 * T * n_s)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    tt = np.arange(T)
    for s in range(n_s):
        o_c, o_d = T + s * blk, T + s * blk + T
        o_u, o_w = o_d + T, o_d + 2 * T
        o_s = o_w + T
        # 平衡：g + d + u − c − w = n_s
        r = s * 2 * T + tt
        rows += [r, r, r, r, r]
        cols += [tt, o_c + tt, o_d + tt, o_u + tt, o_w + tt]
        vals += [np.ones(T), -np.ones(T), np.ones(T), np.ones(T), -np.ones(T)]
        b[r] = scn[s]
        # SOC 递推：S_t − S_{t−1} − ηc + d/η = 0
        soc_rows, soc_cols, soc_vals = soc_recursion(T, s * 2 * T + T,
                                                     o_s, o_c, o_d)
        rows += soc_rows
        cols += soc_cols
        vals += soc_vals
        # 情景目标（π_s·κΣp^(s)·u − π_s·ν·S_144）与变量域（S_0 锚定 s0）
        pi = probs[s]
        obj[o_u:o_u + T] = pi * kappa * (p if p_scn is None else p_scn[s])
        obj[o_s + T] = -pi * nu_end
        lo[o_c:o_c + T] = 0.0
        hi[o_c:o_c + T] = E_MAX_PER
        lo[o_d:o_d + T] = 0.0
        hi[o_d:o_d + T] = E_MAX_PER
        lo[o_s:o_s + T + 1] = S_MIN
        hi[o_s:o_s + T + 1] = S_MAX
        lo[o_s] = hi[o_s] = s0
        if u_cap is not None:
            hi[o_u:o_u + T] = u_cap[s]

    # 求解；Z̃ = obj @ x（含虚拟记账项，非现金费用）
    x, _, _ = solve_eq_lp(obj, rows, cols, vals, b, lo, hi)
    return PlanResult(g=x[:T], z_tilde=float(obj @ x))


def solve_recourse(p: np.ndarray, net: np.ndarray, g: np.ndarray,
                   s_start: float, nu_end: float, kappa: float,
                   u_cap: np.ndarray | None = None) -> PathResult:
    """解单路径补救 LP：min κΣp·u − ν·S_end（文档 §7.3 模板）。

    g 为已锁定的参数（移到平衡式右端），S_0 锚定 s_start。事后最优传
    全天数组；MPC 传剩余时域尾部切片。

    Args:
        p: 电价，元/kWh，与 net 等长（尾部时传 p[t:]）。
        net: 净负荷路径 n，kWh。
        g: 已锁定的计划购电量（与 net 等长的切片）。
        s_start: 锚定储电量。
        nu_end: 日末（或时域末端）储电量估值。
        kappa: 紧急购电倍率 κ。
        u_cap: 紧急购电上界（与 net 等长），None 为 v5 旧口径（无上界）；
            新口径取 emergency_cap(net, g)。

    Returns:
        PathResult: 补救量、储电量轨迹与目标值。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    n_t = len(net)
    blk = 5 * n_t + 1
    o_c, o_d, o_u, o_w, o_s = 0, n_t, 2 * n_t, 3 * n_t, 4 * n_t

    # 目标：min κΣp·u − ν·S_end（诊断口径，非现金费用）
    obj = np.zeros(blk)
    obj[o_u:o_u + n_t] = kappa * p
    obj[o_s + n_t] = -nu_end

    # 变量域：0 ≤ c,d ≤ E_MAX_PER；u,w ≥ 0；S ∈ [S_MIN, S_MAX]，S_0 锚定 s_start
    lo = np.zeros(blk)
    hi = np.full(blk, np.inf)
    hi[o_c:o_c + n_t] = E_MAX_PER
    hi[o_d:o_d + n_t] = E_MAX_PER
    lo[o_s:o_s + n_t + 1] = S_MIN
    hi[o_s:o_s + n_t + 1] = S_MAX
    lo[o_s] = hi[o_s] = s_start
    if u_cap is not None:
        hi[o_u:o_u + n_t] = u_cap

    # 平衡：d + u − c − w = n − g（g 已锁定，移右端）
    tt = np.arange(n_t)
    rows = [tt, tt, tt, tt]
    cols = [o_c + tt, o_d + tt, o_u + tt, o_w + tt]
    vals = [-np.ones(n_t), np.ones(n_t), np.ones(n_t), -np.ones(n_t)]

    # SOC 递推：S_t − S_{t−1} − ηc + d/η = 0
    soc_rows, soc_cols, soc_vals = soc_recursion(n_t, n_t, o_s, o_c, o_d)
    rows += soc_rows
    cols += soc_cols
    vals += soc_vals
    b = np.concatenate([net - g, np.zeros(n_t)])
    x, _, _ = solve_eq_lp(obj, rows, cols, vals, b, lo, hi)
    return PathResult(
        c=x[o_c:o_c + n_t], d=x[o_d:o_d + n_t],
        u=x[o_u:o_u + n_t], w=x[o_w:o_w + n_t],
        S=x[o_s:o_s + n_t + 1], obj=float(obj @ x))


@dataclass
class StepResult:
    """在线随机 MPC 单步两阶段 LP 最优解（文档 §7.2）。

    Attributes:
        c / d / u / w: 当步补救量（母线侧），kWh——here-and-now 决策，
            全部情景共享；执行层只执行本时段。
        obj: 步进目标值 κp_t·u_t + Σπ_s(κΣp·u^s − ν·S^s_end)（诊断用，
            不是现金费用）。
    """

    c: float
    d: float
    u: float
    w: float
    obj: float


def solve_stochastic_step(p: np.ndarray, net: np.ndarray, g: np.ndarray,
                          scn: np.ndarray, probs: np.ndarray, t: int,
                          s_prev: float, nu_end: float, kappa: float,
                          u_cap: np.ndarray | None = None,
                          p_scn: np.ndarray | None = None) -> StepResult:
    """解在线随机 MPC 的单步两阶段 LP（文档 §7.2）。

    第一阶段（here-and-now）：当步 (c,d,u,w) 与 S_{t+1} 全情景共享，
    平衡用真值 n_t、SOC 递推用真实 S_t——非预期性的紧凑写法；第二阶段
    （wait-and-see）：剩余 m = T−1−t 段每情景一块 (c^s,d^s,u^s,w^s,S^s)，
    S^s 锚定共享的 S_{t+1}。目标 κp_t·u_t + Σ_s π_s(κΣ_{τ>t} p_τ·u^s_τ
    − ν·S^s_end)。m = 0（末时段）无未来分支，−ν·S_{t+1} 显式入目标
    （自检 C15：否则求解器对"放电后弃掉"无差异，末段白弃电）。

    Args:
        p: 当步计价电价——题4 读法 R 下为当日实现价 p_t^(j)（交易时刻
            可见，题面"交易时刻电价"），元/kWh，shape (144,)。
        net: 当日真实净负荷，kWh，shape (144,)。
        g: 已锁定的计划购电量，shape (144,)。
        scn: 0:00 净负荷情景 n^(s)，shape (N_s, 144)。
        probs: 情景概率 π_s，shape (N_s,)。
        t: 当前时段（0 基）。
        s_prev: 真实储电量 S_t（时段初）。
        nu_end: 时域末端储电量估值 ν^end。
        kappa: 紧急购电倍率 κ。
        u_cap: 情景紧急购电上界，shape (N_s, 144)，None 为 v5 旧口径。
            当步上界按**真值**口径取 (n_t − g_t)⁺（当步净负荷已实现，
            与情景无关）；剩余时域第 s 块取 u_cap[s, t+1:]。
        p_scn: 未来分支的情景电价 p^(s)（τ > t 段取 p_scn[s, t+1:]，
            题4 档 §7）；None 即常数价口径（题2，各情景用 p）。

    Returns:
        StepResult: 当步决策与步进目标值。S_{t+1} 由执行层按真实递推
        重建，不经求解器往返。

    Raises:
        RuntimeError: LP 不可行或未达最优（§7.7 恒可行性保证不发生，
        出现即为实现错误）。
    """
    m = T - 1 - t
    cap_now = np.inf if u_cap is None else max(net[t] - g[t], 0.0)
    if m == 0:
        # 末时段：无未来分支，末端估值直接挂共享 S_{t+1}（C15）
        # 变量布局 x = [c, d, u, w, S_{t+1}]；目标 = κp_t·u_t − ν·S_{t+1}
        obj = np.array([0.0, 0.0, kappa * p[t], 0.0, -nu_end])
        # 平衡 d+u−c−w = n_t−g_t 与 SOC 递推 S_{t+1}−ηc+d/η = S_t
        rows = [np.array([0, 0, 0, 0]), np.array([1, 1, 1])]
        cols = [np.array([1, 2, 3, 0]), np.array([4, 0, 1])]
        vals = [np.array([1.0, 1.0, -1.0, -1.0]),
                np.array([1.0, -ETA, 1.0 / ETA])]
        b = np.array([net[t] - g[t], s_prev])
        # 变量域：0 ≤ c,d ≤ E_MAX_PER；0 ≤ u ≤ cap；w ≥ 0；S ∈ [S_MIN,S_MAX]
        lo = np.array([0.0, 0.0, 0.0, 0.0, S_MIN])
        hi = np.array([E_MAX_PER, E_MAX_PER, cap_now, np.inf, S_MAX])
        x, _, z = solve_eq_lp(obj, rows, cols, vals, b, lo, hi)
        return StepResult(c=float(x[0]), d=float(x[1]), u=float(x[2]),
                          w=float(x[3]), obj=z)

    # 变量布局：x = [c, d, u, w, S_{t+1}]（here-and-now 共享）
    #          + N_s 个未来块 [c^s, d^s, u^s, w^s, S^s(m+1)]（wait-and-see）
    n_s = len(scn)
    blk = 5 * m + 1
    n = 5 + n_s * blk
    obj = np.zeros(n)
    obj[2] = kappa * p[t]                    # 当步紧急购电（g_t 已锁定，沉没）
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    hi[0] = hi[1] = E_MAX_PER                # 当步 c / d 功率上限
    hi[2] = cap_now                          # 当步紧急购电上界（真值口径）
    lo[4] = S_MIN                            # 共享 S_{t+1}
    hi[4] = S_MAX
    # 当步平衡 d+u−c−w = n_t−g_t 与 SOC 递推 S_{t+1}−ηc+d/η = S_t
    rows = [np.array([0, 0, 0, 0]), np.array([1, 1, 1])]
    cols = [np.array([1, 2, 3, 0]), np.array([4, 0, 1])]
    vals = [np.array([1.0, 1.0, -1.0, -1.0]),
            np.array([1.0, -ETA, 1.0 / ETA])]
    b = [net[t] - g[t], s_prev]
    tt = np.arange(m)
    for s in range(n_s):
        o_c = 5 + s * blk
        o_d, o_u = o_c + m, o_c + 2 * m
        o_w, o_s = o_c + 3 * m, o_c + 4 * m
        # 情景目标：π_s·κΣ_{τ>t}p^(s)·u^s − π_s·ν·S^s_end；变量域同题2 模板
        pi = float(probs[s])
        obj[o_u:o_u + m] = pi * kappa * (p[t + 1:] if p_scn is None
                                         else p_scn[s, t + 1:])
        obj[o_s + m] = -pi * nu_end
        hi[o_c:o_c + m] = E_MAX_PER
        hi[o_d:o_d + m] = E_MAX_PER
        lo[o_s:o_s + m + 1] = S_MIN
        hi[o_s:o_s + m + 1] = S_MAX
        if u_cap is not None:
            hi[o_u:o_u + m] = u_cap[s, t + 1:]
        # 情景块锚定：S^s[0] = S_{t+1}
        r0 = 2 + s * (2 * m + 1)
        rows.append(np.array([r0, r0]))
        cols.append(np.array([o_s, 4]))
        vals.append(np.array([1.0, -1.0]))
        b.append(0.0)
        # 剩余时域平衡：d^s + u^s − c^s − w^s = n̂^s − g（g 已锁定，移右端）
        r = r0 + 1 + tt
        rows += [r, r, r, r]
        cols += [o_c + tt, o_d + tt, o_u + tt, o_w + tt]
        vals += [-np.ones(m), np.ones(m), np.ones(m), -np.ones(m)]
        b.extend((scn[s, t + 1:] - g[t + 1:]).tolist())
        # SOC 递推：S^s_{τ+1} − S^s_τ − ηc^s + d^s/η = 0
        soc_rows, soc_cols, soc_vals = soc_recursion(
            m, r0 + 1 + m, o_s, o_c, o_d)
        rows += soc_rows
        cols += soc_cols
        vals += soc_vals
        b.extend([0.0] * m)
    x, _, z = solve_eq_lp(obj, rows, cols, vals, np.asarray(b), lo, hi)
    return StepResult(c=float(x[0]), d=float(x[1]), u=float(x[2]),
                      w=float(x[3]), obj=z)
