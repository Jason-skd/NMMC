"""题3 规划层与内环步进 LP（稀疏 HiGHS）。

三个入口共用题2 档 §5 的物理模板（nmmc.model.q2 的布局约定，g → a）：
    solve_stage  阶段 m≥1 两阶段场景 LP（文档 §7.1）——本块承诺 a(B_m)
                 是 here-and-now 共享变量，更晚块的试探 a^(s) 是
                 wait-and-see（偏差费计入期望）；δ± 自动互斥（性质 5），
                 纯 LP。⚠️ 交付路径的 m=0（0:00 定 g）不走本函数，见
                 nmmc.run.q3.rollout：沿用题 2 两阶段 SAA（solve_plan）。
                 本函数 m=0 分支保留为 wait-and-see 对照（诊断 CF-D）。
    solve_step   内环在线随机 MPC 的单步两阶段 LP（文档 §8，交付）——
                 当步 (c,d,u,w,S_{t+1}) here-and-now 全情景共享，剩余
                 时域按情景分支；分支内供给分层：锁定段注入移右端
                 （费用沉没）、未锁定段 a^(s) 按偏差费 wait-and-see
                 （题2 档 §7.2 solve_stochastic_step 同款 + 分层）。
    solve_tail   确定性等价 MPC 的剩余时域 LP（CE 对照臂，α^rt=0.65
                 分位参考路径）——已锁定段注入为参数，未锁定段 a 为
                 内部变量并按偏差费计（与 solve_stage 的 wait-and-see
                 同一记账）。

变量布局（题2 档约定）：每情景块 [c(T'), d(T'), u(T'), w(T'), S(T'+1)]；
偏差量 δ+/δ− 仅对可调整时段存在。S 初值由 bounds 钉在锚点。

题3 档常数（docs/题3变量定义.md §1 集合与 §2.3 倍率）：修订时刻 τ_m、
块结构 B_m 与结算倍率 β± 定义于此。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.model.lp import (E_MAX_PER, ETA, S_MAX, S_MIN, T, solve_eq_lp,
                           soc_recursion)
from nmmc.model.q2 import StepResult

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
                nu_end: float, kappa: float,
                p_scn: np.ndarray | None = None,
                u_cap: np.ndarray | None = None) -> StageResult:
    """解阶段 m 两阶段场景 LP（文档 §7.1）。

    Args:
        p: 一阶段计价电价（本块承诺 a(B_m) 与共享 δ± 的期望计价按性质 2
            折叠为 p̂ = Σπ_s p^(s)；题4 档 §6.2），元/kWh，shape (144,)。
        g: 0:00 计划购电量（计费基准）；m=0 时为 None（g 是本阶段决策）。
        committed: m≥1 时的已承诺注入前缀（[0, 6τ_m) 段有效）；m=0 为 None。
        scn: 净负荷场景 n^(s)，kWh，shape (N_s, 144)（已实现前缀为真值）。
        probs: 情景概率 π_s。
        m: 修订阶段 0..3。
        s_anchor: 当前真实储电量（SOC 重锚，文档 §5.4）。
        nu_end: 日末储电量估值 ν^end。
        kappa: 紧急购电倍率 κ。
        u_cap: 情景紧急购电上界，shape (N_s, 144)；None 为无上界。
            上界右端取 (n−a)⁺，a 为本阶段决策，由调用方按序列线性化
            给定（题 2 档 §6.1 同款）。⚠️ 不得写成 u+a≤n。
        p_scn: 情景电价 p^(s)，shape (N_s, 144)——更晚块试探 a^(s)、δ±(s)
            与紧急项 u^(s) 均为情景相关量、不可折叠、逐情景计价（性质 2）；
            None 即常数价口径（题3 / 读法 D / 附件1 退化，逐位不变）。

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
    o_dp = n_hn                              # δ 块起点（m=0 不写共享 δ，但偏移仍按 hn 之后算）
    o_s = o_dp + (2 * n_hn if m else 0)      # 首情景块起点；m=0 时 o_s = n_hn

    # 变量布局：x = [hn(B_m)] (+ m≥1: [δ+(B_m), δ−(B_m)]) + N_s 个情景块
    # [af, δ+f, δ−f, c, d, u, w, S(nh+1)]
    # 目标：Z̃ = E_s[Σ p·a^(s) + ½p·(δ+ + δ−) + κp·u^(s)] − ν·E_s[S_end]
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
            p_fut = p[fe:] if p_scn is None else p_scn[s, fe:]
            obj[o_af:o_af + n_free] = pi * p_fut
            obj[o_dpf:o_dpf + n_free] = pi * 0.5 * p_fut
            obj[o_dmf:o_dmf + n_free] = pi * 0.5 * p_fut
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
        # 情景物理边界与目标：π_s·κΣp^(s)·u^(s) − π_s·ν·S_end；
        # 0 ≤ c,d ≤ E_MAX_PER；S ∈ [S_MIN, S_MAX]，S_0 锚定 s_anchor
        obj[o_u:o_u + nh] = pi * kappa * (p[h0:] if p_scn is None
                                          else p_scn[s, h0:])
        obj[o_st + nh] = -pi * nu_end
        hi[o_c:o_c + nh] = E_MAX_PER
        hi[o_d:o_d + nh] = E_MAX_PER
        lo[o_st:o_st + nh + 1] = S_MIN
        hi[o_st:o_st + nh + 1] = S_MAX
        lo[o_st] = hi[o_st] = s_anchor
        if u_cap is not None:
            hi[o_u:o_u + nh] = u_cap[s, h0:]

    # 求解；Z̃ = obj @ x（期望口径，含末端估值，非现金费用）
    x, _, _ = solve_eq_lp(obj, rows, cols, vals, np.array(b), lo, hi)

    # 解包：hn 段即本块承诺（m=0 为全天计划 g），af 段为逐情景试探
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


def solve_step(p: np.ndarray, net: np.ndarray, committed: np.ndarray,
               free_mask: np.ndarray, g: np.ndarray, scn: np.ndarray,
               probs: np.ndarray, t: int, s_prev: float, nu_end: float,
               kappa: float, p_scn: np.ndarray | None = None,
               u_cap: np.ndarray | None = None) -> StepResult:
    """解内环在线随机 MPC 的单步两阶段 LP（文档 §8，交付）。

    第一阶段（here-and-now）：当步 (c,d,u,w) 与 S_{t+1} 全情景共享，
    平衡用真值 n_t、注入恒为锁定参数移右端（内环无权改 a）——非预期性
    的紧凑写法；第二阶段（wait-and-see）：剩余 m = T−1−t 段每情景一块，
    供给分层——已锁定段注入移平衡右端（费用已沉没）、未锁定段 a^(s) 为
    变量并按偏差费计（与 solve_stage 的 wait-and-see 同一记账）。目标
    κp_t·u_t + Σ_s π_s[Σ_free(p·a^(s) + ½p·δ^(s)) + κΣ_{τ>t}p_τ·u^s_τ
    − ν·S^s_end]。m = 0（末时段）无未来分支，−ν·S_{t+1} 显式入目标
    （继承题2 档 C15：否则求解器对"放电后弃掉"无差异，末段白弃电）。

    Args:
        p: 当步计价电价——题4 读法 R 下为当日实现价 p_t^(j)（交易时刻
            可见），元/kWh，shape (144,)。
        net: 当日真实净负荷，kWh，shape (144,)。
        committed: 各时段已承诺注入（free_mask 为 True 处取值被忽略）。
        free_mask: 未锁定时段掩码，shape (144,)。
        g: 0:00 计划购电量（未锁定段的偏差基准）。
        scn: 当前阶段净负荷场景 n^(s)，kWh，shape (N_s, 144)。
        probs: 情景概率 π_s。
        t: 当前时段（0 基）。
        s_prev: 真实储电量 S_t（时段初）。
        nu_end: 时域末端储电量估值 ν^end。
        kappa: 紧急购电倍率 κ。
        u_cap: 情景紧急购电上界，shape (N_s, 144)，None 为无上界。
            当步上界按真值口径取 (n_t − a_t)⁺（a_t = committed[t] 已锁定）；
            剩余时域第 s 块取 u_cap[s, t+1:]。
        p_scn: 未来分支的情景电价 p^(s)（τ > t 段，题4 档 §7）；None 即
            常数价口径（题3，各情景用 p）。

    Returns:
        StepResult: 当步决策与步进目标值。S_{t+1} 由执行层按真实递推
        重建，不经求解器往返。

    Raises:
        RuntimeError: LP 不可行或未达最优（恒可行性继承题2 档 §7.5，
        出现即为实现错误）。
    """
    m = T - 1 - t
    cap_now = np.inf if u_cap is None else max(net[t] - committed[t], 0.0)
    # 末时段：无未来分支，末端估值直接挂共享 S_{t+1}（继承题2 档 C15）
    # 变量布局 x = [c, d, u, w, S_{t+1}]；目标 = κp_t·u_t − ν·S_{t+1}
    if m == 0:
        obj = np.array([0.0, 0.0, kappa * p[t], 0.0, -nu_end])
        # 平衡 d+u−c−w = n_t−a_t（a_t 恒锁定，移右端）与
        # SOC 递推 S_{t+1}−ηc+d/η = S_t
        rows = [np.array([0, 0, 0, 0]), np.array([1, 1, 1])]
        cols = [np.array([1, 2, 3, 0]), np.array([4, 0, 1])]
        vals = [np.array([1.0, 1.0, -1.0, -1.0]),
                np.array([1.0, -ETA, 1.0 / ETA])]
        b = np.array([net[t] - committed[t], s_prev])
        # 变量域：0 ≤ c,d ≤ E_MAX_PER；0 ≤ u ≤ cap；w ≥ 0；S ∈ [S_MIN,S_MAX]
        lo = np.array([0.0, 0.0, 0.0, 0.0, S_MIN])
        hi = np.array([E_MAX_PER, E_MAX_PER, cap_now, np.inf, S_MAX])
        x, _, z = solve_eq_lp(obj, rows, cols, vals, b, lo, hi)
        return StepResult(c=float(x[0]), d=float(x[1]), u=float(x[2]),
                          w=float(x[3]), obj=z)

    # 剩余时域供给分层：自由段（内部变量 af^(s)，相对位置 ff）与锁定段
    n_s = len(scn)
    fut = np.arange(t + 1, T)
    ff = np.flatnonzero(free_mask[fut])
    n_ff = len(ff)
    n_blk = 3 * n_ff + 5 * m + 1
    n = 5 + n_s * n_blk

    # 变量布局：x = [c, d, u, w, S_{t+1}]（here-and-now 共享）
    #          + N_s 个未来块 [af^s, δ+^s, δ−^s, c^s, d^s, u^s, w^s,
    #            S^s(m+1)]（wait-and-see，S^s[0] 锚定共享 S_{t+1}）
    obj = np.zeros(n)
    obj[2] = kappa * p[t]                    # 当步紧急购电（a_t 已锁定，沉没）
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    hi[0] = hi[1] = E_MAX_PER                # 当步 c / d 功率上限
    hi[2] = cap_now                          # 当步紧急购电上界（真值口径）
    lo[4] = S_MIN                            # 共享 S_{t+1}
    hi[4] = S_MAX
    # 当步平衡 d+u−c−w = n_t−a_t 与 SOC 递推 S_{t+1}−ηc+d/η = S_t
    rows = [np.array([0, 0, 0, 0]), np.array([1, 1, 1])]
    cols = [np.array([1, 2, 3, 0]), np.array([4, 0, 1])]
    vals = [np.array([1.0, 1.0, -1.0, -1.0]),
            np.array([1.0, -ETA, 1.0 / ETA])]
    b = [net[t] - committed[t], s_prev]
    tt = np.arange(m)
    row = 2
    for s in range(n_s):
        o_af = 5 + s * n_blk
        o_dpf = o_af + n_ff
        o_dmf = o_dpf + n_ff
        o_c = o_dmf + n_ff
        o_d, o_u = o_c + m, o_c + 2 * m
        o_w, o_st = o_c + 3 * m, o_c + 4 * m
        pi = float(probs[s])
        # 情景目标：π_s[Σ_free p^(s)·(af + ½δ) + κΣ p^(s)·u^s − ν·S^s_end]；
        # 变量域同题2 模板：0 ≤ c^s,d^s ≤ E_MAX_PER；S^s ∈ [S_MIN, S_MAX]
        p_free = (p[fut[ff]] if p_scn is None else p_scn[s, fut[ff]])
        obj[o_af:o_af + n_ff] = pi * p_free
        obj[o_dpf:o_dpf + n_ff] = pi * 0.5 * p_free
        obj[o_dmf:o_dmf + n_ff] = pi * 0.5 * p_free
        obj[o_u:o_u + m] = pi * kappa * (p[t + 1:] if p_scn is None
                                         else p_scn[s, t + 1:])
        obj[o_st + m] = -pi * nu_end
        hi[o_c:o_c + m] = E_MAX_PER
        hi[o_d:o_d + m] = E_MAX_PER
        lo[o_st:o_st + m + 1] = S_MIN
        hi[o_st:o_st + m + 1] = S_MAX
        if u_cap is not None:
            hi[o_u:o_u + m] = u_cap[s, t + 1:]
        # 情景块锚定：S^s[0] = S_{t+1}
        rows.append(np.array([row, row]))
        cols.append(np.array([o_st, 4]))
        vals.append(np.array([1.0, -1.0]))
        b.append(0.0)
        row += 1
        # 剩余时域平衡：d^s + u^s − c^s − w^s = n̂^s − a（锁定段 a =
        # committed 移右端，自由段 af^s 进平衡式）
        r = row + tt
        rows += [r, r, r, r]
        cols += [o_c + tt, o_d + tt, o_u + tt, o_w + tt]
        vals += [-np.ones(m), np.ones(m), np.ones(m), -np.ones(m)]
        rhs = np.where(free_mask[fut], scn[s, fut],
                       scn[s, fut] - committed[fut])
        b.extend(rhs.tolist())
        if n_ff:
            rows.append(r[ff])
            cols.append(o_af + np.arange(n_ff))
            vals.append(np.ones(n_ff))
        row += m
        # SOC 递推：S^s_{τ+1} − S^s_τ − ηc^s + d^s/η = 0
        soc_rows, soc_cols, soc_vals = soc_recursion(m, row, o_st, o_c, o_d)
        rows += soc_rows
        cols += soc_cols
        vals += soc_vals
        b.extend([0.0] * m)
        row += m
        # 自由段单次比对偏差：af^s − δ+^s + δ−^s = g（与规划层同记账）
        if n_ff:
            rf = row + np.arange(n_ff)
            rows += [rf, rf, rf]
            cols += [o_af + np.arange(n_ff), o_dpf + np.arange(n_ff),
                     o_dmf + np.arange(n_ff)]
            vals += [np.ones(n_ff), -np.ones(n_ff), np.ones(n_ff)]
            b.extend(g[fut[ff]].tolist())
            row += n_ff
    x, _, z = solve_eq_lp(obj, rows, cols, vals, np.asarray(b), lo, hi)
    return StepResult(c=float(x[0]), d=float(x[1]), u=float(x[2]),
                      w=float(x[3]), obj=z)


def solve_tail(p: np.ndarray, net: np.ndarray, committed: np.ndarray,
               free_mask: np.ndarray, g: np.ndarray, s_start: float,
               nu_end: float, kappa: float,
               u_cap: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """解确定性等价 MPC 的剩余时域 LP（CE 对照臂，文档 §12.3 执行层增益）。

    已锁定段注入为参数（移平衡式右端）；未锁定段 a 为变量，按
    p·a + ½p|a−g| 计费（与规划层 wait-and-see 同一记账）。

    Args:
        p: 电价，与 net 等长的尾部切片。
        net: 净负荷路径（首时段真值 + α^rt 分位参考路径）。
        committed: 各时段注入参数（free_mask 为 True 处取值被忽略）。
        free_mask: 未锁定时段掩码，与 net 等长。
        g: 计划购电量切片（未锁定段的偏差基准）。
        s_start: 锚定储电量。
        nu_end / kappa: 末端估值 / 紧急倍率。
        u_cap: 紧急购电上界（与 net 等长），None 为无上界。

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

    # 变量布局：x = [c, d, u, w, S(n+1)] + 未锁定段 [af, δ+, δ−]
    # 目标：min κΣp·u + Σ_free[p·af + ½p·(δ+ + δ−)] − ν·S_end
    obj = np.zeros(n)
    obj[o_u:o_u + n_t] = kappa * p
    obj[o_st + n_t] = -nu_end

    # 变量域：0 ≤ c,d ≤ E_MAX_PER；u,w ≥ 0；S ∈ [S_MIN, S_MAX]，S_0 锚定 s_start
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    hi[o_c:o_c + n_t] = E_MAX_PER
    hi[o_d:o_d + n_t] = E_MAX_PER
    lo[o_st:o_st + n_t + 1] = S_MIN
    hi[o_st:o_st + n_t + 1] = S_MAX
    lo[o_st] = hi[o_st] = s_start
    if u_cap is not None:
        hi[o_u:o_u + n_t] = u_cap

    # 平衡：d + u − c − w = n − a（已锁定段 a = committed 移右端，未锁定段 af 为变量）
    tt = np.arange(n_t)
    rows = [tt, tt, tt, tt]
    cols = [o_c + tt, o_d + tt, o_u + tt, o_w + tt]
    vals = [-np.ones(n_t), np.ones(n_t), np.ones(n_t), -np.ones(n_t)]

    # SOC 递推：S_t − S_{t−1} − ηc + d/η = 0
    soc_rows, soc_cols, soc_vals = soc_recursion(n_t, n_t, o_st, o_c, o_d)
    rows += soc_rows
    cols += soc_cols
    vals += soc_vals
    b = list(np.where(free_mask, net, net - committed)) + [0.0] * n_t
    if n_f:
        # 未锁定段注入 af 进平衡式（列补挂到平衡行）
        rows.append(tt[free])
        cols.append(o_af + np.arange(n_f))
        vals.append(np.ones(n_f))
        # 单次比对偏差：af − δ+ + δ− = g（新行，排在 SOC 递推之后）
        rf = 2 * n_t + np.arange(n_f)
        rows += [rf, rf, rf]
        cols += [o_af + np.arange(n_f), o_af + n_f + np.arange(n_f),
                 o_af + 2 * n_f + np.arange(n_f)]
        vals += [np.ones(n_f), -np.ones(n_f), np.ones(n_f)]
        b += list(g[free])
        # 未锁定段计费：p·af + ½p·(δ+ + δ−)（与规划层 wait-and-see 同记账）
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
