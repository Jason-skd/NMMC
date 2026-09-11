"""题2 规划层与补救 LP（稀疏 HiGHS）。

两个入口共用物理模板 X(g, S_0, n)（文档 §5）：
    solve_planning  日前两阶段场景 LP（SAA）——g 为 here-and-now 共享变量，
                    情景块 (c,d,u,w,S) 独立；计划购电费不进情景平均。
    solve_recourse  单路径补救 LP——g 为参数（平衡式右端移项），
                    事后最优（全天）与 MPC 剩余时域（尾部切片）共用。

变量布局（每情景块）：[c(T'), d(T'), u(T'), w(T'), S(T'+1)]。
等式约束（每块）：T' 供需平衡 g+d+u−c−w = n 与 T' SOC 递推
S_t − S_{t−1} − ηc_t + d_t/η = 0。S_0 由 bounds 钉在给定初值。
物理常数沿用题1档（nmmc.model.lp）。
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from nmmc.model.lp import ETA, E_MAX_PER, S_MAX, S_MIN, T

KAPPA = 5.0    # 紧急购电倍率
NU_END = 0.472  # 日末储电量估值锚，元/kWh（文档 §2.3）


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


def _solve(obj: np.ndarray, rows: list[np.ndarray], cols: list[np.ndarray],
           vals: list[np.ndarray], b: np.ndarray, lo: np.ndarray,
           hi: np.ndarray) -> np.ndarray:
    """组装稀疏等式约束 LP 并求解 HiGHS，返回解向量。"""
    a = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(b), len(obj))).tocsr()
    res = linprog(obj, A_eq=a, b_eq=b,
                  bounds=np.column_stack([lo, hi]), method="highs")
    if not res.success:
        raise RuntimeError(f"LP 求解失败: {res.message}")
    return res.x


def solve_planning(p: np.ndarray, scn: np.ndarray, probs: np.ndarray,
                   s0: float, nu_end: float,
                   kappa: float) -> PlanResult:
    """解日前两阶段场景 LP（文档 §6）。

    Args:
        p: 逐时段电价，元/kWh，shape (144,)。
        scn: 净负荷场景 n^(s)，kWh，shape (N_s, 144)。
        probs: 情景概率 π_s（等权 1/N_s）。
        s0: 当日 0:00 储电量（给定状态）。
        nu_end: 日末储电量估值 ν^end。
        kappa: 紧急购电倍率 κ。

    Returns:
        PlanResult: 计划购电量 g 与规划目标值 Z̃。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    n_s = len(scn)
    blk = 5 * T + 1
    n = T + n_s * blk
    obj = np.zeros(n)
    obj[:T] = p
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    b = np.empty(2 * T * n_s)
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
        r = s * 2 * T + T + tt
        rows += [r, r, r, r]
        cols += [o_s + 1 + tt, o_s + tt, o_c + tt, o_d + tt]
        vals += [np.ones(T), -np.ones(T), np.full(T, -ETA), np.full(T, 1.0 / ETA)]
        b[r] = 0.0
        pi = probs[s]
        obj[o_u:o_u + T] = pi * kappa * p
        obj[o_s + T] = -pi * nu_end
        lo[o_c:o_c + T] = 0.0
        hi[o_c:o_c + T] = E_MAX_PER
        lo[o_d:o_d + T] = 0.0
        hi[o_d:o_d + T] = E_MAX_PER
        lo[o_s:o_s + T + 1] = S_MIN
        hi[o_s:o_s + T + 1] = S_MAX
        lo[o_s] = hi[o_s] = s0
    x = _solve(obj, rows, cols, vals, b, lo, hi)
    return PlanResult(g=x[:T], z_tilde=float(obj @ x))


def solve_recourse(p: np.ndarray, net: np.ndarray, g: np.ndarray,
                   s_start: float, nu_end: float,
                   kappa: float) -> PathResult:
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

    Returns:
        PathResult: 补救量、储电量轨迹与目标值。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    n_t = len(net)
    blk = 5 * n_t + 1
    o_c, o_d, o_u, o_w, o_s = 0, n_t, 2 * n_t, 3 * n_t, 4 * n_t
    obj = np.zeros(blk)
    obj[o_u:o_u + n_t] = kappa * p
    obj[o_s + n_t] = -nu_end
    lo = np.zeros(blk)
    hi = np.full(blk, np.inf)
    hi[o_c:o_c + n_t] = E_MAX_PER
    hi[o_d:o_d + n_t] = E_MAX_PER
    lo[o_s:o_s + n_t + 1] = S_MIN
    hi[o_s:o_s + n_t + 1] = S_MAX
    lo[o_s] = hi[o_s] = s_start
    tt = np.arange(n_t)
    rows = [tt, tt, tt, tt] + [n_t + tt] * 4
    cols = [o_c + tt, o_d + tt, o_u + tt, o_w + tt,
            o_s + 1 + tt, o_s + tt, o_c + tt, o_d + tt]
    vals = [-np.ones(n_t), np.ones(n_t), np.ones(n_t), -np.ones(n_t),
            np.ones(n_t), -np.ones(n_t), np.full(n_t, -ETA),
            np.full(n_t, 1.0 / ETA)]
    b = np.concatenate([net - g, np.zeros(n_t)])
    x = _solve(obj, rows, cols, vals, b, lo, hi)
    return PathResult(
        c=x[o_c:o_c + n_t], d=x[o_d:o_d + n_t],
        u=x[o_u:o_u + n_t], w=x[o_w:o_w + n_t],
        S=x[o_s:o_s + n_t + 1], obj=float(obj @ x))
