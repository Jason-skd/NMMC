"""题1 LP 建模。

变量布局 x = [g(144), c(144), d(144), w(144), S(145)]，共 721 列：
    g  计划购电量 kWh（外网关口侧）
    c  充电量 kWh（母线侧）
    d  放电量 kWh（母线侧）
    w  弃电量 kWh（母线侧）
    S  储电量 kWh，S[0]=S_0，S[144]=24:00

等式约束 289 条：144 供需平衡 + 144 SOC 递推 + 1 昼夜循环。
对偶变量：lam_t = 平衡约束影子价格（母线边际价值，元/kWh）；
         nu_t  = SOC 递推影子价格的相反数（储能边际价值，正号约定，见文档 §7）。
符号与约束见 docs/题1变量定义.md。
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from nmmc.model.lp import E_MAX_PER, ETA, S_MAX, S_MIN, T

NG, NC, ND, NW, NS = 0, T, 2 * T, 3 * T, 4 * T
N = 5 * T + 1


@dataclass
class Solution:
    """LP 最优解。

    Attributes:
        g: 计划购电量，kWh，shape (144,)。
        c: 充电量（母线侧），kWh，shape (144,)。
        d: 放电量（母线侧），kWh，shape (144,)。
        w: 弃电量，kWh，shape (144,)。
        S: 储电量，kWh，shape (145,)；S[0]=S_0，S[144]=24:00。
        cost: 目标函数最优值（全天购电费），元。
        lam: 平衡约束对偶 λ_t（母线边际价值），元/kWh；不可得时为 None。
        nu: 储能边际价值 ν_t（原始对偶取负的正号约定，见文档 §7），元/kWh；
            不可得时为 None。
    """

    g: np.ndarray
    c: np.ndarray
    d: np.ndarray
    w: np.ndarray
    S: np.ndarray
    cost: float
    lam: np.ndarray | None = None
    nu: np.ndarray | None = None


def solve(p: np.ndarray, ell: np.ndarray, r: np.ndarray,
          s0_fixed: float | None = None,
          rule_thresholds: tuple[float, float] | None = None) -> Solution:
    """求解题 1 LP（HiGHS）。

    Args:
        p: 逐时段电价，元/kWh，shape (144,)。
        ell: 逐时段负载需电量，kWh，shape (144,)。
        r: 逐时段光伏发电量，kWh，shape (144,)。
        s0_fixed: 给定时把 S_0 钉在该值（经循环约束 S_144 随之锁定），
            用于 Z^6000 对照与 S_0 平坦性探测；缺省 None 表示 S_0 自由决策。
        rule_thresholds: (low, high) 阈值规则基准——仅当 p_t ≤ low 或光伏
            盈余（r_t > ell_t）时允许充电，仅当 p_t ≥ high 时允许放电，
            许可范围内仍求解 LP；缺省 None 表示不做规则限制。

    Returns:
        Solution: 最优解，含目标值 cost 与约束对偶 lam（λ_t）、nu（ν_t）。

    Raises:
        RuntimeError: LP 求解失败（不可行或未达最优）。
    """
    t = np.arange(T)
    A = np.zeros((2 * T + 1, N))
    b = np.zeros(2 * T + 1)

    # 供需平衡：g + d - c - w = ell - r
    A[t, NG + t] = 1.0
    A[t, ND + t] = 1.0
    A[t, NC + t] = -1.0
    A[t, NW + t] = -1.0
    b[t] = ell - r

    # SOC 递推：S_t - S_{t-1} - eta*c + d/eta = 0
    row = T + t
    A[row, NS + 1 + t] = 1.0
    A[row, NS + t] = -1.0
    A[row, NC + t] = -ETA
    A[row, ND + t] = 1.0 / ETA

    # 昼夜循环：S_144 - S_0 = 0
    A[2 * T, NS + T] = 1.0
    A[2 * T, NS] = -1.0

    obj = np.zeros(N)
    obj[NG:NG + T] = p

    bounds = (
        [(0, None)] * T               # g
        + [(0, E_MAX_PER)] * T        # c
        + [(0, E_MAX_PER)] * T        # d
        + [(0, None)] * T             # w
        + [(S_MIN, S_MAX)] * (T + 1)  # S
    )
    if s0_fixed is not None:
        bounds[NS] = (s0_fixed, s0_fixed)
    if rule_thresholds is not None:
        low, high = rule_thresholds
        for t_ in range(T):
            if p[t_] > low and r[t_] <= ell[t_]:
                bounds[NC + t_] = (0.0, 0.0)   # 该时段禁止充电
            if p[t_] < high:
                bounds[ND + t_] = (0.0, 0.0)   # 该时段禁止放电

    res = linprog(obj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP 求解失败: {res.message}")

    x = res.x
    duals = res.eqlin.marginals
    return Solution(
        g=x[NG:NG + T], c=x[NC:NC + T], d=x[ND:ND + T],
        w=x[NW:NW + T], S=x[NS:], cost=float(res.fun),
        lam=duals[:T].copy(),
        nu=-duals[T:2 * T],
    )
