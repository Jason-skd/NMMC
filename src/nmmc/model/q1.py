"""题1 LP 建模。

变量布局 x = [g(144), c(144), d(144), w(144), S(145)]，共 721 列：
    g  计划购电量 kWh（外网关口侧）
    c  充电量 kWh（母线侧）
    d  放电量 kWh（母线侧）
    w  弃电量 kWh（母线侧）
    S  储电量 kWh，S[0]=S_0，S[144]=24:00

等式约束 289 条：144 供需平衡 + 144 SOC 递推 + 1 昼夜循环。
对偶变量：lam_t = 平衡约束影子价格（母线边际价值，元/kWh）；
         nu_t  = SOC 递推影子价格的相反数（储能边际价值，正号约定，见文档 §7）；
         nu_cycle = 昼夜循环约束 S_144 = S_0 的对偶（池内 1 kWh 跨日结转
         的边际价值，题4 档 §4.3 逐日 ν^end 标定取其绝对值）。
符号与约束见 docs/题1变量定义.md。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.model.lp import (E_MAX_PER, S_MAX, S_MIN, T, solve_eq_lp,
                           soc_recursion)

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
        nu_cycle: 昼夜循环约束对偶（多留 1 kWh 到日末的边际价值），元/kWh；
            附件1 价格下其绝对值 = 0.472（题2/题3 常数锚的来历，题4 档
            §3 第 18 条自检）；不可得时为 None。
    """

    g: np.ndarray
    c: np.ndarray
    d: np.ndarray
    w: np.ndarray
    S: np.ndarray
    cost: float
    lam: np.ndarray | None = None
    nu: np.ndarray | None = None
    nu_cycle: float | None = None


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
    tt = np.arange(T)

    # 供需平衡：g + d - c - w = ell - r
    rows = [tt] * 4
    cols = [NG + tt, ND + tt, NC + tt, NW + tt]
    vals = [np.ones(T), np.ones(T), -np.ones(T), -np.ones(T)]

    # SOC 递推：S_t - S_{t-1} - eta*c + d/eta = 0
    soc_rows, soc_cols, soc_vals = soc_recursion(T, T, NS, NC, ND)
    rows += soc_rows
    cols += soc_cols
    vals += soc_vals

    # 昼夜循环：S_144 - S_0 = 0
    rows += [[2 * T, 2 * T]]
    cols += [[NS + T, NS]]
    vals += [[1.0, -1.0]]

    # 等式右端：b = [ell - r, 0(SOC), 0(循环)]
    b = np.concatenate([ell - r, np.zeros(2 * T)])

    # 目标：min Z = Σ_t p_t·g_t（全天购电费）
    obj = np.zeros(N)
    obj[NG:NG + T] = p

    # 变量域：g ≥ 0；0 ≤ c, d ≤ E_MAX_PER；w ≥ 0；S_MIN ≤ S ≤ S_MAX
    lo = np.zeros(N)
    hi = np.full(N, np.inf)
    hi[NC:NC + T] = E_MAX_PER         # c
    hi[ND:ND + T] = E_MAX_PER         # d
    lo[NS:] = S_MIN
    hi[NS:] = S_MAX                   # S

    # S_0 钉值：S_0 = S_144 = s0_fixed（经循环约束联动锁定）
    if s0_fixed is not None:
        lo[NS] = hi[NS] = s0_fixed

    # 阈值规则基准：仅 p_t ≤ low 或光伏盈余（r_t > ell_t）许充；仅 p_t ≥ high 许放
    if rule_thresholds is not None:
        low, high = rule_thresholds
        for t_ in range(T):
            if p[t_] > low and r[t_] <= ell[t_]:
                hi[NC + t_] = 0.0     # 该时段禁止充电
            if p[t_] < high:
                hi[ND + t_] = 0.0     # 该时段禁止放电

    # 解包与对偶映射：λ_t = 平衡影子价格；ν_t = −SOC 递推影子价格（文档 §7）
    x, duals, cost = solve_eq_lp(obj, rows, cols, vals, b, lo, hi)
    return Solution(
        g=x[NG:NG + T], c=x[NC:NC + T], d=x[ND:ND + T],
        w=x[NW:NW + T], S=x[NS:], cost=cost,
        lam=duals[:T].copy(),
        nu=-duals[T:2 * T],
        nu_cycle=float(duals[2 * T]),
    )
