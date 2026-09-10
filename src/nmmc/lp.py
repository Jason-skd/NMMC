"""题1 LP 建模（题2-4 复用骨架）。

变量布局 x = [g(144), c(144), d(144), w(144), S(145)]，共 721 列：
    g  计划购电量 kWh（外网关口侧）
    c  充电量 kWh（母线侧）
    d  放电量 kWh（母线侧）
    w  弃电量 kWh（母线侧）
    S  储电量 kWh，S[0]=S_0，S[144]=24:00

等式约束 289 条：144 供需平衡 + 144 SOC 递推 + 1 昼夜循环。
符号与约束见 docs/题1变量定义.md。
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

T = 144
ETA = 0.9                 # 单向充放电效率
P_MAX = 5000.0            # kW，母线侧
E_MAX_PER = P_MAX / 6     # 833.33 kWh，单时段充/放电量上限
S_MIN, S_MAX = 1200.0, 10800.0

NG, NC, ND, NW, NS = 0, T, 2 * T, 3 * T, 4 * T
N = 5 * T + 1


@dataclass
class Solution:
    g: np.ndarray
    c: np.ndarray
    d: np.ndarray
    w: np.ndarray
    S: np.ndarray
    cost: float


def solve(p: np.ndarray, ell: np.ndarray, r: np.ndarray,
          s0_fixed: float | None = None) -> Solution:
    """求解 LP；s0_fixed 给定时把 S_0 钉在该值（经循环约束 S_144 随之锁定）。"""
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

    res = linprog(obj, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP 求解失败: {res.message}")

    x = res.x
    return Solution(
        g=x[NG:NG + T], c=x[NC:NC + T], d=x[ND:ND + T],
        w=x[NW:NW + T], S=x[NS:], cost=float(res.fun),
    )
