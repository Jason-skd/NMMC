"""可复用 LP 骨架：微网物理常数、SOC 递推块与稀疏 LP 求解封装（题1-4 共用档）。

时间栅格 T 取自数据层（nmmc.data，与附件栅格同源）；其余为储能物理
常数。各问模型只从此处取常数与求解件，不再各自定义。
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from nmmc.data import T

ETA = 0.9                 # 单向充放电效率
P_MAX = 5000.0            # kW，母线侧
E_MAX_PER = P_MAX / 6     # 833.33 kWh，单时段充/放电量上限
S_MIN, S_MAX = 1200.0, 10800.0


def soc_recursion(n_t: int, row0: int, o_s: int, o_c: int,
                  o_d: int) -> tuple[list, list, list]:
    """SOC 递推块 S_t − S_{t−1} − η·c_t + d_t/η = 0 的 COO 三元组。

    块内第 k 行（0 ≤ k < n_t）对应递推式 S_{k+1} = S_k + η·c_k − d_k/η。

    Args:
        n_t: 块内时段数。
        row0: 块首行号（约束矩阵行偏移）。
        o_s / o_c / o_d: S / c / d 序列的起始列号（S 序列长 n_t+1，
            S[0] 为锚，由 bounds 钉值或入循环约束）。

    Returns:
        (rows, cols, vals)：各为 4 个数组的列表，供 solve_eq_lp 拼接。
    """
    tt = np.arange(n_t)
    rows = [row0 + tt] * 4
    cols = [o_s + 1 + tt, o_s + tt, o_c + tt, o_d + tt]
    vals = [np.ones(n_t), -np.ones(n_t), np.full(n_t, -ETA),
            np.full(n_t, 1.0 / ETA)]
    return rows, cols, vals


def solve_eq_lp(obj: np.ndarray, rows: list, cols: list, vals: list,
                b: np.ndarray, lo: np.ndarray,
                hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """组装稀疏等式约束 LP 并求解 HiGHS。

    Args:
        obj: 目标系数，shape (n,)。
        rows / cols / vals: COO 三元组分块列表，拼接即约束矩阵非零元。
        b: 等式右端，shape (m,)。
        lo / hi: 变量上下界，shape (n,)；hi 取 np.inf 表示无上界。

    Returns:
        (x, duals, obj_val): 最优解、等式约束对偶（HiGHS marginals）
        与目标最优值。

    Raises:
        RuntimeError: LP 不可行或未达最优。
    """
    # 组装稀疏等式约束矩阵：COO 三元组拼接 → CSR
    a = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(b), len(obj))).tocsr()

    # HiGHS 求解；对偶取等式约束 marginals（影子价格）
    res = linprog(obj, A_eq=a, b_eq=b,
                  bounds=np.column_stack([lo, hi]), method="highs")
    if not res.success:
        raise RuntimeError(f"LP 求解失败: {res.message}")
    return res.x, res.eqlin.marginals, float(res.fun)
