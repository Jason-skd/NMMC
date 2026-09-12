"""题4 规划层公共件：逐日末端估值 ν^end 标定与一阶段折叠价。

文档 §4.3：题2/3 的常数 ν^end = 0.472 在波动价下会在便宜日囤积过度、
昂贵日抽干过度，改为逐日标定。三臂并列（消融定夺，不得只写默认）：
    V2（默认，零超参数）  以次日预报价曲线 p̂^(j+1) 配附件1 典型日负载/光伏
                          解题1 型昼夜循环 LP，取循环约束对偶的绝对值。
                          全链路不含任何拟合常数；标定只用 F_0^(j)（预报
                          仅由历史同星期几构成），无泄漏（A22）。
    V1（对照臂）          γ·次日日均价预报，γ 扫 {0.50..0.70}，默认 0.60
                          取 §3 第 18 条实测比值均值 0.5946。
    V0（对照）            常数 0.472，检验逐日化必要性（亦是读法退化
                          a1 下的强制值，C4-2 复现口径）。

依据（§3 第 18 条）：循环对偶 |ν_j| 与日均价近似成比例（比值均值
0.5946、CV 7%），两臂预期一致；12.31 边界取 p̂^(365) 回退（next_price）。
"""
import numpy as np

from nmmc.data.price import YearPrice
from nmmc.model import q1
from nmmc.model.forecast4 import NU_END_CONST, PriceConfig, next_price


def nu_end_daily(cfgp: PriceConfig, price: YearPrice, typ_ell: np.ndarray,
                 typ_r: np.ndarray, typ_p: np.ndarray, days: int,
                 kappa_check: bool = True) -> np.ndarray:
    """逐日末端估值 ν^end^(j)，shape (days,)。

    V2 为 365（或截断 days）个 144 时段小 LP（亚秒级/个，§15.1 ④）；
    V1/V0 无需求解。首日自检（kappa_check）：附件1 价格下 V2 复现
    ν* = 0.472（§3 第 18 条，循环对偶的绝对值口径）。

    Args:
        cfgp: 价格配置（nu_rule / gamma）。
        price: 附件4 全年实现电价。
        typ_ell / typ_r: 附件1 典型日负载/光伏（循环 LP 的物理侧输入）。
        typ_p: 附件1 典型日电价（PR0 先验与 V2 首日自检基准）。
        days: 标定天数（与滚动天数一致）。
        kappa_check: 是否执行附件1 自检（默认开）。

    Returns:
        ν^end 逐日序列，元/kWh。

    Raises:
        ValueError: nu_rule 不在 {V0, V1, V2}。
    """
    # V2 首日自检：附件1 价格下循环 LP 的循环对偶 |ν*| = 0.472（±1e-3）
    if cfgp.nu_rule == "V2" and kappa_check:
        nu0 = q1.solve(typ_p, typ_ell, typ_r).nu_cycle
        if abs(abs(float(nu0)) - NU_END_CONST) > 1e-3:
            raise ValueError(f"V2 标定基准失准: 附件1 循环对偶 {nu0} ≠ "
                             f"{NU_END_CONST}")
    out = np.empty(days)
    for j in range(days):
        p_next = next_price(cfgp, price, j, typ_p)
        if cfgp.nu_rule == "V2":
            out[j] = abs(q1.solve(p_next, typ_ell, typ_r).nu_cycle or 0.0)
        elif cfgp.nu_rule == "V1":
            out[j] = cfgp.gamma * float(p_next.mean())
        elif cfgp.nu_rule == "V0":
            out[j] = NU_END_CONST
        else:
            raise ValueError(f"未知 ν^end 规则: {cfgp.nu_rule}")
    return out


def folded_price(prices: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """一阶段折叠价 p̂ = Σ_s π_s p^(s)（性质 2 的精确折叠，C4-5 核验口径）。"""
    return probs @ prices
