"""题4 预报层：电价点预报（PR1/PR0）与末端估值 ν^end 的逐日标定参数。

文档 §4.1：周内效应是价格预报技能的全部来源（PR1 同星期几 MAPE 7.51%
vs PR2 混星期 13.65%），故默认 PR1（近 W_p 个同星期几同时段均值）；
PR0（附件1 典型日）为零信息基准，回答"电价到底要不要建模"。不做日内
更新、不做偏差校正、不做谐波回归（§4.1 三条各有实测依据）。

全部只用 F_0^(j)（截至昨日的实现价 + 日历）。次日预报（ν^end 标定用）
取目标日星期几在严格历史内的同窗均值——与目标日相邻的最近同星期几日
至少早 6 天，当日实现值结构性不入窗，无泄漏（题4 档 A22）。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.data.price import YearPrice

NU_END_CONST = 0.472  # V0 常数锚 = 题1 档循环对偶（附件1 价格，文档 §2.3）


@dataclass(frozen=True)
class PriceConfig:
    """题4 价格预报与 ν^end 标定配置（文档 §4.1 / §4.3）。

    Attributes:
        price: 点预报候选，PR1 近 W_p 个同星期几同时段均值（默认）/
            PR0 附件1 典型日（零信息基准）。
        wp: PR1 窗长 W_p（PR1′ = 8 的对照只进预报技能表，不进 Z 层网格）。
        nu_rule: ν^end 标定规则，V2 次日预报价循环 LP 对偶（默认，零超
            参数）/ V1 γ·次日日均价预报（对照臂）/ V0 常数 0.472（对照）。
        gamma: V1 的 γ，默认 0.60 取题4 档 §3 第 18 条实测比值均值。
    """

    price: str = "PR1"
    wp: int = 4
    nu_rule: str = "V2"
    gamma: float = 0.60


def _dow_mean(hist: np.ndarray, dows: np.ndarray, dow_target: int,
              wp: int) -> np.ndarray:
    """近 wp 个同星期几日的同点均值；无同星期几样本退化为全历史均值。"""
    idx = np.flatnonzero(dows == dow_target)[-wp:]
    return hist[idx].mean(axis=0) if len(idx) else hist.mean(axis=0)


def forecast_price(cfg: PriceConfig, price: YearPrice, j: int,
                   typ_p: np.ndarray) -> np.ndarray:
    """第 j 日（0 基）电价点预报 p̂，只用前 j 日实现价与日历。

    Args:
        cfg: 价格预报配置。
        price: 附件4 全年实现电价（本函数只切片 [:j]）。
        j: 0 基日索引。
        typ_p: 附件1 典型日电价（PR0 与无历史先验）。

    Returns:
        点预报 p̂，shape (144,)。

    Raises:
        ValueError: 候选编号不在 {PR1, PR0}。
    """
    # 无样本（day 0）或 PR0：附件1 典型日先验 / 基准
    if cfg.price == "PR0" or j == 0:
        return typ_p.copy()

    # PR1：p̂ = 近 W_p 个同星期几日的同点均值（§3 第 13 条周内近二值）
    if cfg.price == "PR1":
        return _dow_mean(price.p[:j], price.dow[:j], price.dow[j], cfg.wp)
    raise ValueError(f"未知价格候选: {cfg.price}")


def next_price(cfg: PriceConfig, price: YearPrice, j: int,
               typ_p: np.ndarray) -> np.ndarray:
    """第 j 日 0:00 视角的次日价格预报（ν^end 标定输入，文档 §4.3）。

    目标日 = j+1；j 为全年最后一日（0 基 364）时按文档边界约定回退为
    当日 j 自身（滚动时域固有瑕疵）。历史取严格 [:j]（0:00 只掌握截至
    昨日的实现值）——目标日 j+1 的最近同星期几日前驱至少早 6 日，与
    [:j+1] 切片选出的窗口逐日相同，无泄漏且口径与 A18 一致。
    """
    tgt = j + 1 if j + 1 < len(price.p) else j
    if cfg.price == "PR0" or j == 0:
        return typ_p.copy()
    return _dow_mean(price.p[:j], price.dow[:j], price.dow[tgt], cfg.wp)
