"""题4 数据层：附件4 全年实现电价加载。

读法 R（文档 §2.3 / A18）：附件4 与附件2 同为"按日解锁的真实世界"，
0:00 不可知当日价格。诚实回测纪律在调用侧执行：预报只用 `price.p[:j]`，
执行/结算层当步可见 `p[:t+1]`，全年真值仅供结算与事后诊断。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data import ATTACH, T

DAYS = 365  # 全年日数（附件4 数据行数）
IDENTITY_TOL = 6e-5  # §3 第 1 条恒等式的容差，元/kWh（C4-6）


@dataclass(frozen=True)
class YearPrice:
    """附件4 全年实现电价。

    Attributes:
        dates: 日期，shape (365,)。
        p: 实现购电单价，元/kWh，shape (365, 144)。
        dow: 星期几（0=周一 … 6=周日），shape (365,)——价格预报的
            周内节律索引（§3 第 13 条，与负载同源）。
    """

    dates: np.ndarray
    p: np.ndarray
    dow: np.ndarray


def load_price(p_typ: np.ndarray, path: Path | None = None) -> YearPrice:
    """加载附件4 全年实现电价并断言 §3 第 1 条恒等式。

    恒等式（C4-6）：附件1 电价 = 附件4 全年 365 日的逐时段算术平均
    （max 偏差 5.15e-5，四舍五入量级）。它是"题 2/3 用无条件期望、
    题 4 用条件期望 + 残差情景"这一定位的输入事实依据，加载即核验。

    Args:
        p_typ: 附件1 典型日电价，shape (144,)。
        path: 附件4 xlsx 路径；缺省用 docs/附件/附件4.xlsx。

    Returns:
        YearPrice: 全年实现电价与星期几日历。

    Raises:
        ValueError: 行数、时间列数、日期序列、数值域或 §3 第 1 条
            恒等式不符合预期。
    """
    df = pd.read_excel(path or ATTACH / "附件4.xlsx")
    if df.shape != (DAYS, T + 1):
        raise ValueError(f"附件4 行数或时间列数不符合预期: {df.shape}")
    dates = df.iloc[:, 0].to_numpy(dtype="datetime64[D]")
    idx = pd.DatetimeIndex(dates)
    if not (idx.is_monotonic_increasing and idx[0] == pd.Timestamp("2025-01-01")):
        raise ValueError("附件4 日期序列不符合预期")
    p = df.iloc[:, 1:].to_numpy(float)
    if np.isnan(p).any() or (p <= 0.0).any():
        raise ValueError("附件4 存在缺失值或非正值")
    dev = float(np.abs(p.mean(axis=0) - p_typ).max())
    if dev > IDENTITY_TOL:
        raise ValueError(f"附件4 与附件1 电价恒等式偏差超限: {dev:.2e} > "
                         f"{IDENTITY_TOL:.0e}")
    return YearPrice(dates=dates, p=p, dow=idx.dayofweek.to_numpy())
