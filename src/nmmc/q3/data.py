"""题3 数据层：附件3 官方光伏预报加载与修订阶段索引。

文档引用 docs/题3变量定义.md（v3）：§1 集合（m/τ_m/B_m）、§2.2 附件3
结构与时间对齐、§2.3 倍率 β±。日索引 0 基（day 0 = 2025-01-01 = 文档 j=1），
与 q2.data 一致；系统常数（κ、ν^end、暖机期等）直接沿用 q2.data。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data import ATTACH

BETA_MINUS = 0.5        # 违约倍率 β⁻（计划高于调整的部分，文档 §2.3）
BETA_PLUS = 1.5         # 加价倍率 β⁺（调整高于计划的部分）
VINTAGE_HOURS = (0, 6, 12, 18)   # 修订时刻 τ_m（小时），m = 0..3
BLOCK_LEN = 36                    # 修订块 B_m 的时段数（6 小时 × 6 时段/h）
BLOCK_STARTS = (0, 36, 72, 108)   # B_m 的 0 基起始时段 = 6·τ_m


@dataclass(frozen=True)
class YearForecast:
    """附件3 官方光伏预报（kW，按题面原单位保存，调用侧再折算）。

    Attributes:
        phi: 整点功率预报，shape (365, 4, 24)。phi[j, m, k] 为日 j、发布
            时刻 τ_m、提前 k+1 小时（目标整点 τ_m+k+1）的预报。跨日部分
            （τ_m + h > 24）弃用（文档 §2.2，数据核验第 3 条）。
    """

    phi: np.ndarray


def load_forecast(path: Path | None = None) -> YearForecast:
    """加载附件3 官方光伏预报。

    Args:
        path: 附件3 xlsx 路径；缺省用 docs/附件/附件3.xlsx。

    Returns:
        YearForecast: 整理为 (日, 发布时刻, 提前小时) 三维数组的预报。

    Raises:
        ValueError: 行数、列名、日期或发布时刻序列不符合预期。
    """
    df = pd.read_excel(path or ATTACH / "附件3.xlsx")
    if len(df) != 365 * 4 or df.shape[1] != 2 + 24:
        raise ValueError(f"附件3 行数或列数不符合预期: {df.shape}")
    if (list(df.columns[:2]) != ["日期", "预报时刻"]
            or list(df.columns[2:]) != [f"预报{h}小时" for h in range(1, 25)]):
        raise ValueError("附件3 列名不符合预期")
    dates = (pd.to_datetime(df["日期"].ffill())
             .to_numpy(dtype="datetime64[D]"))
    uniq = pd.DatetimeIndex(np.unique(dates))
    if not (uniq.is_monotonic_increasing and len(uniq) == 365
            and uniq[0] == pd.Timestamp("2025-01-01")):
        raise ValueError("附件3 日期序列不符合预期")
    hours = df["预报时刻"].astype(str).str.replace(":00", "", regex=False)
    if set(hours) != {"0", "6", "12", "18"}:
        raise ValueError("附件3 发布时刻不符合预期")
    phi = df.iloc[:, 2:].to_numpy(float).reshape(365, 4, 24)
    if (phi < 0).any() or np.isnan(phi).any():
        raise ValueError("附件3 存在负值或缺失值")
    return YearForecast(phi=phi)
