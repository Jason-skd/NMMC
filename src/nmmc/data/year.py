"""题2 数据层：附件2 全年实际值加载与日历，kW → kWh 预折算。

诚实回测的信息纪律在调用侧执行：预报只用 `year.ell[:j]` / `year.r[:j]`
（截至昨日的实际值）与日历列。代码内日索引一律 0 基（day 0 = 2025-01-01，
即文档 §1 的 j=1）。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data import ATTACH, DELTA, T

DAYS = 365  # 全年日数（附件2 数据行数）


@dataclass(frozen=True)
class YearData:
    """附件2 全年实际值（kW 已预折算为 kWh）。

    Attributes:
        dates: 日期，shape (365,)。
        ell: 负载需电量，shape (365, 144)。
        r: 光伏发电量，shape (365, 144)。
        dow: 星期几（0=周一 … 6=周日），shape (365,)。
        doy: 年内日序 1..365，shape (365,)。
    """

    dates: np.ndarray
    ell: np.ndarray
    r: np.ndarray
    dow: np.ndarray
    doy: np.ndarray


def load_year(path: Path | None = None) -> YearData:
    """加载附件2 全年实际值并完成 kW → kWh 预折算。

    Args:
        path: 附件2 xlsx 路径；缺省用 docs/附件/附件2.xlsx。

    Returns:
        YearData: 折算后的全年数据与日历。

    Raises:
        ValueError: 行数、时间列数或日期序列不符合预期。
    """
    path = path or ATTACH / "附件2.xlsx"
    ell_df = pd.read_excel(path, sheet_name="小区负载")
    r_df = pd.read_excel(path, sheet_name="光伏发电实际功率")
    for df, name in ((ell_df, "小区负载"), (r_df, "光伏发电实际功率")):
        if df.shape != (DAYS, T + 1):
            raise ValueError(f"附件2「{name}」行数或时间列数不符合预期: {df.shape}")
    dates = ell_df.iloc[:, 0].to_numpy(dtype="datetime64[D]")
    if not np.array_equal(dates, r_df.iloc[:, 0].to_numpy(dtype="datetime64[D]")):
        raise ValueError("附件2 两表日期不一致")
    idx = pd.DatetimeIndex(dates)
    if not (idx.is_monotonic_increasing and idx[0] == pd.Timestamp("2025-01-01")):
        raise ValueError("附件2 日期序列不符合预期")
    return YearData(
        dates=dates,
        ell=ell_df.iloc[:, 1:].to_numpy(float) * DELTA,
        r=r_df.iloc[:, 1:].to_numpy(float) * DELTA,
        dow=idx.dayofweek.to_numpy(),
        doy=idx.dayofyear.to_numpy(float),
    )
