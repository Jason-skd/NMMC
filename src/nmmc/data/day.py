"""附件1 典型日数据加载。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data import ATTACH, DELTA


@dataclass(frozen=True)
class DayData:
    """附件1 典型日数据（kW 已预折算为 kWh）。

    Attributes:
        p: 逐时段电价，元/kWh，shape (144,)。
        ell: 逐时段负载需电量，kWh，shape (144,)。
        r: 逐时段光伏发电量，kWh，shape (144,)。
        labels: 附件1 行标签（时段末点），仅用于展示。
    """

    p: np.ndarray
    ell: np.ndarray
    r: np.ndarray
    labels: np.ndarray


def load_day1(path: Path | None = None) -> DayData:
    """加载附件1 典型日数据并完成 kW → kWh 预折算。

    Args:
        path: 附件1 xlsx 路径；缺省用 docs/附件/附件1.xlsx。

    Returns:
        DayData: 折算后的当日数据。

    Raises:
        ValueError: 列名或行数不符合预期。
    """
    df = pd.read_excel(path or ATTACH / "附件1.xlsx")
    required = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    if list(df.columns) != required or len(df) != 144:
        raise ValueError("附件1 的列名或行数不符合预期")
    return DayData(
        p=df["电价"].to_numpy(float),
        ell=df["小区负载"].to_numpy(float) * DELTA,
        r=df["光伏发电预测功率"].to_numpy(float) * DELTA,
        labels=df["时间"].astype(str).to_numpy(),
    )
