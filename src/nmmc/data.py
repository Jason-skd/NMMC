"""数据加载与预处理。功率（kW）一律在此预折算为电量（kWh），模型内不再出现 Δ。"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ATTACH = ROOT / "docs" / "附件"

DELTA = 1 / 6  # 单时段时长 h


@dataclass(frozen=True)
class DayData:
    p: np.ndarray       # 电价 元/kWh，(144,)
    ell: np.ndarray     # 负载需电量 kWh
    r: np.ndarray       # 光伏发电量 kWh
    labels: np.ndarray  # 附件1 行标签（时段末点），仅用于展示


def load_day1(path: Path | None = None) -> DayData:
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
