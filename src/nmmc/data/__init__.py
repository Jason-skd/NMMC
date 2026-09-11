"""数据层：附件定位与时间栅格。功率（kW）一律在此预折算为电量（kWh），模型内不再出现 Δ。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATTACH = ROOT / "docs" / "附件"

DELTA = 1 / 6  # 单时段时长 h
