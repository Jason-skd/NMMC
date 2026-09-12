"""导出层公共件：markdown 表渲染、时段标签、论文时段槽位与双表填写器。

充放电量表与紧急购电量表在 result2 / result3 / result4-2 / result4-3
四份交付中结构完全同构（题4 档 §3 第 22 条：writer 可直接复用题 2/3），
统一收口于此；费用列口径各题不同，留在各自的导出器。
"""
import datetime as dt

import numpy as np

from nmmc.data.year import YearData
from nmmc.model.lp import ETA, T

PAPER_SLOTS = [6 * h + 1 for h in (10, 12, 14, 16, 18, 20)]  # t = 61..121
BLOCK_LABELS = [f"{4 * k}:00-{4 * (k + 1)}:00" for k in range(6)]
EPS_EMERG = 1e-6      # 紧急购电报表阈值，kWh


def _hm(minutes: int) -> str:
    """分钟数转 HH:MM。"""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def interval_label(t: int) -> str:
    """返回时段 t 的物理区间标签。

    Args:
        t: 时段编号，1..144（标签为时段末点，t=1 即 [00:00, 00:10)）。

    Returns:
        区间标签字符串，如 "00:00-00:10"。
    """
    return f"{_hm((t - 1) * 10)}-{_hm(t * 10)}"


def segment_label(i_a: int, i_b: int) -> str:
    """连续段 [i_a, i_b]（0 基）的填报标签 [i_a·Δ, (i_b+1)·Δ]，如 13:00-13:30。"""
    return f"{_hm(i_a * 10)}-{_hm((i_b + 1) * 10)}"


def md_table(headers: list[str], rows: list[tuple]) -> str:
    """把表头与数据行渲染为 markdown 表格文本。"""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def py_date(j: int, year: YearData) -> dt.date:
    """0 基日索引转 Python 日期。"""
    return year.dates[j].astype(object)


def emergency_segments(u: np.ndarray,
                       eps: float = EPS_EMERG) -> list[tuple[int, int, float]]:
    """紧急购电极大连续段：u_t > ε 的 [i_a, i_b]（0 基含端点）与段内合计。

    Args:
        u: 真实路径紧急购电量，shape (144,)。
        eps: 报表阈值，kWh。

    Returns:
        [(i_a, i_b, Σu)] 列表；无断供日为空。
    """
    on = u > eps
    out = []
    i = 0
    while i < len(u):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(u) and on[j + 1]:
            j += 1
        out.append((i, j, float(u[i:j + 1].sum())))
        i = j + 1
    return out


def write_battery_sheet(ws, days: list, year: YearData) -> None:
    """充放电量表（电池侧 ηc、d/η，6 个 4h 块 + 0:00/24:00 储电量）。"""
    ws.delete_rows(2, ws.max_row - 1)
    for day in days:
        row = ws.max_row + 1
        for k in range(6):
            seg = slice(24 * k, 24 * (k + 1))
            ws.cell(row=row + k, column=2, value=BLOCK_LABELS[k])
            ws.cell(row=row + k, column=3,
                    value=float(ETA * day.c[seg].sum()))
            ws.cell(row=row + k, column=4,
                    value=float(day.d[seg].sum() / ETA))
        ws.cell(row=row, column=1, value=py_date(day.j, year))
        ws.cell(row=row, column=5, value=dt.time(0, 0))
        ws.cell(row=row, column=6, value=float(day.S[0]))
        ws.cell(row=row + 1, column=5, value="24:00")
        ws.cell(row=row + 1, column=6, value=float(day.S[T]))


def write_emergency_sheet(ws, days: list, year: YearData) -> None:
    """紧急购电量表（u > ε 的极大连续段合并，稀疏填报）。"""
    ws.delete_rows(2, ws.max_row - 1)
    for day in days:
        segs = emergency_segments(day.u)
        row = ws.max_row + 1
        for k, (i_a, i_b, e) in enumerate(segs):
            ws.cell(row=row + k, column=2,
                    value=segment_label(i_a, i_b))
            ws.cell(row=row + k, column=3, value=e)
        if segs:
            ws.cell(row=row, column=1, value=py_date(day.j, year))
