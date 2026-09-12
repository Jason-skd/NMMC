"""导出层公共件：markdown 表渲染、时段标签与论文时段槽位。"""
PAPER_SLOTS = [6 * h + 1 for h in (10, 12, 14, 16, 18, 20)]  # t = 61..121


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
