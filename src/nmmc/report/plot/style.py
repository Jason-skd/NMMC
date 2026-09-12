"""图表共享样式层：配色槽位、统一样式 / 图例 / 保存与横轴助手。

所有图（q1 / q2 / …）从这里取色与样式，保证风格不分裂。

跨图实体配色（固定槽位，颜色跟随实体；已过 dataviz 校验）：
    p=蓝  λ=橙  g=水青  c=黄  d=洋红(#b92fc0)  S=绿  R=紫  L=红  u=深红
例外注记：
    ν（储能边际价值）借水青 dash-dot——与 g 不同图出现，无混淆；
    u（紧急购电）用 serious 深红 #9b2226——与 L 红不同图出现；
    状态矩阵 / 状态带的"闲"态用 GRID 灰作中性底（等同背景，非类别槽）；
    洋红槽自 #e87ba4 重校为 #b92fc0（与 L 红的正常视觉 ΔE ≥ 20，dataviz 六检通过）。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from nmmc.model.lp import T

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
C_PRICE, C_DUAL, C_BUY, C_CH, C_DIS, C_SOC = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b92fc0", "#008300")
C_PV, C_LOAD, C_EMERG = "#4a3aa7", "#e34948", "#9b2226"

HOURS = (np.arange(1, T + 1) - 0.5) / 6.0   # 各时段中心，h
ENDS = np.arange(T + 1) / 6.0               # 时段端点（0..24 h）

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                        "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
})


def style_ax(ax, title: str, ylabel: str) -> None:
    """给子图套用统一样式（面/网格/脊线/刻度/标题/轴标签）。

    Args:
        ax: 目标子图。
        title: 左对齐子图标题。
        ylabel: y 轴标签（含单位）。
    """
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def legend(ax, handles=None, labels=None, *, loc: str = "upper left",
           ncol: int = 1, below: bool = False) -> None:
    """统一图例层：无边框、9pt、文字 INK2。

    位置策略：默认轴内左上（时序图数据多为日间中部峰值，左上最常空）；
    below=True 移到轴下方居中（单幅宽图、条目多时用）；数据占住左上的
    图以 loc 显式越位并在调用处注明原因。

    Args:
        ax: 目标子图。
        handles / labels: 图例句柄与标签；缺省取轴上已绘制的系列
            （handles 可自带 label，此时 labels 传 None）。
        loc: 轴内位置（below=True 时忽略）。
        ncol: 图例列数。
        below: 是否放轴下方。
    """
    kw = dict(frameon=False, fontsize=9, labelcolor=INK2)
    pos = (dict(loc="upper center", bbox_to_anchor=(0.5, -0.18)) if below
           else dict(loc=loc))
    if handles is None:
        ax.legend(ncol=ncol, **pos, **kw)
    elif labels is None:
        ax.legend(handles=handles, ncol=ncol, **pos, **kw)
    else:
        ax.legend(handles, labels, ncol=ncol, **pos, **kw)


def price_bands(ax, p: np.ndarray) -> list[Patch]:
    """按电价三分位划分谷/平/峰时段，叠加浅色背景带（谷=冷、峰=暖）。

    Args:
        ax: 目标子图。
        p: 逐时段电价，shape (144,)。

    Returns:
        [谷段, 峰段] 两个图例 patch 句柄（由调用方并入各自图例）。
    """
    lo, hi = np.quantile(p, (1 / 3, 2 / 3))
    for mask, color in ((p <= lo, C_PRICE), (p >= hi, C_LOAD)):
        for s, n, v in runs(mask.astype(np.int8)):
            if v:
                ax.axvspan(s / 6.0, (s + n) / 6.0, color=color, alpha=0.09,
                           lw=0)
    return [Patch(facecolor=C_PRICE, alpha=0.35, lw=0),
            Patch(facecolor=C_LOAD, alpha=0.35, lw=0)]


BAND_LABELS = ["谷段（电价低 1/3）", "峰段（电价高 1/3）"]


def runs(code: np.ndarray) -> list[tuple[int, int, int]]:
    """等值游程分解：[(起点, 长度, 值)]，供状态带 / 背景带横条用。

    Args:
        code: 一维整型编码数组。

    Returns:
        游程列表；空数组返回空列表。
    """
    if not len(code):
        return []
    brk = np.flatnonzero(np.diff(code)) + 1
    starts = np.r_[0, brk]
    ends = np.r_[brk, len(code)]
    return [(int(s), int(e - s), int(code[s])) for s, e in zip(starts, ends)]


def day_axis(ax) -> None:
    """单日横轴：0–24h 双时刻度、24.8 截断与轴标签。"""
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24.8)
    ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)


def month_axis(ax, dates: np.ndarray, n: int, *, on_y: bool = False) -> None:
    """全年横轴（或纵轴）：月首日刻度 + 月份标签。

    Args:
        ax: 目标子图。
        dates: datetime64 日数组。
        n: 总天数（定轴范围）。
        on_y: True 时刻度打在 y 轴（状态矩阵的"行 = 日"方向）。
    """
    months = dates.astype("datetime64[M]")
    idx = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    labels = [f"{int(str(m)[5:7])}月" for m in months[idx]]
    if on_y:
        ax.set_yticks(idx)
        ax.set_yticklabels(labels)
        ax.set_ylim(n, 0)
    else:
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
        ax.set_xlim(0, n)


def save(fig, out_dir: Path, name: str, *, tight: bool = False) -> Path:
    """落盘 PNG、关闭画布并返回路径（统一 facecolor 与留白策略）。

    Args:
        fig: 已完成布局的画布。
        out_dir: 输出目录（自动创建）。
        name: 文件名（如 q1_schedule.png）。
        tight: 轴下方图例等溢出元素需要 bbox_inches="tight" 时置 True。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if tight:
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.15)
    else:
        fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
