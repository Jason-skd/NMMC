"""图表共享样式层：配色槽位、统一样式 / 图例 / 保存与横轴助手。

所有图（q1 / q2 / …）从这里取色与样式，保证风格不分裂。

跨图实体配色（固定槽位，颜色跟随实体；已过 dataviz 校验）：
    p=蓝  λ=橙  g=水青  c=黄  d=洋红(#b92fc0)  S=绿  R=紫  L=红  u=深红
例外注记：
    ν（储能边际价值）借水青 dash-dot——与 g 不同图出现，无混淆；
    u（紧急购电）用 serious 深红 #9b2226——与 L 红不同图出现；
    状态矩阵 / 状态带的"闲"态用 GRID 灰作中性底（等同背景，非类别槽）；
    洋红槽自 #e87ba4 重校为 #b92fc0（与 L 红的正常视觉 ΔE ≥ 20，dataviz 六检通过）；
    q3 起引入极性语言：上调/正差 = 红（借 L 槽，L 不同图）、下调/负差 = 蓝
    （借 p 槽，p 线不同图）——红↔蓝 protan ΔE 21.6，可作无缝隙像元编码。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.text import Text

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


# 压在曲线 / 色块上的标注：浅底描边，避免和数据抢像素
HALO = [patheffects.withStroke(linewidth=3.4, foreground=SURFACE)]


def halo(text: Text) -> Text:
    """给 Text 加画布底色描边，叠在数据上仍可读。"""
    text.set_path_effects(HALO)
    return text


def legend(ax, handles=None, labels=None, *, loc: str = "upper left",
           ncol: int = 1, below: bool = False, filled: bool = False) -> None:
    """统一图例层：无边框、9pt、文字 INK2。

    位置策略：默认轴内左上（时序图数据多为日间中部峰值，左上最常空）；
    below=True 移到轴下方居中（旧版 q1/q2 无 constrained layout 时用
    bbox）；论文图优先 loc="outside …"（需 constrained layout）。
    filled=True 给轴内图例铺 SURFACE 底，避免字和曲线咬在一起。

    Args:
        ax: 目标子图。
        handles / labels: 图例句柄与标签；缺省取轴上已绘制的系列
            （handles 可自带 label，此时 labels 传 None）。
        loc: 轴内或 outside 位置（below=True 时忽略）。
        ncol: 图例列数。
        below: 是否放轴下方（兼容旧图）。
        filled: 轴内图例是否铺实底。
    """
    kw = dict(frameon=False, fontsize=9, labelcolor=INK2)
    if filled:
        kw.update(frameon=True, fancybox=False, edgecolor="none",
                  facecolor=SURFACE, framealpha=0.94, borderpad=0.45)
    if below:
        pos = dict(loc="upper center", bbox_to_anchor=(0.5, -0.18))
        target = ax
    elif isinstance(loc, str) and loc.startswith("outside"):
        # Matplotlib 3.11：outside 只对 figure.legend 生效
        pos = dict(loc=loc)
        target = ax.figure
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    else:
        pos = dict(loc=loc)
        target = ax
    if handles is None:
        target.legend(ncol=ncol, **pos, **kw)
    elif labels is None:
        target.legend(handles=handles, ncol=ncol, **pos, **kw)
    else:
        target.legend(handles, labels, ncol=ncol, **pos, **kw)


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


def block_lines(ax) -> None:
    """修订块边界：τ_m = 6/12/18 h 竖虚线（0/24 为轴缘不画）。"""
    for h in (6.0, 12.0, 18.0):
        ax.axvline(h, color=BASE, lw=1, ls="--", zorder=1)


def waterfall(ax, labels: list[str], values: list[float],
              fmt="{:,.2f}".format) -> list[float]:
    """瀑布图底座：首末柱为总量（INK2），中间柱为增量（正 = 红 /
    负 = 蓝，极性配色），柱间虚线连接器 + 柱顶数值直接标注。

    Args:
        ax: 目标子图（y 轴单位由调用方设置）。
        labels: 各柱 x 标签（可含换行）。
        values: 首柱 = 起始总量、末柱 = 最终总量、中间 = 带符号增量。
        fmt: 柱顶数值格式化（缺省千分位两位小数）。

    Returns:
        各柱之后的累计值（连接器高度序列，供调用方续画）。
    """
    x = np.arange(len(values))
    running, cum, tops = 0.0, [], []
    for i, val in enumerate(values):
        total = i in (0, len(values) - 1)
        if total:
            ax.bar(x[i], val, 0.6, color=INK2)
            running = val
            top = val
        else:
            bottom = running if val >= 0 else running + val
            ax.bar(x[i], abs(val), 0.6, bottom=bottom,
                   color=C_LOAD if val >= 0 else C_PRICE)
            top = max(running, running + val)
            running += val
        txt = fmt(val) if total else (("+" if val >= 0 else "−")
                                      + fmt(abs(val)))
        ax.text(x[i], top, txt, ha="center", va="bottom", fontsize=9,
                color=INK2)
        cum.append(running)
        tops.append(top)
    for i in range(len(values) - 1):
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [cum[i], cum[i]],
                color=BASE, lw=1, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    return cum


def state_matrix(ax, code: np.ndarray, dates: np.ndarray, colors: list[str],
                 *, warmup: int = 0) -> None:
    """状态矩阵底座：无缝隙像元 imshow + 暖机分线（白描边标注）+
    月首纵轴 + 时刻横轴。像元配色由调用方给出（须过 CVD 校验）。

    Args:
        ax: 目标子图（调用方先 style_ax 再调本函数关网格）。
        code: 状态编码矩阵，shape (天数, 144)，值 0..len(colors)−1。
        dates: datetime64 日数组（月首刻度）。
        colors: 各状态像元色（"闲"等中性态放 GRID 灰）。
        warmup: 暖机天数；0 表示不画分线。
    """
    n = len(code)
    ax.imshow(code, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1,
              extent=(0.0, 24.0, float(n), 0.0))
    ax.grid(False)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
    month_axis(ax, dates, n, on_y=True)
    if warmup:
        stroke = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]
        ax.axhline(warmup, color=INK2, lw=1)
        ax.text(0.4, warmup - 0.4, "暖机期", fontsize=8, color=INK2,
                va="bottom", path_effects=stroke)
        ax.text(0.4, warmup + 0.6, "报告期", fontsize=8, color=INK2,
                va="top", path_effects=stroke)


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


def apply_paper_style() -> None:
    """论文出图 rcParams：CJK 正文字体、PDF 以 TrueType 嵌入、留白与刻度。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
                            "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.titlepad": 8.0,
        "axes.labelpad": 5.0,
        "xtick.major.pad": 3.5,
        "ytick.major.pad": 3.5,
        "legend.borderaxespad": 0.5,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.15,
        "legend.labelspacing": 0.35,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def new_fig(w: float, h: float, *, mosaic=None, **gs_kw):
    """新建论文画布。mosaic 非空时走 subplot_mosaic，否则单轴。"""
    apply_paper_style()
    if mosaic is not None:
        fig, axes = plt.subplots(figsize=(w, h), dpi=200, facecolor=SURFACE,
                                 layout="constrained")
        fig.clear()
        axes = fig.subplot_mosaic(mosaic, **gs_kw)
        fig.set_facecolor(SURFACE)
        fig.set_layout_engine("constrained")
        return fig, axes
    fig, ax = plt.subplots(figsize=(w, h), dpi=200, facecolor=SURFACE,
                           layout="constrained")
    return fig, ax


def panel_title(ax, tag: str, title: str = "") -> None:
    """子图编号。说明性文字不进图（留给文件名与题注）；
    title 仅用于面板身份，如日期名。"""
    label = f"({tag})" if not title else f"({tag})  {title}"
    ax.set_title(label, loc="left", fontsize=10.5, color=INK, pad=6)


def check_text_overlap(fig, *, min_area: float = 20.0,
                       clip_pad: float = 28.0) -> list[str]:
    """检查画布上文字是否互相重叠、是否画出画布。

    在 savefig 之前调用。返回问题列表；空列表表示通过。
    相邻刻度数字的轻微重叠忽略；标题与数据标注冲突会报出。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.get_window_extent(renderer)
    items: list[tuple[str, object, object]] = []
    for artist in fig.findobj(Text):
        raw = artist.get_text()
        if not artist.get_visible() or raw is None or not str(raw).strip():
            continue
        try:
            bb = artist.get_window_extent(renderer)
        except (ValueError, RuntimeError):
            continue
        if bb.width < 1.0 or bb.height < 1.0:
            continue
        label = " ".join(str(raw).split())[:36]
        items.append((label, bb, artist))

    issues: list[str] = []
    # 论文图一律 bbox_inches=tight，轴外 1–2 个刻度不算裁切。
    # 只在文字明显落到画布外很远时报警。
    far = max(clip_pad * 3.0, 80.0)
    for label, bb, _ in items:
        if (bb.x0 < fig_box.x0 - far
                or bb.x1 > fig_box.x1 + far
                or bb.y0 < fig_box.y0 - far
                or bb.y1 > fig_box.y1 + far):
            issues.append(f"CLIP {label!r}")
    for i, (s1, b1, t1) in enumerate(items):
        for s2, b2, t2 in items[i + 1:]:
            if t1 is t2 or not b1.overlaps(b2):
                continue
            x0, x1 = max(b1.x0, b2.x0), min(b1.x1, b2.x1)
            y0, y1 = max(b1.y0, b2.y0), min(b1.y1, b2.y1)
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if area < min_area:
                continue
            # 两边都是刻度数字：相邻子图的轴标签擦边，由 spacing 处理，这里放过
            def _tickish(s: str) -> bool:
                t = s.replace("−", "-").replace(",", "").replace(" ", "")
                t = t.replace("万", "").replace("元", "").replace("%", "")
                return t.replace(".", "").replace("-", "").isdigit()
            if _tickish(s1) and _tickish(s2):
                continue
            issues.append(f"OVERLAP {s1!r} × {s2!r} ({area:.0f}px)")
    return issues


def save_paper(fig, out_dir: Path, stem: str) -> tuple[Path, list[str]]:
    """论文落盘：PNG + PDF，先做文字碰撞检查，不关闭调用方以外的逻辑。

    Returns:
        (png 路径, 碰撞/裁切问题列表)。调用方负责 plt.close。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    issues = check_text_overlap(fig)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, facecolor=SURFACE, dpi=300,
                bbox_inches="tight", pad_inches=0.28)
    fig.savefig(pdf, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.28)
    return png, issues
