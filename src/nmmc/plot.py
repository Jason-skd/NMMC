"""调度图：三联时序面板（电价与边际价值 / 功率流 / 储电量轨迹）。

SchedulePlot 结构化满足 extract.Extractor 协议
（extract(data, sol, out_dir) -> Path），无需显式继承。
配色与图表规范遵循 dataviz 参考实例（浅色模式）：单轴、固定槽位、细标记、隐性网格。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nmmc.data import DayData
from nmmc.lp import S_MAX, S_MIN, Solution, T

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
# 分类槽位按实体固定分配：电价/边际价值/购电/充电/放电/储电量
C_PRICE, C_DUAL, C_BUY, C_CH, C_DIS, C_SOC = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei",
                        "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
})


def _style(ax, title: str, ylabel: str) -> None:
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


class SchedulePlot:
    """三联调度图：电价与边际价值 / 功率流 / 储电量轨迹（q1_schedule.png）。

    结构化满足 extract.Extractor 协议，无需显式继承。
    """

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染三联调度图并返回 PNG 路径。

        Args:
            data: 当日输入数据（用 p_t 曲线）。
            sol: LP 最优解（λ、充放购电量、SOC 轨迹）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        hours = (np.arange(1, T + 1) - 0.5) / 6.0   # 各时段中心，h
        fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True,
                                 dpi=200, facecolor=SURFACE)

        ax = axes[0]
        ax.plot(hours, data.p, color=C_PRICE, lw=2, label="电价 $p_t$")
        ax.plot(hours, sol.lam, color=C_DUAL, lw=2, ls="--",
                label="母线边际价值 $\\lambda_t$")
        _style(ax, "电价与母线边际价值（削峰填谷：λ 在放电时段低于 p）", "元/kWh")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

        ax = axes[1]
        width = 1 / 6 * 0.9
        ax.bar(hours, sol.c * 6, width=width, color=C_CH, label="充电 $c_t$")
        ax.bar(hours, -sol.d * 6, width=width, color=C_DIS, label="放电 $d_t$")
        ax.plot(hours, sol.g * 6, color=C_BUY, lw=2, label="购电 $g_t$")
        ax.axhline(0, color=BASE, lw=1)
        _style(ax, "功率流（柱：充电向上 / 放电向下；折线：购电）", "kW")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

        ax = axes[2]
        ends = np.arange(T + 1) / 6.0
        ax.plot(ends, sol.S, color=C_SOC, lw=2, drawstyle="steps-post")
        for lvl, txt in ((S_MIN, "下限 1200"), (S_MAX, "上限 10800")):
            ax.axhline(lvl, color=MUTED, lw=1, ls="--")
            ax.text(24.15, lvl, txt, fontsize=8, color=MUTED, va="center")
        ax.set_ylim(0, S_MAX * 1.12)
        _style(ax, "储电量 $S_t$ 轨迹", "kWh")

        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlim(0, 24.8)
        ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
        fig.suptitle(f"问题 1 典型日调度计划（Z* = {sol.cost:,.2f} 元）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q1_schedule.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        return path
