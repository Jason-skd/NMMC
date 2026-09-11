"""论文图表：调度图、数据总览、对偶分析、S_0 敏感性。

所有类结构化满足 extract.Extractor 协议
（extract(data, sol, out_dir) -> Path），无需显式继承。
配色与图表规范遵循 dataviz 参考实例（浅色模式）：单轴、固定槽位、细标记、隐性网格。

跨图实体配色（固定槽位，颜色跟随实体）：
    p=蓝  λ=橙  g=水青  c=黄  d=洋红  S=绿  R=紫  L=红
唯一例外：ν（储能边际价值）用水青色 dash-dot——与购电 g 不同图出现，无混淆。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from nmmc.data.day import DayData
from nmmc.model.lp import S_MAX, S_MIN, T
from nmmc.model.q1 import Solution

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
C_PRICE, C_DUAL, C_BUY, C_CH, C_DIS, C_SOC = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
C_PV, C_LOAD = "#4a3aa7", "#e34948"

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


def _price_bands(ax, p: np.ndarray) -> list[Patch]:
    """按电价三分位划分谷/平/峰时段，叠加浅色背景带（谷=冷、峰=暖）。

    Args:
        ax: 目标子图。
        p: 逐时段电价，shape (144,)。

    Returns:
        [谷段, 峰段] 两个图例 patch 句柄（由调用方并入各自图例）。
    """
    lo, hi = np.quantile(p, (1 / 3, 2 / 3))
    for mask, color in ((p <= lo, C_PRICE), (p >= hi, C_LOAD)):
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(s / 6.0, (e + 1) / 6.0, color=color, alpha=0.09, lw=0)
    return [Patch(facecolor=C_PRICE, alpha=0.35, lw=0),
            Patch(facecolor=C_LOAD, alpha=0.35, lw=0)]


BAND_LABELS = ["谷段（电价低 1/3）", "峰段（电价高 1/3）"]


class SchedulePlot:
    """三联调度图：电价与边际价值 / 功率流 / 储电量轨迹（q1_schedule.png）。"""

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
        ax.fill_between(hours, sol.lam, data.p, color=C_SOC, alpha=0.25, lw=0,
                        zorder=1.5)
        for a in axes:
            band_patches = _price_bands(a, data.p)
        _style(ax, "电价与母线边际价值（绿色填充 = 储能价值空间 p−λ）", "元/kWh")
        handles, labels_ = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=C_SOC, alpha=0.35, lw=0))
        labels_.append("储能价值空间 $p_t-\\lambda_t$")
        handles += band_patches
        labels_ += BAND_LABELS
        ax.legend(handles, labels_, frameon=False, fontsize=9, labelcolor=INK2,
                  loc="upper left")

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


class DataOverview:
    """附件1 原始数据总览：电价 / 光伏与负载 / 净负荷（q1_data_overview.png）。"""

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染三联数据总览图并返回 PNG 路径。

        Args:
            data: 当日输入数据（用 p、R、L 曲线）。
            sol: LP 最优解（本图未使用，保持协议签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        hours = (np.arange(1, T + 1) - 0.5) / 6.0
        fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True,
                                 dpi=200, facecolor=SURFACE)

        ax = axes[0]
        ax.plot(hours, data.p, color=C_PRICE, lw=2)
        band_patches = _price_bands(ax, data.p)
        _style(ax, "外网购电单价（双谷双峰形态）", "元/kWh")
        ax.legend(band_patches, BAND_LABELS, frameon=False, fontsize=9,
                  labelcolor=INK2, loc="upper right")

        ax = axes[1]
        ax.plot(hours, data.r * 6, color=C_PV, lw=2, label="光伏 $R_t$")
        ax.plot(hours, data.ell * 6, color=C_LOAD, lw=2, label="负载 $L_t$")
        _style(ax, "光伏出力与小区负载", "kW")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)

        ax = axes[2]
        net = (data.ell - data.r) * 6
        ax.plot(hours, net, color=INK2, lw=2)
        ax.axhline(0, color=BASE, lw=1)
        ax.fill_between(hours, net, 0, where=net < 0, color=C_CH, alpha=0.25, lw=0)
        _style(ax, "净负载 $L_t - R_t$（阴影 = 光伏盈余时段）", "kW")

        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlim(0, 24.8)
        ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
        fig.suptitle("附件1 典型日原始数据总览", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q1_data_overview.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        return path


class DualEconomics:
    """对偶分析：电价 p、母线边际价值 λ 与储能边际价值 ν（q1_duals.png）。

    三条曲线同为 元/kWh，单轴合规。ν 平滑如"内部汇率"，p 尖刺；
    放电时段 λ 低于 p（电池是更便宜的边际电源），充电时段 λ = p。
    """
    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染对偶分析图并返回 PNG 路径。

        Args:
            data: 当日输入数据（用 p_t 曲线）。
            sol: LP 最优解（λ、ν 来自约束对偶）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        hours = (np.arange(1, T + 1) - 0.5) / 6.0
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)

        ax.plot(hours, data.p, color=C_PRICE, lw=2, label="电价 $p_t$")
        ax.plot(hours, sol.lam, color=C_DUAL, lw=2, ls="--",
                label="母线边际价值 $\\lambda_t$")
        ax.plot(hours, sol.nu, color=C_BUY, lw=2, ls="-.",
                label="储能边际价值 $\\nu_t$")
        band_patches = _price_bands(ax, data.p)
        _style(ax, "对偶视角：购电时 λ=p；放电时 λ<p（电池为便宜边际源）；ν 平滑",
               "元/kWh")
        handles, labels_ = ax.get_legend_handles_labels()
        handles += band_patches
        labels_ += BAND_LABELS
        ax.legend(handles, labels_, frameon=False, fontsize=9, labelcolor=INK2,
                  loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3)

        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlim(0, 24.8)
        ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
        fig.suptitle("对偶变量与电价（元/kWh）", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q1_duals.png"
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        return path


class S0Basin:
    """S_0 敏感性盆地：钉死 S_0 重解 LP 的费用曲线（q1_s0_sensitivity.png）。

    Attributes:
        grid: 扫描的 S_0 取值数组，kWh（含最优点）。
        costs: 对应的全天购电费，元。
        s0_star: 最优 S_0（自由解）。
        z_star: 最优费用 Z*。
        z6000: S_0=6000（附录标定值）处的费用。
    """

    def __init__(self, grid: np.ndarray, costs: list[float],
                 s0_star: float, z_star: float, z6000: float):
        self.grid = grid
        self.costs = costs
        self.s0_star = s0_star
        self.z_star = z_star
        self.z6000 = z6000

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染 S_0 敏感性曲线并返回 PNG 路径。

        Args:
            data: 当日输入数据（本图未使用，保持协议签名）。
            sol: LP 最优解（本图未使用，保持协议签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)

        ax.plot(self.grid, self.costs, color=C_PRICE, lw=2)
        ax.plot([self.s0_star], [self.z_star], "o", color=C_SOC, ms=9,
                label=f"最优 $S_0^*$ = {self.s0_star:.0f}（自由解）")
        ax.plot([6000.0], [self.z6000], "s", color=C_LOAD, ms=8,
                label=f"附录标定 6000：$Z^{{6000}}$ 高出 "
                      f"{self.z6000 - self.z_star:.2f} 元")
        ax.annotate(f"$Z^*$ = {self.z_star:,.2f} 元",
                    xy=(self.s0_star, self.z_star),
                    xytext=(self.s0_star - 150, self.z_star + 18),
                    fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        _style(ax, "钉死 S_0 重解 LP 的全天购电费（8550 附近为平坦盆地）",
               "全天购电费 Z（元）")

        ax.set_xlabel("$S_0$（kWh）", fontsize=9, color=INK2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper center")
        fig.suptitle("$S_0$ 敏感性：决策化收益有限、最优值附近平坦",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q1_s0_sensitivity.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        return path
