"""问题1 论文图：四联调度图、两联数据总览、对偶分析、S_0 敏感性盆地。

所有类结构化满足 report.q1.Exporter 协议
（export(data, sol, out_dir) -> Path），无需显式继承。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from nmmc.data.day import DayData
from nmmc.model.lp import S_MAX, S_MIN, T
from nmmc.model.q1 import Solution
from nmmc.report.plot.style import (BAND_LABELS, BASE, C_BUY, C_CH, C_DIS,
                                    C_DUAL, C_LOAD, C_PV, C_PRICE, C_SOC,
                                    ENDS, GRID, HOURS, INK, INK2, MUTED,
                                    SURFACE, day_axis, legend, price_bands,
                                    runs, save, style_ax)

EPS = 1e-6  # 充/放/弃的活跃判定阈值，kWh


class SchedulePlot:
    """四联调度图：电价与边际价值 / 功率平衡 / 充放状态带 / 储电量轨迹
    （q1_schedule.png）。"""

    def export(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染四联调度图并返回 PNG 路径。

        Args:
            data: 当日输入数据（用 p、R、L 曲线）。
            sol: LP 最优解（λ、充放购电量、SOC 轨迹）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        fig, axes = plt.subplots(
            4, 1, figsize=(9, 8.6), sharex=True, dpi=200, facecolor=SURFACE,
            gridspec_kw={"height_ratios": (3.0, 3.4, 0.5, 2.6)})
        for a in (axes[0], axes[1], axes[3]):
            band_patches = price_bands(a, data.p)

        # 面板 1：电价与母线边际价值（绿色填充 = 储能价值空间 p−λ）
        ax = axes[0]
        ax.plot(HOURS, data.p, color=C_PRICE, lw=2, label="电价 $p_t$")
        ax.plot(HOURS, sol.lam, color=C_DUAL, lw=2, ls="--",
                label="母线边际价值 $\\lambda_t$")
        ax.fill_between(HOURS, sol.lam, data.p, color=C_SOC, alpha=0.25,
                        lw=0, zorder=1.5)
        style_ax(ax, "电价与母线边际价值（绿色填充 = 储能价值空间 p−λ）",
               "元/kWh")
        handles, labels_ = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=C_SOC, alpha=0.35, lw=0), *band_patches]
        labels_ += ["储能价值空间 $p_t-\\lambda_t$", *BAND_LABELS]
        legend(ax, handles, labels_, loc="upper left")

        # 面板 2：供需双侧堆叠柱——上 = 供给侧 g+R+d，下 = 需求侧 L+c+w，
        # 两侧等高即逐时段功率平衡 g+R+d = L+c+w 的视觉证明；
        # 段间 SURFACE 描边 = 2px 缝隙（紫↔洋红 CVD 6–8 带的次级编码）
        ax = axes[1]
        width = 1 / 6 * 0.9
        edge = dict(edgecolor=SURFACE, linewidth=0.6)
        ax.bar(HOURS, sol.g * 6, width, color=C_BUY, label="购电 $g_t$", **edge)
        ax.bar(HOURS, data.r * 6, width, bottom=sol.g * 6, color=C_PV,
               label="光伏 $R_t$", **edge)
        ax.bar(HOURS, sol.d * 6, width, bottom=(sol.g + data.r) * 6,
               color=C_DIS, label="放电 $d_t$", **edge)
        ax.bar(HOURS, -data.ell * 6, width, color=C_LOAD,
               label="负载 $L_t$", **edge)
        ax.bar(HOURS, -sol.c * 6, width, bottom=-data.ell * 6, color=C_CH,
               label="充电 $c_t$", **edge)
        if sol.w.sum() > EPS:
            ax.bar(HOURS, -sol.w * 6, width,
                   bottom=-(data.ell + sol.c) * 6, color=MUTED,
                   label="弃光 $w_t$", **edge)
        ax.axhline(0, color=BASE, lw=1)
        style_ax(ax, "功率平衡（上：供给侧 g+R+d；下：需求侧 L+c+w，两侧等高 = 平衡）",
               "kW")
        legend(ax, ncol=3)

        # 面板 3：充/放/闲状态带（游程横条，替代从柱状图符号读状态）
        ax = axes[2]
        code = np.zeros(T, dtype=np.int8)
        code[sol.c > EPS] = 1
        code[sol.d > EPS] = 2
        for s, n, v in runs(code):
            ax.broken_barh([(s / 6.0, n / 6.0)], (0, 1),
                           color=(GRID, C_CH, C_DIS)[v], lw=0)
        style_ax(ax, "电池状态（黄 = 充电，洋红 = 放电，灰 = 闲置）", "")
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        # 面板 4：储电量轨迹（时段常数，steps-post 为诚实形态）
        ax = axes[3]
        ax.plot(ENDS, sol.S, color=C_SOC, lw=2, drawstyle="steps-post")
        for lvl, txt in ((S_MIN, "下限 1200"), (S_MAX, "上限 10800")):
            ax.axhline(lvl, color=MUTED, lw=1, ls="--")
            ax.text(24.15, lvl, txt, fontsize=8, color=MUTED, va="center")
        ax.set_ylim(0, S_MAX * 1.12)
        style_ax(ax, "储电量 $S_t$ 轨迹", "kWh")

        day_axis(ax)
        fig.suptitle(f"问题 1 典型日调度计划（Z* = {sol.cost:,.2f} 元）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return save(fig, out_dir, "q1_schedule.png")


class DataOverview:
    """附件1 数据总览（两联）：电价 / 负载与光伏（q1_data_overview.png）。

    第二联两线之间的面积即净负荷：红 = 缺口（需购电/放电），
    黄 = 光伏盈余（可充/弃）——原始双线与净负荷合并为一个面板。
    """

    def export(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染两联数据总览图并返回 PNG 路径。

        Args:
            data: 当日输入数据（用 p、R、L 曲线）。
            sol: LP 最优解（本图未使用，保持协议签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True,
                                 dpi=200, facecolor=SURFACE)

        # 面板 1：电价与谷/峰背景带
        ax = axes[0]
        ax.plot(HOURS, data.p, color=C_PRICE, lw=2)
        band_patches = price_bands(ax, data.p)
        style_ax(ax, "外网购电单价（双谷双峰形态）", "元/kWh")
        legend(ax, band_patches, BAND_LABELS, loc="upper center")

        # 面板 2：负载与光伏交叠，线间面积分色 = 净负荷的缺口/盈余
        ax = axes[1]
        ax.plot(HOURS, data.ell * 6, color=C_LOAD, lw=2, label="负载 $L_t$")
        ax.plot(HOURS, data.r * 6, color=C_PV, lw=2, label="光伏 $R_t$")
        ax.fill_between(HOURS, data.r * 6, data.ell * 6, where=data.ell >= data.r,
                        color=C_LOAD, alpha=0.25, lw=0, label="缺口（需购电/放电）")
        ax.fill_between(HOURS, data.r * 6, data.ell * 6, where=data.ell < data.r,
                        color=C_CH, alpha=0.25, lw=0, label="光伏盈余（可充/弃）")
        ax.axhline(0, color=BASE, lw=1)
        style_ax(ax, "负载与光伏出力（两线之间即净负荷：红 = 缺口，黄 = 盈余）",
               "kW")
        legend(ax, ncol=2)

        day_axis(ax)
        fig.suptitle("附件1 典型日原始数据总览", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return save(fig, out_dir, "q1_data_overview.png")


class DualEconomics:
    """对偶分析：电价 p、母线边际价值 λ 与储能边际价值 ν（q1_duals.png）。

    三条曲线同为 元/kWh，单轴合规。ν 平滑如"内部汇率"，p 尖刺；
    放电时段 λ 低于 p（电池是更便宜的边际电源），充电时段 λ = p。
    """
    def export(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染对偶分析图并返回 PNG 路径。

        Args:
            data: 当日输入数据（用 p_t 曲线）。
            sol: LP 最优解（λ、ν 来自约束对偶）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)

        ax.plot(HOURS, data.p, color=C_PRICE, lw=2, label="电价 $p_t$")
        ax.plot(HOURS, sol.lam, color=C_DUAL, lw=2, ls="--",
                label="母线边际价值 $\\lambda_t$")
        ax.plot(HOURS, sol.nu, color=C_BUY, lw=2, ls="-.",
                label="储能边际价值 $\\nu_t$")
        band_patches = price_bands(ax, data.p)
        style_ax(ax, "对偶视角：购电时 λ=p；放电时 λ<p（电池为便宜边际源）；ν 平滑",
               "元/kWh")
        handles, labels_ = ax.get_legend_handles_labels()
        handles += band_patches
        labels_ += BAND_LABELS
        legend(ax, handles, labels_, ncol=3, below=True)

        day_axis(ax)
        fig.suptitle("对偶变量与电价（元/kWh）", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q1_duals.png", tight=True)


class S0Basin:
    """S_0 敏感性盆地：钉死 S_0 重解 LP 的费用曲线（q1_s0_sensitivity.png）。

    Attributes:
        grid: 扫描的 S_0 取值数组，kWh（含最优点）。
        costs: 对应的全天购电费，元。
        s0_star: 最优 S_0（自由解）。
        z_star: 最优费用 Z*。
        z6000: S_0=6000（附录标定值）处的费用。
    """

    BASIN_TOL = 1.0   # 平坦盆地判据：费用抬升 ≤ 该值（元）

    def __init__(self, grid: np.ndarray, costs: list[float],
                 s0_star: float, z_star: float, z6000: float):
        self.grid = grid
        self.costs = costs
        self.s0_star = s0_star
        self.z_star = z_star
        self.z6000 = z6000

    def export(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """渲染 S_0 敏感性曲线并返回 PNG 路径。

        Args:
            data: 当日输入数据（本图未使用，保持协议签名）。
            sol: LP 最优解（本图未使用，保持协议签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)

        # 平坦盆地阴影：费用抬升 ≤ BASIN_TOL 的 S_0 连续区段（宽度并入图例）
        flat = np.flatnonzero(
            np.asarray(self.costs) <= self.z_star + self.BASIN_TOL)
        basin_label = None
        if len(flat):
            basin_patch = Patch(facecolor=C_SOC, alpha=0.10, lw=0)
            ax.axvspan(self.grid[flat[0]], self.grid[flat[-1]],
                       color=C_SOC, alpha=0.10, lw=0)
            basin_label = (f"平坦盆地（宽 "
                           f"{self.grid[flat[-1]] - self.grid[flat[0]]:.0f}"
                           f" kWh，Z 抬升 ≤ {self.BASIN_TOL:g} 元）")

        ax.plot(self.grid, self.costs, color=C_PRICE, lw=2)
        ax.plot([self.s0_star], [self.z_star], "o", color=C_SOC, ms=9,
                markeredgecolor=SURFACE, label=f"最优 $S_0^*$ = "
                f"{self.s0_star:.0f}（自由解）")
        ax.plot([6000.0], [self.z6000], "s", color=C_LOAD, ms=8,
                markeredgecolor=SURFACE,
                label=f"附录标定 6000：$Z^{{6000}}$ 高出 "
                      f"{self.z6000 - self.z_star:.2f} 元")
        ax.annotate(f"$Z^*$ = {self.z_star:,.2f} 元",
                    xy=(self.s0_star, self.z_star),
                    xytext=(self.s0_star - 150, self.z_star + 18),
                    fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        style_ax(ax, "钉死 S_0 重解 LP 的全天购电费（绿色阴影 = 平坦盆地）",
               "全天购电费 Z（元）")

        ax.set_xlabel("$S_0$（kWh）", fontsize=9, color=INK2)
        handles, labels_ = ax.get_legend_handles_labels()
        if basin_label is not None:
            handles.append(basin_patch)
            labels_.append(basin_label)
        # 越位右上：曲线左端（低 S_0）高、右端平低，左上与上方均被占
        legend(ax, handles, labels_, loc="upper right")
        fig.suptitle("$S_0$ 敏感性：决策化收益有限、最优值附近平坦",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q1_s0_sensitivity.png")
