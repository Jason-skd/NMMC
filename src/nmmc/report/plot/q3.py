"""问题3 论文图：修订机制演示、三块费用瀑布、逐级修订阶梯、融合权重
诊断、预报质量（q2 同构复用）、修订行为指纹、q2 vs q3 逐日费用差。

除 EpochsLadder（读阶梯聚合行）与 FusionWeights（读附件3）外，各类
结构化满足题3 导出链签名（export(days, year, p, out_dir) -> Path），
与 report.q3 的表格导出器并列；报告期口径同 report.q2.report_days。

q3 专属配色（极性语言，见 style 模块注记）：上调/正差 = 红、下调/负差 =
蓝；演示图计划 g = 水青（实体槽）、最终注入 a 借紫槽（R 不同图出现）。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nmmc.data.fcst import YearForecast
from nmmc.data.year import YearData
from nmmc.model.forecast3 import fusion_sigmas, hour_powers
from nmmc.model.lp import S_MAX, S_MIN, T
from nmmc.model.q2 import KAPPA
from nmmc.report.plot.q2 import ForecastQuality  # noqa: F401  （q3 复用入口）
from nmmc.report.plot.style import (BASE, C_BUY, C_CH, C_LOAD, C_PRICE,
                                    C_PV, C_SOC, GRID, HOURS, INK, INK2,
                                    MUTED, SURFACE, block_lines, day_axis,
                                    legend, month_axis, price_bands, save,
                                    state_matrix, style_ax, waterfall)
from nmmc.report.q2 import WARMUP_DAYS, report_days
from nmmc.run.q3 import DayResult

EPS = 1e-6  # 调整/紧急的活跃判定阈值，kWh


class EpochsLadder:
    """逐级修订阶梯：Z_online 与 Z_PI 随行使档位变化（q3_epochs_ladder.png）。

    每档为一次独立全年回测（--ladder 模式），数据源为 metrics_row 聚合行，
    不重跑求解器。

    Attributes:
        rows: ladder.csv 的行（metrics_row 输出 + level/delta_z 增列）。
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def export(self, out_dir: Path) -> Path:
        """渲染阶梯图并返回 PNG 路径。"""
        x = np.arange(len(self.rows))
        z = np.array([r["z_online"] for r in self.rows]) / 1e4
        zpi = np.array([r["z_pi"] for r in self.rows]) / 1e4
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200, facecolor=SURFACE)

        ax.plot(x, z, "o-", color=C_PRICE, lw=2, ms=8,
                markeredgecolor=SURFACE, label="Z^online（交付费用）")
        ax.plot(x, zpi, "s--", color=C_SOC, lw=1.5, ms=6,
                markeredgecolor=SURFACE, label="Z^PI（完美信息下界）")
        # 相邻档位的费用降幅标注（负 = 省钱）+ 末档距完美信息的残余差距
        for i in range(1, len(z)):
            ax.annotate(f"Δ {z[i] - z[i - 1]:+,.1f} 万",
                        xy=((x[i] + x[i - 1]) / 2.0, (z[i] + z[i - 1]) / 2.0),
                        xytext=(0, 16), textcoords="offset points",
                        fontsize=8.5, color=INK2, ha="center")
        ax.annotate(f"距完美信息 {z[-1] - zpi[-1]:,.0f} 万",
                    xy=(x[-1], zpi[-1]), xytext=(-8, 22),
                    textcoords="offset points", fontsize=8.5, color=INK2,
                    ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([r["epochs"].replace("-", "/") for r in self.rows])
        ax.set_xlabel("行使修订权的时刻（h）", fontsize=9, color=INK2)
        style_ax(ax, "逐级行使修订权：交付费用下降、与完美信息差距收窄",
                 "报告期费用（万元）")
        legend(ax, ncol=2, below=True)
        fig.suptitle("问题 3 修订机制逐级阶梯（每档独立全年回测）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q3_epochs_ladder.png", tight=True)


class RevisionDemo:
    """单日修订机制演示：计划 vs 最终注入 / 偏差 δ / SOC（q3_revision_demo.png）。

    自动选取报告期 |a−g| 总量最大的日子——修订机制"被用到极致"的样本日；
    B₀ 段（0–6h）无修订权（a≡g），以灰色底带标出。
    """

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染修订演示图并返回 PNG 路径。

        Args:
            days: 全部回测日（取报告期）。
            year: 全年数据（日期标签）。
            p: 逐时段电价（峰谷背景带）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        rep = report_days(days)
        day = rep[int(np.argmax([np.abs(d.a - d.g)[36:].sum() for d in rep]))]
        date = str(year.dates[day.j].astype(object))
        dev_fee = 0.5 * float(p @ np.abs(day.a - day.g))
        g_kw, a_kw = day.g * 6, day.a * 6          # kWh/时段 → kW
        fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True,
                                 dpi=200, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": (3, 2, 2.2)})

        # 面板 1：计划 g 与最终注入 a 双阶梯（a 借紫槽：R 不同图出现）
        ax = axes[0]
        price_bands(ax, p)
        block_lines(ax)
        ax.axvspan(0, 6, color=GRID, alpha=0.5, lw=0)
        ax.plot(HOURS, g_kw, color=C_BUY, lw=2, drawstyle="steps-mid",
                label="计划购电 $g_t$")
        ax.plot(HOURS, a_kw, color=C_PV, lw=2, ls="--",
                drawstyle="steps-mid", label="最终注入 $a_t$")
        style_ax(ax, f"计划 vs 修订（{date}；全年 |a−g| 最大日，"
                     f"偏差费 {dev_fee:,.0f} 元）", "kW")
        handles = [Line2D([], [], color=C_BUY, lw=2),
                   Line2D([], [], color=C_PV, lw=2, ls="--"),
                   Patch(facecolor=GRID, alpha=0.5, lw=0)]
        legend(ax, handles, ["计划购电 $g_t$", "最终注入 $a_t$",
                             "B₀ 段（0–6h 无修订权，a≡g）"])

        # 面板 2：偏差 δ = a − g（红 = 上调 δ+，蓝 = 下调 δ−；τ_m 虚线分块）
        ax = axes[1]
        delta = a_kw - g_kw
        width = 1 / 6 * 0.9
        ax.bar(HOURS, np.where(delta > 0, delta, 0.0), width,
               color=C_LOAD, label="上调 $\\delta^+_t$")
        ax.bar(HOURS, np.where(delta < 0, delta, 0.0), width,
               color=C_PRICE, label="下调 $\\delta^-_t$")
        ax.axhline(0, color=BASE, lw=1)
        block_lines(ax)
        style_ax(ax, "偏差量 $a_t - g_t$（B₀ 段恒零；虚线 = 修订块边界）",
                 "kW")
        legend(ax)

        # 面板 3：真实路径 SOC 与上下限
        ax = axes[2]
        ax.plot(np.arange(T + 1) / 6.0, day.S, color=C_SOC, lw=2,
                drawstyle="steps-post")
        for lvl, txt in ((S_MIN, "下限 1200"), (S_MAX, "上限 10800")):
            ax.axhline(lvl, color=MUTED, lw=1, ls="--")
            ax.text(24.15, lvl, txt, fontsize=8, color=MUTED, va="center")
        ax.set_ylim(0, S_MAX * 1.12)
        style_ax(ax, "真实路径储电量 $S_t$", "kWh")

        day_axis(ax)
        fig.suptitle("问题 3 分块承诺与修订机制演示", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return save(fig, out_dir, "q3_revision_demo.png")


class CostWaterfall:
    """三块费用瀑布：名义计划费 → 基础费差 → 偏差费 → 紧急费 → Z_online
    （q3_cost_waterfall.png）。承诺机制的代价与收益分解，报告期口径。"""

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染费用瀑布并返回 PNG 路径。

        Args:
            days: 全部回测日（取报告期）。
            year: 全年数据（本图未使用，保持导出链签名）。
            p: 逐时段电价（费用折算）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        rep = report_days(days)
        plan = sum(float(p @ d.g) for d in rep)
        base = sum(float(p @ d.a) for d in rep)
        dev = sum(0.5 * float(p @ np.abs(d.a - d.g)) for d in rep)
        emerg = sum(KAPPA * float(p @ d.u) for d in rep)
        z = sum(d.z for d in rep)
        wan = 1e4
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200, facecolor=SURFACE)
        waterfall(ax,
                  ["名义计划费\nΣp·g", "基础费差\nΣp·a−Σp·g",
                   "偏差费\n½Σp|a−g|", "紧急费\nκΣp·u", "Z^online"],
                  [plan / wan, (base - plan) / wan, dev / wan,
                   emerg / wan, z / wan], fmt="{:,.1f}".format)
        ax.margins(y=0.18)
        style_ax(ax, "报告期费用构成（正 = 红，负 = 蓝）：修订以偏差费为代价，"
                     "换取基础购电费下降", "费用（万元）")
        fig.suptitle("问题 3 承诺机制费用分解（报告期）", fontsize=13,
                     color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q3_cost_waterfall.png")


class FusionWeights:
    """反方差融合诊断：两预报器 σ(h) 与官方权重 φ(h)（q3_fusion_weights.png）。

    以年末滚动估计为例（j = 365）：官方误差随提前期劣化、持续法平稳，
    φ(h) 随之下降——附件3 数据核验第 4 条与融合合理性的视觉证据。
    """

    def export(self, year: YearData, fc: YearForecast, out_dir: Path) -> Path:
        """渲染融合诊断图并返回 PNG 路径。

        Args:
            year: 全年数据（整点功率真值）。
            fc: 附件3 官方预报。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        j = len(year.dates)
        sig_fc, sig_pe = fusion_sigmas(hour_powers(year.r), fc.phi, j)
        var_fc, var_pe = sig_fc**2, sig_pe**2
        phi = var_pe / (var_pe + var_fc)
        h = np.arange(1, 25)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                 dpi=200, facecolor=SURFACE)

        # 上联：两预报器误差标准差（官方劣化快、持续法平稳）
        ax = axes[0]
        ax.plot(h, sig_fc, color=C_PV, lw=2, label="官方预报 $\\sigma_{fc}(h)$")
        ax.plot(h, sig_pe, color=C_CH, lw=2, label="持续法 $\\sigma_{pe}(h)$")
        style_ax(ax, "两预报器误差标准差（官方随提前期急剧劣化，持续法平稳）",
                 "σ（kW）")
        legend(ax)

        # 下联：官方权重 φ(h)（0.5 为两预报器等权参考）
        ax = axes[1]
        ax.plot(h, phi, color=C_PRICE, lw=2)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.text(24.15, 0.5, "等权 0.5", fontsize=8, color=MUTED, va="center")
        ax.set_ylim(0, 1.05)
        style_ax(ax, "反方差融合的官方权重 φ(h)（短提前期信官方，长提前期转持续法）",
                 "官方权重 φ")
        ax.set_xlabel("预报提前期（h）", fontsize=9, color=INK2)

        fig.suptitle(f"问题 3 反方差融合诊断（第 {j} 日滚动估计）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return save(fig, out_dir, "q3_fusion_weights.png")


class RevisionHeatmap:
    """全年修订行为指纹：上调/下调/无调整三态（q3_revision_heatmap.png）。

    B₀ 段（0–6h）无修订权恒为"无调整"；极性配色红/蓝为无缝隙像元编码
    （protan ΔE 21.6，dataviz 校验通过）。
    """

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染修订指纹矩阵并返回 PNG 路径。

        Args:
            days: 全部回测日（含暖机期）。
            year: 全年数据（月首刻度）。
            p: 逐时段电价（本图未使用，保持导出链签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        n = len(days)
        diff = np.array([d.a - d.g for d in days])
        code = np.zeros((n, T), dtype=np.uint8)   # 0 无调整 1 上调 2 下调
        code[diff > EPS] = 1
        code[diff < -EPS] = 2
        code[:, :36] = 0                          # B₀ 段无修订权

        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=200, facecolor=SURFACE)
        style_ax(ax, "修订行为指纹（行 = 日，列 = 时段；B₀ 段无修订权）",
                 "月份")
        state_matrix(ax, code, year.dates[:n], [GRID, C_LOAD, C_PRICE],
                     warmup=WARMUP_DAYS)
        handles = [Patch(facecolor=c, lw=0) for c in (C_LOAD, C_PRICE, GRID)]
        legend(ax, handles, ["上调 δ+", "下调 δ−", "无调整（中性底）"],
               ncol=3, below=True)
        fig.suptitle("问题 3 全年修订行为指纹", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return save(fig, out_dir, "q3_revision_heatmap.png", tight=True)


class Q2Q3Delta:
    """q2 vs q3 逐日费用差（q3_vs_q2_cost.png）：分块承诺 + 修订相对题2
    在线随机 MPC 的逐日得失（红 = q3 更贵，蓝 = q3 更省）。

    Attributes:
        q2_csv: 题2 明细 CSV（date, z_online 两列），按日期与题3 报告期对齐。
    """

    DEFAULT_Q2 = Path("results/q2/L1-P1-R2-smpc/q2_details.csv")

    def __init__(self, q2_csv: Path | None = None):
        self.q2_csv = Path(q2_csv) if q2_csv is not None else self.DEFAULT_Q2

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染逐日费用差图并返回 PNG 路径。

        Args:
            days: 全部回测日（取报告期）。
            year: 全年数据（日期对齐与月首刻度）。
            p: 逐时段电价（本图未使用，保持导出链签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        q2 = pd.read_csv(self.q2_csv).set_index("date")["z_online"]
        rep = report_days(days)
        js = [d.j for d in rep]
        dates = np.array([str(year.dates[j].astype(object)) for j in js])
        z2 = q2.reindex(dates).to_numpy(float)
        z3 = np.array([d.z for d in rep])
        ok = ~np.isnan(z2)
        diff = z3[ok] - z2[ok]
        dates_ok = year.dates[np.array(js)[ok]]

        x = np.arange(len(diff))
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)
        ax.bar(x, np.where(diff > 0, diff, 0.0), width=1.0, color=C_LOAD)
        ax.bar(x, np.where(diff < 0, diff, 0.0), width=1.0, color=C_PRICE)
        ax.axhline(0, color=INK2, lw=1)
        # 均值线与零线在全年尺度下近乎重合：加粗 + 右端数值标注保证可读
        ax.axhline(diff.mean(), color=MUTED, lw=1.5, ls="--")
        ax.annotate(f"均值 {diff.mean():+,.0f} 元/日", xy=(len(diff), diff.mean()),
                    xytext=(-4, 10), textcoords="offset points", fontsize=8.5,
                    color=INK2, ha="right",
                    path_effects=[patheffects.withStroke(linewidth=2.5,
                                                         foreground=SURFACE)])
        ax.margins(y=0.12)
        month_axis(ax, dates_ok, len(diff))
        style_ax(ax, "逐日费用差 $Z^{q3}_j - Z^{q2}_j$"
                      f"（报告期合计 {diff.sum():+,.0f} 元）", "费用差（元）")
        handles = [Patch(facecolor=C_LOAD, lw=0),
                   Patch(facecolor=C_PRICE, lw=0),
                   Line2D([], [], color=MUTED, ls="--", lw=1)]
        legend(ax, handles, ["q3 更贵", "q3 更省",
                             f"均值 {diff.mean():+,.0f} 元/日"], ncol=3,
               below=True)
        fig.suptitle("问题 2 vs 问题 3：分块承诺与修订的逐日得失",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q3_vs_q2_cost.png", tight=True)
