"""问题2 论文图：预报质量四联、逐日费用主线、SOC 全年包络、调度状态矩阵。

所有类结构化满足题2 导出链签名
（export(days, year, p, out_dir) -> Path），与 report.q2 的表格导出器并列。
报告期口径与 report.q2.report_days 一致（j ≥ 31，暖机期不交付）。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nmmc.data.year import YearData
from nmmc.model.lp import S_MAX, S_MIN, T
from nmmc.report.plot.style import (C_CH, C_DIS, C_EMERG, C_LOAD, C_PV,
                                    C_PRICE, C_SOC, ENDS, GRID, INK, INK2,
                                    MUTED, SURFACE, day_axis, legend,
                                    month_axis, save, style_ax)
from nmmc.report.q2 import WARMUP_DAYS, report_days
from nmmc.run.q2 import DayResult

EPS = 1e-6  # 充/放/紧急的活跃判定阈值，kWh


class ForecastQuality:
    """预报质量四联图：预报 vs 实际散点 + 逐时残差箱线
    （q2_forecast_quality.png）。预报层是题2 的建模贡献，此图为其证据。"""

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染预报质量图并返回 PNG 路径。

        Args:
            days: 全部回测日（取报告期）。
            year: 全年数据（真值）。
            p: 逐时段电价（本图未使用，保持导出链签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        rep = report_days(days)
        js = [d.j for d in rep]
        ell, ell_hat = year.ell[js], np.array([d.ell_hat for d in rep])
        r, r_hat = year.r[js], np.array([d.r_hat for d in rep])
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=200,
                                 facecolor=SURFACE)

        # 上排：预报 vs 实际散点（45° 虚线 = 完美预报；光伏滤除全夜时段）
        for ax, hat, act, col, name in (
                (axes[0, 0], ell_hat, ell, C_LOAD, "负载"),
                (axes[0, 1], r_hat, r, C_PV, "光伏")):
            if name == "光伏":
                day_mask = (act + hat) > EPS
                hat, act, note = hat[day_mask], act[day_mask], "，仅白天"
            else:
                note = ""
            mae = float(np.abs(hat - act).mean())
            lim = float(max(act.max(), hat.max())) * 1.02
            ax.plot([0.0, lim], [0.0, lim], ls="--", color=MUTED, lw=1)
            ax.scatter(act, hat, s=4, lw=0, alpha=0.25, color=col)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            style_ax(ax, f"{name}预报 vs 实际（MAE {mae:.2f}{note}）",
                     "预报（kWh）")
            ax.set_xlabel("实际（kWh）", fontsize=9, color=INK2)

        # 下排：残差箱线（预报 − 实际，按小时聚合 6 时段；滤离群点保形态）
        for ax, hat, act, col, name in (
                (axes[1, 0], ell_hat - ell, None, C_LOAD, "负载"),
                (axes[1, 1], r_hat - r, None, C_PV, "光伏")):
            groups = [hat[:, 6 * h:6 * (h + 1)].ravel() for h in range(24)]
            bp = ax.boxplot(groups, showfliers=False, widths=0.6,
                            patch_artist=True,
                            medianprops=dict(color=INK, lw=1.2),
                            whiskerprops=dict(color=MUTED, lw=0.8),
                            capprops=dict(color=MUTED, lw=0.8))
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.45, edgecolor=MUTED, lw=0.8)
            ax.axhline(0.0, color=MUTED, lw=1, ls="--")
            style_ax(ax, f"{name}残差按时段分布（箱 = 四分位，须 = 1.5 IQR）",
                     "残差（kWh）")
            ax.set_xticks(range(1, 25, 3))
            ax.set_xticklabels([str(h) for h in range(0, 24, 3)])
            ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)

        fig.suptitle(f"问题 2 预报层质量（报告期 {len(rep)} 天，"
                     "0:00 点预报 vs 实际）", fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return save(fig, out_dir, "q2_forecast_quality.png")


class DailyCost:
    """全年逐日费用主线：现金费用 Z、暖机期阴影与报告期均值
    （q2_daily_cost.png）。回测主结果图，暖机期退化一眼可见。"""

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染逐日费用主线图并返回 PNG 路径。

        Args:
            days: 全部回测日（含暖机期）。
            year: 全年数据（月首刻度）。
            p: 逐时段电价（本图未使用，保持导出链签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        z = np.array([d.z for d in days])
        z_rep = z[WARMUP_DAYS:]
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)

        ax.plot(np.arange(len(z)), z, color=C_PRICE, lw=1.2)
        ax.axvspan(0, WARMUP_DAYS, color=GRID, alpha=0.6, lw=0)
        ax.axhline(z_rep.mean(), color=MUTED, lw=1, ls="--")
        month_axis(ax, year.dates[:len(z)], len(z))
        style_ax(ax, "全年滚动逐日现金费用（灰带 = 暖机期，虚线 = 报告期均值）",
                 "日费用 Z（元）")
        handles = [
            Line2D([], [], color=C_PRICE, lw=1.2, label="逐日现金费用 $Z^{(j)}$"),
            Patch(facecolor=GRID, alpha=0.6, lw=0,
                  label=f"暖机期（前 {WARMUP_DAYS} 天，不交付）"),
            Line2D([], [], color=MUTED, ls="--", lw=1,
                   label=f"报告期均值 {z_rep.mean():,.0f} 元/日"),
        ]
        legend(ax, handles, ncol=3, below=True)

        fig.suptitle(f"问题 2 全年回测费用主线"
                     f"（报告期 Z^online = {z_rep.sum():,.0f} 元）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q2_daily_cost.png", tight=True)


class SocEnvelope:
    """SOC 轨迹全年包络：分位带 + 中位轨迹 + 上下限（q2_soc_envelope.png）。

    展示运行规律（早晚充、峰段放）在全年的稳定性，与题1 单日图
    形成"单日 → 全年"呼应。
    """

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染 SOC 包络图并返回 PNG 路径。

        Args:
            days: 全部回测日（分位取报告期）。
            year: 全年数据（本图未使用，保持导出链签名）。
            p: 逐时段电价（本图未使用，保持导出链签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        S = np.array([d.S for d in report_days(days)])
        q05, q25, med, q75, q95 = np.quantile(
            S, (0.05, 0.25, 0.5, 0.75, 0.95), axis=0)
        fig, ax = plt.subplots(figsize=(9, 4), dpi=200, facecolor=SURFACE)

        ax.fill_between(ENDS, q05, q95, color=C_SOC, alpha=0.12, lw=0,
                        label="5–95% 分位带")
        ax.fill_between(ENDS, q25, q75, color=C_SOC, alpha=0.25, lw=0,
                        label="25–75% 分位带")
        ax.plot(ENDS, med, color=C_SOC, lw=2, label="中位轨迹")
        for lvl, txt in ((S_MIN, "下限 1200"), (S_MAX, "上限 10800")):
            ax.axhline(lvl, color=MUTED, lw=1, ls="--")
            ax.text(24.15, lvl, txt, fontsize=8, color=MUTED, va="center")
        ax.set_ylim(0, S_MAX * 1.12)
        style_ax(ax, f"储电量 $S_t$ 全年分位包络（报告期 {len(S)} 天）",
                 "kWh")

        day_axis(ax)
        legend(ax, ncol=3, below=True)
        fig.suptitle("问题 2 SOC 运行规律（早晚充、峰段放的全年稳定性）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save(fig, out_dir, "q2_soc_envelope.png", tight=True)


class DispatchHeatmap:
    """全年调度状态矩阵：日 × 时段，充/放/闲/紧急四态
    （q2_dispatch_heatmap.png）。全年行为指纹，暖机前后与季节性一图定性。

    着色优先级：紧急 > 放电 > 充电 > 闲（同一时段多态并存时标最严重者）。
    """

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """渲染调度状态矩阵并返回 PNG 路径。

        Args:
            days: 全部回测日（含暖机期，暖机/报告期以横线分隔）。
            year: 全年数据（月首刻度）。
            p: 逐时段电价（本图未使用，保持导出链签名）。
            out_dir: 输出目录。

        Returns:
            写出的 PNG 文件路径。
        """
        n = len(days)
        code = np.zeros((n, T), dtype=np.uint8)   # 0 闲 1 充 2 放 3 紧急
        code[np.array([d.c for d in days]) > EPS] = 1
        code[np.array([d.d for d in days]) > EPS] = 2
        code[np.array([d.u for d in days]) > EPS] = 3

        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=200, facecolor=SURFACE)
        ax.imshow(code, aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([GRID, C_CH, C_DIS, C_EMERG]),
                  vmin=0, vmax=3, extent=(0.0, 24.0, float(n), 0.0))
        # 暖机/报告分区标注：紧贴分隔线上下、白描边保证在彩色像元上可读
        stroke = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]
        ax.axhline(WARMUP_DAYS, color=INK2, lw=1)
        ax.text(0.4, WARMUP_DAYS - 0.4, "暖机期", fontsize=8, color=INK2,
                va="bottom", path_effects=stroke)
        ax.text(0.4, WARMUP_DAYS + 0.6, "报告期", fontsize=8, color=INK2,
                va="top", path_effects=stroke)
        style_ax(ax, "调度状态指纹（行 = 日，列 = 时段；紧急优先于充放标记）",
                 "月份")
        ax.grid(False)

        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xlim(0, 24)
        ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
        month_axis(ax, year.dates[:n], n, on_y=True)
        handles = [Patch(facecolor=c, lw=0) for c in
                   (C_CH, C_DIS, GRID, C_EMERG)]
        labels_ = ["充电", "放电", "闲置（中性底）", "紧急购电"]
        legend(ax, handles, labels_, ncol=4, below=True)

        fig.suptitle("问题 2 全年调度行为指纹（充/放/闲/紧急四态）",
                     fontsize=13, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return save(fig, out_dir, "q2_dispatch_heatmap.png", tight=True)
