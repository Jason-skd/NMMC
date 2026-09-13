"""国一论文结果图：从定稿 xlsx / csv / 附件重绘，不依赖滚动轨迹 pickle。

编号与 docs 叙事对齐：fig03–fig08 题 1，fig10–fig13 题 2，
fig15–fig18 题 3（fig18 修订阶梯），fig19 / fig21 题 4；
框架示意图在 schematic 模块。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nmmc.data import ROOT, T
from nmmc.data.day import DayData, load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig, forecast_load, forecast_pv
from nmmc.model.forecast3 import hour_powers
from nmmc.model.lp import ETA, S_MAX, S_MIN
from nmmc.model.q3 import VINTAGE_HOURS
from nmmc.report.plot.q3 import EpochsLadder
from nmmc.report.plot.style import (BAND_LABELS, BASE, C_BUY, C_CH, C_DIS,
                                    C_DUAL, C_EMERG, C_LOAD, C_PRICE, C_PV,
                                    C_SOC, ENDS, GRID, HOURS, INK, INK2, MUTED,
                                    SURFACE, apply_paper_style, day_axis,
                                    halo, legend, month_axis, panel_title,
                                    price_bands, runs, save_paper, style_ax)
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.run.q1 import run_q1

EPS = 1e-6
PAPER_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
PAPER_DATE_CN = ("春分 3.20", "夏至 6.21", "秋分 9.23", "冬至 12.21")
Q2_DIR = ROOT / "results" / "q2_v6" / "L1-P1-R2-smpc"
Q3_DIR = ROOT / "results" / "q3_v7" / "F1-L1-LU1-S-b"


def _pads(fig, *, hspace: float = 0.16, wspace: float = 0.12) -> None:
    """拉开子图间距，避免相邻轴刻度互相撞。"""
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(h_pad=0.05, w_pad=0.05, hspace=hspace, wspace=wspace)


# ---------------------------------------------------------------------------
# 读数
# ---------------------------------------------------------------------------

def _as_date(series) -> pd.DatetimeIndex:
    return pd.to_datetime(series).dt.normalize()


def load_plan(xlsx: Path, sheet: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """result2/3 购电表：334 × 144，按行号对位（模板错一格，不改标签）。"""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    dates = _as_date(df.iloc[:, 0])
    g = df.iloc[:, 1:1 + T].to_numpy(float)
    return dates, g


def load_battery(xlsx: Path) -> pd.DataFrame:
    """4 小时块充放电 + 日初/日末储电量。返回按日索引的表。"""
    raw = pd.read_excel(xlsx, sheet_name="充放电量")
    raw["日期"] = raw["日期"].ffill()
    rows = []
    for date, grp in raw.groupby("日期", sort=False):
        g = grp.reset_index(drop=True)
        c = g["充电量"].to_numpy(float)
        d = g["放电量"].to_numpy(float)
        s0 = float(g.loc[0, "储电量"])
        s144 = float(g.loc[1, "储电量"])
        s = np.empty(7)
        s[0] = s0
        s[1:] = s0 + np.cumsum(c - d)
        rows.append({
            "date": pd.Timestamp(date).normalize(),
            "c": c, "d": d, "s4h": s, "S0": s0, "S144": s144,
        })
    return pd.DataFrame(rows).set_index("date")


def load_daily(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")


# ---------------------------------------------------------------------------
# 题 1
# ---------------------------------------------------------------------------

def fig03_data_overview(day: DayData, out_dir: Path):
    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.9), sharex=True,
                             dpi=200, facecolor=SURFACE, layout="constrained")
    _pads(fig, hspace=0.18)
    ax = axes[0]
    ax.plot(HOURS, day.p, color=C_PRICE, lw=2.1)
    bands = price_bands(ax, day.p)
    style_ax(ax, "", "元/kWh")
    panel_title(ax, "a")
    legend(ax, bands, BAND_LABELS, loc="upper left", filled=True)

    ax = axes[1]
    load_kw, pv_kw = day.ell * 6, day.r * 6
    ax.plot(HOURS, load_kw, color=C_LOAD, lw=2.1, label="负载")
    ax.plot(HOURS, pv_kw, color=C_PV, lw=2.1, label="光伏")
    ax.fill_between(HOURS, pv_kw, load_kw, where=load_kw >= pv_kw,
                    color=C_LOAD, alpha=0.22, lw=0, label="缺口")
    ax.fill_between(HOURS, pv_kw, load_kw, where=load_kw < pv_kw,
                    color=C_CH, alpha=0.22, lw=0, label="光伏盈余")
    style_ax(ax, "", "kW")
    panel_title(ax, "b")
    legend(ax, ncol=4, loc="upper left", filled=True)
    day_axis(ax)
    return save_paper(fig, out_dir, "fig03_附件1典型日-电价负荷光伏")


def fig04_schedule(day: DayData, sol, out_dir: Path):
    apply_paper_style()
    fig, axes = plt.subplots(
        4, 1, figsize=(10.2, 9.5), sharex=True, dpi=200,
        facecolor=SURFACE, layout="constrained",
        gridspec_kw={"height_ratios": (2.8, 3.2, 0.45, 2.4)})
    _pads(fig, hspace=0.24)
    for a in (axes[0], axes[1], axes[3]):
        price_bands(a, day.p)

    ax = axes[0]
    ax.plot(HOURS, day.p, color=C_PRICE, lw=2.0, label=r"电价 $p_t$")
    ax.plot(HOURS, sol.lam, color=C_DUAL, lw=1.8, ls="--",
            label=r"母线边际 $\lambda_t$")
    ax.plot(HOURS, sol.nu, color=C_BUY, lw=1.8, ls="-.",
            label=r"储能边际 $\nu_t$")
    ax.fill_between(HOURS, sol.lam, day.p, color=C_SOC, alpha=0.18, lw=0)
    style_ax(ax, "", "元/kWh")
    panel_title(ax, "a")
    legend(ax, ncol=3, loc="upper left", filled=True)

    ax = axes[1]
    w = (1 / 6) * 0.88
    edge = dict(edgecolor=SURFACE, linewidth=0.4)
    ax.bar(HOURS, sol.g * 6, w, color=C_BUY, label="购电", **edge)
    ax.bar(HOURS, day.r * 6, w, bottom=sol.g * 6, color=C_PV,
           label="光伏", **edge)
    ax.bar(HOURS, sol.d * 6, w, bottom=(sol.g + day.r) * 6,
           color=C_DIS, label="放电", **edge)
    ax.bar(HOURS, -day.ell * 6, w, color=C_LOAD, label="负载", **edge)
    ax.bar(HOURS, -sol.c * 6, w, bottom=-day.ell * 6, color=C_CH,
           label="充电", **edge)
    ax.axhline(0, color=BASE, lw=0.8)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.22)
    style_ax(ax, "", "kW")
    panel_title(ax, "b")
    legend(ax, ncol=3, loc="upper left", filled=True)

    ax = axes[2]
    code = np.zeros(T, dtype=np.int8)
    code[sol.c > EPS] = 1
    code[sol.d > EPS] = 2
    for s, n, v in runs(code):
        ax.broken_barh([(s / 6.0, n / 6.0)], (0, 1),
                       color=(GRID, C_CH, C_DIS)[v], lw=0)
    style_ax(ax, "", "")
    panel_title(ax, "c")
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    ax = axes[3]
    ax.plot(ENDS, sol.S, color=C_SOC, lw=2.0, drawstyle="steps-post")
    ax.axhline(S_MIN, color=MUTED, lw=0.9, ls="--")
    ax.axhline(S_MAX, color=MUTED, lw=0.9, ls="--")
    halo(ax.text(1.2, S_MAX + 220, "上限 10800", fontsize=8, color=MUTED,
                 va="bottom"))
    halo(ax.text(1.2, S_MIN - 80, "下限 1200", fontsize=8, color=MUTED,
                 va="top"))
    ax.set_ylim(0, S_MAX * 1.16)
    style_ax(ax, "", "kWh")
    panel_title(ax, "d")
    day_axis(ax)
    return save_paper(fig, out_dir, "fig04_题1典型日调度")


def fig05_energy_balance(day: DayData, sol, out_dir: Path):
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.15), dpi=200,
                             facecolor=SURFACE, layout="constrained")
    _pads(fig, wspace=0.20)
    inflow = np.array([sol.g.sum(), day.r.sum(), sol.d.sum()])
    outflow = np.array([day.ell.sum(), sol.c.sum(), sol.w.sum()])
    in_lab = ("购电 $g$", "光伏 $r$", "放电 $d$")
    out_lab = ("负载 $\\ell$", "充电 $c$", "弃电 $w$")
    in_c = (C_BUY, C_PV, C_DIS)
    out_c = (C_LOAD, C_CH, MUTED)

    ax = axes[0]
    left = 0.0
    for val, lab, col in zip(inflow, in_lab, in_c):
        ax.barh(1, val, left=left, color=col, height=0.55)
        if val > 8000:
            ax.text(left + val / 2, 1, f"{val:,.0f}", ha="center", va="center",
                    fontsize=8.5, color="white")
        left += val
    left = 0.0
    for val, lab, col in zip(outflow, out_lab, out_c):
        ax.barh(0, val, left=left, color=col, height=0.55)
        if val > 8000:
            ax.text(left + val / 2, 0, f"{val:,.0f}", ha="center", va="center",
                    fontsize=8.5, color="white")
        left += val
    ax.set_yticks([0, 1], ["流出母线", "流入母线"])
    ax.set_xlabel("kWh", fontsize=9, color=INK2)
    style_ax(ax, "", "")
    ax.grid(False)
    panel_title(ax, "a")
    handles = [Patch(facecolor=c, lw=0) for c in (*in_c, *out_c)]
    legend(ax, handles, [*in_lab, *out_lab], ncol=3, loc="outside lower center")
    ax.set_xlim(0, inflow.sum() * 1.02)
    ax.set_ylim(-0.55, 1.55)

    ax = axes[1]
    bag, take = ETA * sol.c.sum(), sol.d.sum() / ETA
    loss = (1 - ETA) * sol.c.sum() + (1 / ETA - 1) * sol.d.sum()
    names = ("充电落袋\n" r"$\eta\sum c$", "放电取出\n" r"$\sum d/\eta$",
             "往返损耗", r"$\Delta S$")
    vals = (bag, take, loss, sol.S[-1] - sol.S[0])
    cols = (C_CH, C_DIS, MUTED, C_SOC)
    ax.barh(range(4), vals, color=cols, height=0.55)
    xmax = max(abs(v) for v in vals) * 1.22
    for i, v in enumerate(vals):
        if abs(v) < 1.0:
            ax.text(xmax * 0.02, i, "0（昼夜循环）", va="center",
                    fontsize=8.5, color=INK2)
        else:
            ax.text(v + xmax * 0.015, i, f"{v:,.0f}", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("kWh", fontsize=9, color=INK2)
    style_ax(ax, "", "")
    ax.grid(axis="y", visible=False)
    panel_title(ax, "b")
    return save_paper(fig, out_dir, "fig05_典型日电量对账")


def fig06_cost_compare(sol, baselines: dict, out_dir: Path):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.2), dpi=200, facecolor=SURFACE,
                           layout="constrained")
    labels = (r"无电池直买  $Z^{\mathrm{base}}$",
              r"阈值规则  $Z^{\mathrm{rule}}$",
              r"钉死 $S_0=6000$",
              r"最优  $Z^\star$")
    z_base = float(baselines.get("Z_base", 48052.0466))
    vals = (z_base, baselines["Z^rule"], baselines["Z^6000"], sol.cost)
    cols = (MUTED, C_CH, C_LOAD, C_SOC)
    y = np.arange(4)
    ax.barh(y, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 400, i, f"{v:,.1f} 元", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.20)
    ax.set_xlabel("全天购电费（元）", fontsize=9, color=INK2)
    style_ax(ax, "", "")
    ax.grid(axis="y", visible=False)
    return save_paper(fig, out_dir, "fig06_题1费用对照")


def fig07_s0_basin(outcome, out_dir: Path):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.3), dpi=200, facecolor=SURFACE,
                           layout="constrained")
    grid, costs = outcome.s0_grid, np.asarray(outcome.s0_costs)
    z_star, s0 = outcome.sol.cost, float(outcome.sol.S[0])
    z6000 = outcome.baselines["Z^6000"]
    flat = np.flatnonzero(costs <= z_star + 1.0)
    if len(flat):
        ax.axvspan(grid[flat[0]], grid[flat[-1]], color=C_SOC, alpha=0.12, lw=0,
                   label=f"平坦盆地（宽 {grid[flat[-1]] - grid[flat[0]]:.0f} kWh）")
    ax.plot(grid, costs, color=C_PRICE, lw=2.0)
    ax.plot([s0], [z_star], "o", color=C_SOC, ms=9, markeredgecolor=SURFACE,
            label=rf"最优 $S_0^\star$ = {s0:.0f}")
    ax.plot([6000], [z6000], "s", color=C_LOAD, ms=8, markeredgecolor=SURFACE,
            label=f"附录标定 6000：高出 {z6000 - z_star:.2f} 元")
    ax.set_xlabel(r"$S_0$（kWh）", fontsize=9, color=INK2)
    style_ax(ax, "", "全天购电费 Z（元）")
    legend(ax, loc="upper right", filled=True)
    return save_paper(fig, out_dir, "fig07_S0敏感性")


# ---------------------------------------------------------------------------
# 题 2
# ---------------------------------------------------------------------------

def fig08_year_climate(year, out_dir: Path):
    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.0), sharex=True,
                             dpi=200, facecolor=SURFACE, layout="constrained")
    _pads(fig, hspace=0.20)
    ell_d = year.ell.sum(axis=1)
    r_d = year.r.sum(axis=1)
    roll = lambda x: pd.Series(x).rolling(7, min_periods=1, center=True).mean()
    x = np.arange(len(ell_d))
    marks = []
    for ds, cn in zip(PAPER_DATES, PAPER_DATE_CN):
        j = int(np.where(year.dates.astype("datetime64[D]")
                         == np.datetime64(ds))[0][0])
        marks.append((j, cn))

    ax = axes[0]
    ax.fill_between(x, ell_d, color=C_LOAD, alpha=0.18, lw=0)
    ax.plot(x, roll(ell_d), color=C_LOAD, lw=1.8, label="负载日电量（7 日平滑）")
    style_ax(ax, "", "kWh")
    panel_title(ax, "a")
    ax.set_ylim(0, ell_d.max() * 1.16)
    for k, (j, cn) in enumerate(marks):
        ax.axvline(j, color=BASE, lw=0.9, ls="--", zorder=1)
        ha = "right" if k == len(marks) - 1 else ("left" if k == 0 else "center")
        halo(ax.text(j, ell_d.max() * 1.08, cn, ha=ha, va="center",
                     fontsize=8, color=INK2, clip_on=False))
    legend(ax, loc="lower left", filled=True)

    ax = axes[1]
    ax.fill_between(x, r_d, color=C_PV, alpha=0.18, lw=0)
    ax.plot(x, roll(r_d), color=C_PV, lw=1.8, label="光伏日电量（7 日平滑）")
    style_ax(ax, "", "kWh")
    panel_title(ax, "b")
    ax.set_ylim(0, r_d.max() * 1.16)
    for k, (j, cn) in enumerate(marks):
        ax.axvline(j, color=BASE, lw=0.9, ls="--", zorder=1)
        ha = "right" if k == len(marks) - 1 else ("left" if k == 0 else "center")
        halo(ax.text(j, r_d.max() * 1.08, cn, ha=ha, va="center",
                     fontsize=8, color=INK2, clip_on=False))
    legend(ax, loc="upper left", filled=True)
    month_axis(ax, year.dates, len(x))
    return save_paper(fig, out_dir, "fig08_附件2全年负荷光伏")


def fig10_forecast(year, day1: DayData, out_dir: Path):
    apply_paper_style()
    cfg = ForecastConfig()
    js = np.arange(WARMUP_DAYS, len(year.dates))
    ell_hat = np.stack([forecast_load(cfg, year, int(j), day1.ell) for j in js])
    r_hat = np.stack([forecast_pv(cfg, year, int(j), day1.r) for j in js])
    ell, r = year.ell[js], year.r[js]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), dpi=200,
                             facecolor=SURFACE, layout="constrained")
    _pads(fig, hspace=0.18, wspace=0.14)
    tags = (("a", "b"), ("c", "d"))
    for ax, hat, act, col, name, tag in (
            (axes[0, 0], ell_hat, ell, C_LOAD, "负载", tags[0][0]),
            (axes[0, 1], r_hat, r, C_PV, "光伏", tags[0][1])):
        if name == "光伏":
            mask = (act + hat) > EPS
            hat, act = hat[mask], act[mask]
        else:
            hat, act = hat.ravel(), act.ravel()
        mae = float(np.abs(hat - act).mean())
        lim = float(max(act.max(), hat.max())) * 1.02
        ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1.0, zorder=1)
        ax.scatter(act, hat, s=3.2, lw=0, alpha=0.14, color=col,
                   rasterized=True, zorder=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        style_ax(ax, "", "预报（kWh）")
        panel_title(ax, tag)
        halo(ax.text(0.04, 0.96, f"{name}  MAE {mae:.1f} kWh",
                     transform=ax.transAxes, ha="left", va="top",
                     fontsize=8.5, color=INK2))
        ax.set_xlabel("实际（kWh）", fontsize=9, color=INK2)

    for ax, resid, col, name, tag in (
            (axes[1, 0], ell_hat - ell, C_LOAD, "负载", tags[1][0]),
            (axes[1, 1], r_hat - r, C_PV, "光伏", tags[1][1])):
        groups = [resid[:, 6 * h:6 * (h + 1)].ravel() for h in range(24)]
        bp = ax.boxplot(groups, showfliers=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.1),
                        whiskerprops=dict(color=MUTED, lw=0.7),
                        capprops=dict(color=MUTED, lw=0.7))
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.45, edgecolor=MUTED, lw=0.7)
        ax.axhline(0.0, color=MUTED, lw=0.9, ls="--")
        style_ax(ax, "", "残差（kWh）")
        panel_title(ax, tag)
        ax.set_xticks(range(1, 25, 3), [str(h) for h in range(0, 24, 3)])
        ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
    return save_paper(fig, out_dir, "fig10_题2预报质量-L1P1")


def fig11_daily_cost(q2: pd.DataFrame, year, out_dir: Path):
    apply_paper_style()
    # 拼上暖机期占位：csv 只有报告期。用 NaN 在 1 月留空，灰带表示暖机。
    n = len(year.dates)
    z = np.full(n, np.nan)
    em = np.full(n, np.nan)
    idx = {str(pd.Timestamp(d).date()): i for i, d in enumerate(year.dates)}
    for date, row in q2.iterrows():
        key = str(date.date())
        if key in idx:
            z[idx[key]] = row["z_online"]
            em[idx[key]] = row["emerg_fee"]
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.0), sharex=True,
                             dpi=200, facecolor=SURFACE, layout="constrained")
    _pads(fig, hspace=0.20)
    x = np.arange(n)
    ax = axes[0]
    ax.axvspan(0, WARMUP_DAYS, color=GRID, alpha=0.7, lw=0)
    ax.plot(x, z, color=C_PRICE, lw=1.15)
    z_rep = z[WARMUP_DAYS:]
    ax.axhline(np.nanmean(z_rep), color=MUTED, lw=1, ls="--")
    style_ax(ax, "", "日费用（元）")
    panel_title(ax, "a")
    legend(ax, [
        Line2D([], [], color=C_PRICE, lw=1.2, label="日费用"),
        Patch(facecolor=GRID, alpha=0.7, lw=0, label="暖机期"),
        Line2D([], [], color=MUTED, ls="--", lw=1,
               label=f"报告期均值 {np.nanmean(z_rep):,.0f} 元/日"),
    ], loc="upper right", filled=True)

    ax = axes[1]
    ax.axvspan(0, WARMUP_DAYS, color=GRID, alpha=0.7, lw=0)
    ax.bar(x, np.nan_to_num(em), width=1.0, color=C_EMERG, lw=0)
    style_ax(ax, "", "紧急费（元）")
    panel_title(ax, "b")
    month_axis(ax, year.dates, n)
    return save_paper(fig, out_dir, "fig11_题2逐日费用与紧急费")


def fig12_solstice(year, day1: DayData, g2: np.ndarray, dates2: pd.DatetimeIndex,
                   bat2: pd.DataFrame, out_dir: Path):
    apply_paper_style()
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 10.0), sharex="col",
                             dpi=200, facecolor=SURFACE, layout="constrained")
    _pads(fig, hspace=0.24, wspace=0.18)
    date_to_row = {str(d.date()): i for i, d in enumerate(dates2)}
    hours4 = np.arange(0, 25, 4)
    tags = "abcd"
    y_left = 0.0
    handles = labels = None
    for i, (ds, cn) in enumerate(zip(PAPER_DATES, PAPER_DATE_CN)):
        j = int(np.where(year.dates.astype("datetime64[D]")
                         == np.datetime64(ds))[0][0])
        ax = axes[i, 0]
        price_bands(ax, day1.p)
        ax.plot(HOURS, year.ell[j] * 6, color=C_LOAD, lw=1.5, label="负载")
        ax.plot(HOURS, year.r[j] * 6, color=C_PV, lw=1.5, label="光伏")
        row = date_to_row[ds]
        ax.plot(HOURS, g2[row] * 6, color=C_BUY, lw=1.8, label="计划购电")
        y_left = max(y_left, ax.get_ylim()[1])
        style_ax(ax, "", "kW")
        panel_title(ax, tags[i], cn)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax = axes[i, 1]
        s = bat2.loc[pd.Timestamp(ds), "s4h"]
        ax.plot(hours4, s, color=C_SOC, lw=2.0, marker="o", ms=4,
                drawstyle="steps-post")
        ax.axhline(S_MIN, color=MUTED, lw=0.8, ls="--")
        ax.axhline(S_MAX, color=MUTED, lw=0.8, ls="--")
        ax.set_ylim(0, S_MAX * 1.12)
        style_ax(ax, "", "kWh")
        ax.set_title("")
        if i == 3:
            ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
            axes[i, 0].set_xlabel("时刻（h）", fontsize=9, color=INK2)
            axes[i, 0].set_xticks(range(0, 25, 4))
            ax.set_xticks(hours4)
    for i in range(4):
        axes[i, 0].set_ylim(0, y_left * 1.04)
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=9, labelcolor=INK2)
    return save_paper(fig, out_dir, "fig12_题2二分二至剖面")


def fig13_ladder_attr(out_dir: Path):
    """基准阶梯用摘要数字；归因用 v6 vs nocap 对照。"""
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=200,
                             facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"width_ratios": (1.05, 1.15)})
    _pads(fig, wspace=0.22)
    ax = axes[0]
    names = ("完美信息  PI", "事后补救  ex-post",
             "交付  online", "确定性等价  CE")
    vals = np.array([12223092.57, 13463630.76, 13745816.13, 14254554.72]) / 1e4
    cols = (C_SOC, C_BUY, C_PRICE, MUTED)
    y = np.arange(4)
    ax.barh(y, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 4, i, f"{v:,.1f} 万", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, vals.max() * 1.22)
    ax.set_xlabel("报告期费用（万元）", fontsize=9, color=INK2)
    style_ax(ax, "", "")
    ax.grid(axis="y", visible=False)
    panel_title(ax, "a")

    ax = axes[1]
    labels = ("①功率\n强制", "②储能\n耗尽", "③a囤电", "③b配给", "④低水\n位带")
    v6 = np.array([8.71, 18.99, 0.00, 49.06, 23.23])
    nocap = np.array([3.50, 17.80, 41.70, 19.31, 17.69])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, nocap, 0.34, color="#d9c4b0", label="无上界（v5）")
    ax.bar(x + 0.18, v6, 0.34, color=C_PRICE, label="交付（v6 上界）")
    # 交付③a = 0：柱高看不见，标注放在无上界柱顶上方，避免被读成「无上界是 0」
    halo(ax.text(2 + 0.18, nocap[2] + 1.8, "交付 0", ha="center", va="bottom",
                 fontsize=8.5, color=C_PRICE, fontweight="medium"))
    ax.set_xticks(x, labels)
    ax.set_ylabel("紧急电量占比（%）", fontsize=9, color=INK2)
    ax.set_ylim(0, 56)
    style_ax(ax, "", "")
    panel_title(ax, "b")
    legend(ax, loc="upper left", filled=True)
    return save_paper(fig, out_dir, "fig13_题2费用阶梯与紧急归因")


# ---------------------------------------------------------------------------
# 题 3
# ---------------------------------------------------------------------------

def fig15_fusion(year, fc, out_dir: Path):
    """提前期 RMSE（白天目标）+ 反方差权重。夜间目标会把长提前伪造成「很准」。"""
    apply_paper_style()
    r_hour = hour_powers(year.r)          # kW
    phi = fc.phi
    tau = np.array(VINTAGE_HOURS)
    err_fc = [[] for _ in range(24)]
    err_pe = [[] for _ in range(24)]
    for j in range(WARMUP_DAYS, len(year.dates)):
        pers = r_hour[max(0, j - 7):j].mean(axis=0)
        for m, t0 in enumerate(tau):
            for h in range(1, 25):
                tgt = t0 + h
                if tgt > 24:
                    continue
                actual = r_hour[j, tgt - 1]
                if actual < 80:          # 夜间 / 无光：不进提前期曲线
                    continue
                err_fc[h - 1].append(phi[j, m, h - 1] - actual)
                err_pe[h - 1].append(pers[tgt - 1] - actual)
    h = np.arange(1, 25)
    rmse_fc = np.array([np.sqrt(np.mean(np.square(e))) if e else np.nan
                        for e in err_fc])
    rmse_pe = np.array([np.sqrt(np.mean(np.square(e))) if e else np.nan
                        for e in err_pe])
    var_fc, var_pe = rmse_fc ** 2, rmse_pe ** 2
    phi_w = var_pe / (var_pe + var_fc)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True,
                             dpi=200, facecolor=SURFACE, layout="constrained")
    _pads(fig, hspace=0.18)
    ax = axes[0]
    ax.plot(h, rmse_fc, color=C_PV, lw=2.1, marker="o", ms=4,
            label="官方预报（附件 3）")
    ax.plot(h, rmse_pe, color=C_CH, lw=2.1, marker="s", ms=4,
            label="持续法（近 7 日）")
    # 交叉点
    diff = rmse_fc - rmse_pe
    cross = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
    if len(cross):
        i = int(cross[0])
        ax.axvline(h[i] + 0.5, color=MUTED, lw=0.9, ls="--")
        halo(ax.text(h[i] + 0.6, np.nanmax(rmse_fc) * 0.92, "交叉 ≈ 9–10 h",
                     fontsize=8.5, color=INK2))
    style_ax(ax, "", "RMSE（kW）")
    panel_title(ax, "a")
    legend(ax, loc="upper left", filled=True)

    ax = axes[1]
    ax.plot(h, phi_w, color=C_PRICE, lw=2.1, marker="o", ms=4)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
    finite = np.where(np.isfinite(phi_w))[0]
    x_lab = float(h[finite[0]]) if len(finite) else 1.0
    halo(ax.text(x_lab + 0.15, 0.54, "等权 0.5", fontsize=8, color=MUTED,
                 ha="left", va="bottom"))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("预报提前期（h）", fontsize=9, color=INK2)
    style_ax(ax, "", r"官方权重 $\varphi$")
    panel_title(ax, "b")
    return save_paper(fig, out_dir, "fig15_题3白天预报融合")


def fig16_revision(year, day1, dates3, g3, a3, bat3, out_dir: Path):
    apply_paper_style()
    dev = np.abs(a3[:, 36:] - g3[:, 36:]).sum(axis=1)
    i = int(np.argmax(dev))
    date = dates3[i]
    j = int(np.where(year.dates.astype("datetime64[D]")
                     == np.datetime64(str(date.date())))[0][0])
    g_kw, a_kw = g3[i] * 6, a3[i] * 6
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.4), sharex=True,
                             dpi=200, facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"height_ratios": (3.0, 2.0, 2.2)})
    _pads(fig, hspace=0.20)
    ax = axes[0]
    price_bands(ax, day1.p)
    ax.axvspan(0, 6, color=GRID, alpha=0.55, lw=0)
    for h in (6, 12, 18):
        ax.axvline(h, color=BASE, lw=0.9, ls="--")
    ax.plot(HOURS, g_kw, color=C_BUY, lw=2.0, drawstyle="steps-mid",
            label="计划 $g$")
    ax.plot(HOURS, a_kw, color="#5b3ea8", lw=2.0, ls="--",
            drawstyle="steps-mid", label="最终注入 $a$")
    style_ax(ax, "", "kW")
    panel_title(ax, "a", str(date.date()))
    legend(ax, [
        Line2D([], [], color=C_BUY, lw=2),
        Line2D([], [], color="#5b3ea8", lw=2, ls="--"),
        Patch(facecolor=GRID, alpha=0.55, lw=0),
    ], ["计划购电 $g_t$", "最终注入 $a_t$", r"$\mathcal{B}_0$ 不可修订"],
        loc="upper left", filled=True)

    ax = axes[1]
    delta = a_kw - g_kw
    w = (1 / 6) * 0.9
    ax.bar(HOURS, np.where(delta > 0, delta, 0.0), w, color=C_LOAD,
           label=r"上调 $\delta^+$")
    ax.bar(HOURS, np.where(delta < 0, delta, 0.0), w, color=C_PRICE,
           label=r"下调 $\delta^-$")
    ax.axhline(0, color=BASE, lw=0.8)
    for h in (6, 12, 18):
        ax.axvline(h, color=BASE, lw=0.9, ls="--")
    style_ax(ax, "", "kW")
    panel_title(ax, "b")
    legend(ax, loc="upper right", filled=True)

    ax = axes[2]
    s = bat3.loc[pd.Timestamp(date), "s4h"]
    ax.plot(np.arange(0, 25, 4), s, color=C_SOC, lw=2.0, marker="o", ms=4,
            drawstyle="steps-post")
    ax.axhline(S_MIN, color=MUTED, lw=0.8, ls="--")
    ax.axhline(S_MAX, color=MUTED, lw=0.8, ls="--")
    halo(ax.text(0.3, S_MIN + 160, "1200", fontsize=8, color=MUTED))
    halo(ax.text(0.3, S_MAX - 280, "10800", fontsize=8, color=MUTED))
    ax.set_ylim(0, S_MAX * 1.12)
    style_ax(ax, "", "kWh")
    panel_title(ax, "c")
    day_axis(ax)
    return save_paper(fig, out_dir, "fig16_题3修订演示-最大偏差日")


def fig17_q2q3(q2: pd.DataFrame, q3: pd.DataFrame, year, out_dir: Path):
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), dpi=200,
                             facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"width_ratios": (1.0, 1.25)})
    _pads(fig, wspace=0.22)
    ax = axes[0]
    q2_emerg = float(q2["emerg_fee"].sum())
    q2_z = float(q2["z_online"].sum())
    q2_plan = q2_z - q2_emerg
    q3_base = float(q3["base_fee"].sum())
    q3_dev = float(q3["dev_fee"].sum())
    q3_emerg = float(q3["emerg_fee"].sum())
    s2 = np.array([q2_plan, 0.0, q2_emerg]) / 1e4
    s3 = np.array([q3_base, q3_dev, q3_emerg]) / 1e4
    labels = ("基础购电", "偏差费", "紧急费")
    cols = (C_BUY, C_CH, C_EMERG)
    x = np.array([0, 1])
    b2 = b3 = 0.0
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(0, s2[i], 0.52, bottom=b2, color=c, label=lab)
        ax.bar(1, s3[i], 0.52, bottom=b3, color=c)
        b2 += s2[i]
        b3 += s3[i]
    ax.set_xticks(x, ("问题 2", "问题 3"))
    ax.set_xlim(-0.55, 2.15)
    ax.set_ylabel("报告期费用（万元）", fontsize=9, color=INK2)
    ax.text(0, b2 + 12, f"{b2:,.1f}", ha="center", fontsize=9, color=INK2)
    ax.text(1, b3 + 12, f"{b3:,.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, max(b2, b3) * 1.16)
    style_ax(ax, "", "")
    ax.grid(axis="x", visible=False)
    panel_title(ax, "a")
    legend(ax, loc="center right", filled=True)

    ax = axes[1]
    aligned = q3["z_online"].to_numpy() - q2.reindex(q3.index)["z_online"].to_numpy()
    x = np.arange(len(aligned))
    ax.bar(x, np.where(aligned > 0, aligned, 0.0), width=1.0, color=C_LOAD, lw=0)
    ax.bar(x, np.where(aligned < 0, aligned, 0.0), width=1.0, color=C_PRICE, lw=0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(np.nanmean(aligned), color=MUTED, lw=1.2, ls="--")
    ax.set_ylabel("元 / 日", fontsize=9, color=INK2)
    style_ax(ax, "", "")
    panel_title(ax, "b")
    month_axis(ax, year.dates[WARMUP_DAYS:WARMUP_DAYS + len(aligned)],
               len(aligned))
    legend(ax, [
        Patch(facecolor=C_LOAD, lw=0, label="题 3 更贵"),
        Patch(facecolor=C_PRICE, lw=0, label="题 3 更省"),
        Line2D([], [], color=MUTED, ls="--",
               label=f"均值 {np.nanmean(aligned):+,.0f} 元/日"),
    ], loc="upper left", filled=True)
    return save_paper(fig, out_dir, "fig17_题2与题3费用对照")


# ---------------------------------------------------------------------------
# 题 3 修订阶梯（fig18）与 题 4（fig19 / fig21）：CSV 驱动，不依赖滚动轨迹
# ---------------------------------------------------------------------------

# 修订阶梯四臂（v7 计划器；前三档为受控回测臂，末档 = 交付臂）
Q3_LADDER = (
    ("0", ROOT / "results" / "q3_v7-ladder" / "F1-L1-LU1-S-b-E0"),
    ("0-6", ROOT / "results" / "q3_v7-ladder" / "F1-L1-LU1-S-b-E0-6"),
    ("0-6-12", ROOT / "results" / "q3_v7-ladder" / "F1-L1-LU1-S-b-E0-6-12"),
    ("0-6-12-18", Q3_DIR),
)
Q4_42_DIR = ROOT / "results" / "q4" / "4-2"


def _zon42(arm: str) -> float:
    """题 4 计划线某臂全年 Z_online（q4-2_details.csv 逐日求和）。"""
    return float(pd.read_csv(Q4_42_DIR / arm / "q4-2_details.csv")
                 .z_online.sum())


def fig18_q3_ladder(out_dir: Path):
    """修订阶梯四档，口径同 experiments/gap/collect.py：各臂 Z = 逐日求和。"""
    rows = []
    for epochs, arm in Q3_LADDER:
        det = pd.read_csv(arm / "q3_details.csv")
        rows.append({"epochs": epochs,
                     "z_online": float(det.z_online.sum()),
                     "z_pi": float(det.z_pi.sum())})
    # EpochsLadder 走 save()（不做文字碰撞检查），issues 恒空
    return EpochsLadder(rows).export(out_dir,
                                     name="fig18_题3修订阶梯.png"), []


def fig21_q4_price_ladder(out_dir: Path):
    """题 4 计划线三笔消融账（相对主交付 PR1-V2-r，读法 R 全年 334 天）。"""
    main = _zon42("PR1-V2-r")
    items = [("不建模价格\n（PR0：附件1 常数价）", _zon42("PR0-V2-r") - main),
             ("常数终端价值\n（V0：ν 恒 0.472）", _zon42("PR1-V0-r") - main),
             ("读法 D\n（0:00 已知全天实现价）", _zon42("PR1-V2-d") - main)]
    y = np.arange(len(items))[::-1]
    vals = np.array([v for _, v in items])
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200, facecolor=SURFACE)
    colors = [C_EMERG if v > 0 else C_BUY for v in vals]
    ax.barh(y, vals, height=0.55, color=colors, alpha=0.9, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + (2600 if v > 0 else -2600), yi,
                f"{v:+,.0f} 元", va="center",
                ha="left" if v > 0 else "right", fontsize=9.5, color=INK)
    ax.axvline(0.0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(y, [k for k, _ in items], fontsize=9.5)
    ax.set_xlim(vals.min() * 1.45, vals.max() * 1.45)
    style_ax(ax, "相对主交付的全年差额（元）：正 = 更贵，负 = 更省", "消融臂")
    fig.tight_layout()
    return save_paper(fig, out_dir, "fig21_题4价格信息阶梯")


def fig19_q4_weekday_step(out_dir: Path):
    """题 4 计划线星期台阶：日末 S144 按次日公历星期分组（与 report/q4.py 一致）。"""
    df = pd.read_csv(Q4_42_DIR / "PR1-V2-r" / "q4-2_details.csv",
                     usecols=["date", "S144"])
    nxt = pd.to_datetime(df.date) + pd.Timedelta(days=1)
    key = nxt.dt.weekday.map({4: "次日周五", 5: "次日周六",
                              0: "次日周一"}).fillna("其他")
    order = ["次日周五", "次日周六", "其他", "次日周一"]
    groups = [df.loc[key == g, "S144"].to_numpy() for g in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200, facecolor=SURFACE)
    bp = ax.boxplot(groups, tick_labels=order, widths=0.5, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", ms=5,
                                                   mfc=INK, mec=SURFACE),
                    medianprops=dict(color=INK2, lw=1.2), zorder=3,
                    flierprops=dict(ms=3, mfc=INK2, mec=INK2, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set(facecolor=C_SOC, alpha=0.30, edgecolor=C_SOC)
    for wh in bp["whiskers"] + bp["caps"]:
        wh.set(color=INK2, lw=1.0)
    for i, g in enumerate(groups, start=1):   # 组均值标注（台阶读数）
        ax.annotate(f"{g.mean():,.0f}", xy=(i, g.mean()), xytext=(14, 2),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(float(df.S144.mean()), color=GRID, lw=1.0, ls="--", zorder=1)
    style_ax(ax, "计划线日末水位按次日星期分组：次日周五主动清仓",
             "日末 SOC（kWh）")
    ax.text(0.99, 0.02, f"虚线 = 全年均值 {df.S144.mean():,.0f} kWh",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save_paper(fig, out_dir, "fig19_题4日末水位星期台阶")


def fig_a2_revision_heat(dates3, g3, a3, year, out_dir: Path):
    """题 3 修订指纹。图面不放标题（题注/文件名承担说明）。"""
    apply_paper_style()
    diff = a3 - g3
    code = np.zeros_like(diff, dtype=np.uint8)
    code[diff > EPS] = 1
    code[diff < -EPS] = 2
    n = len(code)
    idle = "#ffffff"
    b0 = "#ebe8e1"
    fig, ax = plt.subplots(figsize=(9.8, 7.0), dpi=200, facecolor=SURFACE,
                           layout="constrained")
    ax.set_facecolor(b0)
    ax.imshow(code[:, 36:], aspect="auto", interpolation="nearest",
              cmap=ListedColormap([idle, C_LOAD, C_PRICE]), vmin=0, vmax=2,
              extent=(6.0, 24.0, float(n), 0.0))
    ax.axvspan(0.0, 6.0, facecolor=b0, edgecolor="none", zorder=0)
    ax.text(3.0, n / 2, r"$\mathcal{B}_0$  不可修订",
            ha="center", va="center", rotation=90, fontsize=9.5, color=MUTED)
    for h in (6.0, 12.0, 18.0):
        ax.axvline(h, color=BASE, lw=0.8, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks((0, 6, 12, 18, 24))
    ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
    month_axis(ax, year.dates[WARMUP_DAYS:WARMUP_DAYS + n], n, on_y=True)
    ax.grid(False)
    ax.set_title("")
    ax.set_ylabel("")
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    legend(ax, [
        Patch(facecolor=C_LOAD, lw=0),
        Patch(facecolor=C_PRICE, lw=0),
        Patch(facecolor=idle, edgecolor=BASE, linewidth=0.6),
    ], [r"上调 $\delta^+$", r"下调 $\delta^-$", "未改"],
        ncol=3, loc="outside lower center")
    stem = "figA2_题3修订指纹-红上调蓝下调-B0法定空白"
    return save_paper(fig, out_dir, stem)


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def _report(path: Path, issues: list[str]) -> None:
    flag = "OK" if not issues else f"{len(issues)} issues"
    print(f"  [{flag}] {path.name}")
    for it in issues[:12]:
        print(f"      {it}")
    if len(issues) > 12:
        print(f"      … {len(issues) - 12} more")


def render_all(out_dir: Path | None = None) -> Path:
    """绘制正文 + 附录图到 results/paper/。"""
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    from nmmc.report.plot import schematic

    day1 = load_day1()
    year = load_year()
    fc = load_forecast()
    q1 = run_q1(day1)
    # 无电池基准不在 baselines 里，补上
    z_base = float(day1.p @ np.maximum(day1.ell - day1.r, 0.0))
    q1.baselines["Z_base"] = z_base

    q2x, q3x = Q2_DIR / "result2.xlsx", Q3_DIR / "result3.xlsx"
    dates2, g2 = load_plan(q2x, "计划购电量")
    dates3, g3 = load_plan(q3x, "计划购电量")
    _, a3 = load_plan(q3x, "调整购电量")
    bat2, bat3 = load_battery(q2x), load_battery(q3x)
    daily2 = load_daily(Q2_DIR / "q2_details.csv")
    daily3 = load_daily(Q3_DIR / "q3_details.csv")

    steps = [
        ("fig01_bus", lambda: schematic.fig01_bus(out_dir)),
        ("fig02_framework", lambda: schematic.fig02_framework(out_dir)),
        ("fig03_data", lambda: fig03_data_overview(day1, out_dir)),
        ("fig04_schedule", lambda: fig04_schedule(day1, q1.sol, out_dir)),
        ("fig05_balance", lambda: fig05_energy_balance(day1, q1.sol, out_dir)),
        ("fig06_cost", lambda: fig06_cost_compare(q1.sol, q1.baselines, out_dir)),
        ("fig07_s0", lambda: fig07_s0_basin(q1, out_dir)),
        ("fig08_climate", lambda: fig08_year_climate(year, out_dir)),
        ("fig09_arch", lambda: schematic.fig09_q2_arch(out_dir)),
        ("fig10_fcst", lambda: fig10_forecast(year, day1, out_dir)),
        ("fig11_cost", lambda: fig11_daily_cost(daily2, year, out_dir)),
        ("fig12_solstice", lambda: fig12_solstice(
            year, day1, g2, dates2, bat2, out_dir)),
        ("fig13_attr", lambda: fig13_ladder_attr(out_dir)),
        ("fig14_commit", lambda: schematic.fig14_q3_commit(out_dir)),
        ("fig15_fusion", lambda: fig15_fusion(year, fc, out_dir)),
        ("fig16_rev", lambda: fig16_revision(
            year, day1, dates3, g3, a3, bat3, out_dir)),
        ("fig17_q2q3", lambda: fig17_q2q3(daily2, daily3, year, out_dir)),
        ("fig18_ladder", lambda: fig18_q3_ladder(out_dir)),
        ("fig19_weekday", lambda: fig19_q4_weekday_step(out_dir)),
        ("fig21_ablate", lambda: fig21_q4_price_ladder(out_dir)),
        ("figA2_heat", lambda: fig_a2_revision_heat(
            dates3, g3, a3, year, out_dir)),
    ]
    print(f"写出 {len(steps)} 张 → {out_dir}")
    keep: set[str] = set()
    for name, fn in steps:
        try:
            path, issues = fn()
            keep.add(path.stem)
            _report(path, issues)
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            raise
        finally:
            plt.close("all")
    for p in out_dir.glob("fig*.*"):
        if p.suffix.lower() in {".png", ".pdf"} and p.stem not in keep:
            p.unlink()
    return out_dir
