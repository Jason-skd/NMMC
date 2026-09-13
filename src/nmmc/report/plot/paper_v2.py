"""正文用论文图 v2：纯白底、栏宽示意图、去冗余色。

写出 results/paper-v2/。不改 style.SURFACE，避免动到既有诊断图。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from matplotlib.text import Text

from nmmc.data import ROOT, T
from nmmc.data.day import DayData, load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.price import load_price
from nmmc.data.year import load_year
from nmmc.model.forecast3 import hour_powers
from nmmc.model.lp import S_MAX, S_MIN
from nmmc.model.q3 import VINTAGE_HOURS
from nmmc.report.plot import paper as src
from nmmc.report.plot.style import (BAND_LABELS, BASE, C_BUY, C_CH, C_DIS,
                                    C_DUAL, C_EMERG, C_LOAD, C_PRICE, C_PV,
                                    C_SOC, ENDS, GRID, HOURS, INK, INK2, MUTED,
                                    apply_paper_style, check_text_overlap,
                                    day_axis, month_axis, panel_title,
                                    price_bands, runs, style_ax)
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.run.q1 import run_q1

WHITE = "#ffffff"
EPS = 1e-6
HALO = [patheffects.withStroke(linewidth=3.4, foreground=WHITE)]
# 四档价签：同一橙–红热阶，不借用实体槽
PENALTY = ("#f2d4b8", "#e39a52", "#c45c18", "#9b2226")
Q4_DIR = ROOT / "results" / "q4" / "4-2" / "PR1-V2-r"
Q4_DIR_43 = ROOT / "results" / "q4" / "4-3" / "PR1-V2-r"
LADDER_Z = (
    ("仅 0:00", 13_633_398.76),
    ("+6:00", 13_345_217.23),
    ("+12:00", 13_176_494.13),
    ("全开", 13_023_147.06),
)
# 计划线 334 天，与 results/q4/4-2/*/q4-2_summary.md 一致
Q4_LADDER = (
    ("不建模价格", 14_485_987.96),
    ("常数终端价值", 14_466_703.19),
    ("主交付（读法 R）", 14_447_039.68),
    ("知当日价（读法 D）", 14_402_278.06),
)
Q4_FIGS = ("fig19", "fig20", "fig21", "fig22")


def apply_v2() -> None:
    apply_paper_style()
    plt.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })


def pads(fig, *, hspace: float = 0.16, wspace: float = 0.12) -> None:
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(h_pad=0.05, w_pad=0.05, hspace=hspace, wspace=wspace)


def face(ax, ylabel: str = "") -> None:
    style_ax(ax, "", ylabel)
    ax.set_facecolor(WHITE)


def halo(text: Text) -> Text:
    text.set_path_effects(HALO)
    return text


def legend_w(ax, handles=None, labels=None, *, loc: str = "upper left",
             ncol: int = 1) -> None:
    kw = dict(frameon=True, fancybox=False, edgecolor="none",
              facecolor=WHITE, framealpha=0.94, borderpad=0.45,
              fontsize=9, labelcolor=INK2)
    target = ax.figure if isinstance(loc, str) and loc.startswith("outside") else ax
    if handles is None:
        target.legend(ncol=ncol, loc=loc, **kw)
    elif labels is None:
        target.legend(handles=handles, ncol=ncol, loc=loc, **kw)
    else:
        target.legend(handles, labels, ncol=ncol, loc=loc, **kw)


def save_v2(fig, out_dir: Path, stem: str) -> tuple[Path, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(WHITE)
    issues = check_text_overlap(fig)
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    for path in (png, pdf):
        fig.savefig(path, facecolor=WHITE, edgecolor="none",
                    bbox_inches="tight", pad_inches=0.12,
                    dpi=300 if path.suffix == ".png" else None)
    return png, issues


def new_plot(w: float, h: float, **kw):
    apply_v2()
    fig, ax = plt.subplots(figsize=(w, h), dpi=200, facecolor=WHITE,
                           layout="constrained", **kw)
    fig.set_facecolor(WHITE)
    return fig, ax


def new_rows(n: int, w: float, h: float, *, ratios=None, sharex=True):
    apply_v2()
    fig, axes = plt.subplots(n, 1, figsize=(w, h), sharex=sharex, dpi=200,
                             facecolor=WHITE, layout="constrained",
                             gridspec_kw=({} if ratios is None
                                          else {"height_ratios": ratios}))
    fig.set_facecolor(WHITE)
    pads(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# 示意图（栏宽 160 mm ≈ 6.3 in，字号按印刷 9 pt）
# ---------------------------------------------------------------------------

def _schematic(w_in: float, h_in: float):
    apply_v2()
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200, facecolor=WHITE)
    ax.set_xlim(0, w_in * 25.4)
    ax.set_ylim(0, h_in * 25.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.98)
    fig.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    return fig, ax


def _box(ax, x, y, w, h, title, sub="", *, fs=9.0, fs_sub=7.5):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=WHITE, edgecolor=INK,
                           linewidth=0.9, joinstyle="miter"))
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="medium")
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=fs_sub, color=INK2)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="medium")


def _arrow(ax, x1, y1, x2, y2, *, color=INK, lw=1.05):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8.5,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0.6))


def fig01_bus(out_dir: Path):
    fig, ax = _schematic(6.30, 3.55)
    ax.text(80.0, 85.0,
            r"$g_t+r_t+d_t=\ell_t+c_t+w_t$",
            ha="center", va="center", fontsize=11, color=INK)
    ax.text(80.0, 78.2,
            r"$S_t=S_{t-1}+\eta c_t-d_t/\eta,\ \eta=0.9$",
            ha="center", va="center", fontsize=11, color=INK)
    names = ("外网", "光伏", "小区负载", "弃电")
    xs = (8.0, 46.0, 84.0, 122.0)
    bw, bh, yb = 30.0, 11.0, 60.5
    for x, name in zip(xs, names):
        _box(ax, x, yb, bw, bh, name, fs=9)
    y_bus = 48.0
    ax.plot([6.0, 154.0], [y_bus, y_bus], color=INK, lw=2.6, solid_capstyle="butt")
    ax.text(155.5, y_bus, "母线", fontsize=9, color=INK, va="center")
    feeders = (
        (xs[0] + bw / 2, C_PRICE, r"$g_t$", True),
        (xs[1] + bw / 2, C_PV, r"$r_t$", True),
        (xs[2] + bw / 2, C_LOAD, r"$\ell_t$", False),
        (xs[3] + bw / 2, MUTED, r"$w_t$", False),
    )
    for x, col, lab, down in feeders:
        if down:
            _arrow(ax, x, yb, x, y_bus + 1.6, color=col, lw=1.15)
            ax.text(x - 2.2, (yb + y_bus) / 2, lab, ha="right", va="center",
                    fontsize=9, color=col)
        else:
            _arrow(ax, x, y_bus + 1.6, x, yb, color=col, lw=1.15)
            ax.text(x + 2.2, (yb + y_bus) / 2, lab, ha="left", va="center",
                    fontsize=9, color=col)
        ax.add_patch(Circle((x, y_bus), 1.35, facecolor=WHITE,
                            edgecolor=INK, linewidth=0.8, zorder=3))
    mid, iw, ih = 80.0, 32.0, 10.5
    inv_y, bat_y = 30.5, 13.0
    ax.add_patch(Circle((mid, y_bus), 1.35, facecolor=WHITE,
                        edgecolor=INK, linewidth=0.8, zorder=3))
    _box(ax, mid - iw / 2, inv_y, iw, ih, r"逆变器  $\eta=0.9$", fs=8.5)
    _box(ax, mid - iw / 2, bat_y, iw, ih, "电池  1200–10800 kWh", fs=8.0)
    xc, xd = mid - 4.2, mid + 4.2
    _arrow(ax, xc, y_bus - 1.8, xc, inv_y + ih, color=C_CH, lw=1.1)
    _arrow(ax, xd, inv_y + ih, xd, y_bus - 1.8, color=C_DIS, lw=1.1)
    ax.text(xc - 2.0, (y_bus + inv_y + ih) / 2, r"$c_t$", ha="right",
            va="center", fontsize=8.5, color=C_CH)
    ax.text(xd + 2.0, (y_bus + inv_y + ih) / 2, r"$d_t$", ha="left",
            va="center", fontsize=8.5, color=C_DIS)
    _arrow(ax, xc, inv_y, xc, bat_y + ih, color=C_SOC, lw=1.05)
    _arrow(ax, xd, bat_y + ih, xd, inv_y, color=C_SOC, lw=1.05)
    ax.text(xc - 2.0, (inv_y + bat_y + ih) / 2, r"$\eta c_t$", ha="right",
            va="center", fontsize=8, color=C_SOC)
    ax.text(xd + 2.0, (inv_y + bat_y + ih) / 2, r"$d_t/\eta$", ha="left",
            va="center", fontsize=8, color=C_SOC)
    return save_v2(fig, out_dir, "fig01_微网母线与电量流转")


def fig02_framework(out_dir: Path):
    fig, ax = _schematic(6.30, 4.20)
    x0, x1 = 4.0, 156.0
    y_top, y_head, y_bot = 103.0, 94.5, 4.0
    for y, lw in ((y_top, 1.35), (y_head, 0.55), (y_bot, 1.35)):
        ax.plot([x0, x1], [y, y], color=INK, lw=lw, solid_capstyle="butt")
    heads = ("信息结构", "决策节奏", "结算罚则")
    col_x = (38.0, 80.0, 122.0)
    for x, h in zip(col_x, heads):
        ax.text(x, (y_top + y_head) / 2, h, ha="center", va="center",
                fontsize=10, color=INK, fontweight="medium")
    rows = (
        ("题 1", (
            "典型日已知，无随机",
            r"0:00 一次 LP，循环 $S_{144}=S_0$",
            r"$\min\sum p\,g$；弃电免费",
        )),
        ("题 2", (
            "读法 C：仅历史 + 日历",
            r"0:00 锁 $g$；日内随机 MPC",
            r"$Z=\sum pg+5\sum pu$；$\nu^{\mathrm{end}}$",
        )),
        ("题 3", (
            "＋附件 3，一日四次",
            r"$6/12/18$ 各锁一块 $a$",
            r"$Z=\sum[pa+\frac{1}{2}p|a-g|]$",
        )),
        ("题 4", (
            "读法 R：电价同为实现值",
            r"三元情景；$\nu^{\mathrm{end}}$ 逐日",
            r"同题 2/3，按实现价结算",
        )),
    )
    y_rows = (82.5, 62.0, 41.5, 21.0)
    for i, ((name, cells), y) in enumerate(zip(rows, y_rows)):
        if i:
            ax.plot([x0, x1], [y + 10.0, y + 10.0], color="#c8c7c0",
                    lw=0.4, solid_capstyle="butt")
        ax.text(16.0, y, name, ha="center", va="center",
                fontsize=10.5, color=INK, fontweight="medium")
        for x, line in zip(col_x, cells):
            ax.text(x, y, line, ha="center", va="center",
                    fontsize=8.6, color=INK2)
    return save_v2(fig, out_dir, "fig02_三问递进-信息决策罚则")


def fig09_arch(out_dir: Path):
    fig, ax = _schematic(6.30, 3.70)
    x, w, bh = 16.0, 128.0, 13.2
    gap = 5.8
    y_set = 5.5
    y_exec = y_set + bh + gap
    eh = 25.5
    y_plan = y_exec + eh + gap
    y_fcst = y_plan + bh + gap
    _box(ax, x, y_fcst, w, bh, "预报层", r"L1 $\times$ P1 $\times$ R2")
    _box(ax, x, y_plan, w, bh, "规划层", r"两阶段 SAA，锁 $g$")
    ax.add_patch(Rectangle((x, y_exec), w, eh, facecolor=WHITE, edgecolor=INK,
                           linewidth=0.9))
    ax.text(x + w / 2, y_exec + eh - 3.4, "执行层 · 在线随机 MPC",
            ha="center", va="center", fontsize=9, color=INK, fontweight="medium")
    iw, ih = 50.0, 13.0
    _box(ax, x + 9.0, y_exec + 3.0, iw, ih, "当步 $t$", "真值 · 水位", fs=8.5)
    _box(ax, x + w - 9.0 - iw, y_exec + 3.0, iw, ih, "未来时域",
         "12 个平行情景", fs=8.5)
    y_mid = y_exec + 3.0 + ih / 2
    _arrow(ax, x + 9.0 + iw + 1.2, y_mid, x + w - 9.0 - iw - 1.4, y_mid)
    _box(ax, x, y_set, w, bh, "结算层", r"$\sum pg+5\sum pu$")
    cx = x + w / 2
    _arrow(ax, cx, y_fcst - 0.3, cx, y_plan + bh + 0.5)
    _arrow(ax, cx, y_plan - 0.3, cx, y_exec + eh + 0.5)
    _arrow(ax, cx, y_exec - 0.3, cx, y_set + bh + 0.5)
    return save_v2(fig, out_dir, "fig09_题2四层架构与随机MPC")


def fig14_commit(out_dir: Path):
    fig, ax = _schematic(6.30, 2.55)
    x0, span, yb, hb = 10.0, 140.0, 38.0, 8.2
    bw, strip_h = span / 4.0, 2.6
    locked = "#eceae4"
    titles = (r"$\mathcal{B}_0$  0–6 时", r"$\mathcal{B}_1$  6–12 时",
              r"$\mathcal{B}_2$  12–18 时", r"$\mathcal{B}_3$  18–24 时")
    notes = (r"$a\equiv g$，不可修订", "光伏预报", "短前置", "负载观测")
    for i in range(4):
        x = x0 + bw * i
        ax.add_patch(Rectangle((x, yb), bw, hb,
                               facecolor=(locked if i == 0 else WHITE),
                               edgecolor=INK, linewidth=0.9))
        pal = (PENALTY[1], PENALTY[3]) if i == 0 else PENALTY
        sw = bw / len(pal)
        for k, col in enumerate(pal):
            ax.add_patch(Rectangle((x + k * sw, yb), sw, strip_h,
                                   facecolor=col, edgecolor="none",
                                   linewidth=0, zorder=2))
        ax.text(x + bw / 2, yb + hb + 8.4, titles[i], ha="center",
                va="center", fontsize=8.5, color=INK, fontweight="medium")
        ax.text(x + bw / 2, yb + hb + 3.5, notes[i], ha="center",
                va="center", fontsize=7.4, color=INK2)
    ax.plot([x0, x0 + span], [yb + strip_h, yb + strip_h],
            color=INK, lw=0.35, zorder=3)
    for i, lab in enumerate(("0:00", "6:00", "12:00", "18:00", "24:00")):
        x = x0 + span * i / 4
        ax.plot([x, x], [yb, yb - 2.0], color=INK, lw=0.7)
        ax.text(x, yb - 6.0, lab, ha="center", va="center",
                fontsize=7.5, color=INK2)
    chips = (
        (PENALTY[0], r"违约  $0.5p$"),
        (PENALTY[1], r"计划  $1.0p$"),
        (PENALTY[2], r"加订  $1.5p$"),
        (PENALTY[3], r"紧急  $5.0p$"),
    )
    item_w, chip, cy = 30.0, 3.4, 16.6
    xa = 80.0 - 2 * item_w
    for i, (col, title) in enumerate(chips):
        x = xa + i * item_w
        ax.add_patch(Rectangle((x, cy), chip, chip, facecolor=col,
                               edgecolor=INK, linewidth=0.35))
        ax.text(x + chip + 1.8, cy + chip / 2, title, ha="left",
                va="center", fontsize=8.5, color=INK)
    ax.text(80.0, 6.0,
            r"$Z^{\mathrm{buy}}_t=p_t a_t+\frac{1}{2}p_t|a_t-g_t|$",
            ha="center", va="center", fontsize=11, color=INK)
    return save_v2(fig, out_dir, "fig14_题3分块承诺与四档价签")


# ---------------------------------------------------------------------------
# 数据 / 题 1
# ---------------------------------------------------------------------------

def fig03_data(day: DayData, out_dir: Path):
    fig, axes = new_rows(2, 6.30, 4.35)
    ax = axes[0]
    ax.plot(HOURS, day.p, color=C_PRICE, lw=1.9)
    bands = price_bands(ax, day.p)
    face(ax, "元/kWh")
    panel_title(ax, "a")
    legend_w(ax, bands, BAND_LABELS)

    ax = axes[1]
    load_kw, pv_kw = day.ell * 6, day.r * 6
    ax.plot(HOURS, load_kw, color=C_LOAD, lw=1.9, label="负载")
    ax.plot(HOURS, pv_kw, color=C_PV, lw=1.9, label="光伏")
    ax.fill_between(HOURS, pv_kw, load_kw, where=load_kw >= pv_kw,
                    color=C_LOAD, alpha=0.18, lw=0, label="缺口")
    ax.fill_between(HOURS, pv_kw, load_kw, where=load_kw < pv_kw,
                    color=C_CH, alpha=0.18, lw=0, label="光伏盈余")
    face(ax, "kW")
    panel_title(ax, "b")
    legend_w(ax, ncol=4)
    day_axis(ax)
    return save_v2(fig, out_dir, "fig03_附件1典型日-电价负荷光伏")


def fig04_schedule(day: DayData, sol, out_dir: Path):
    fig, axes = new_rows(3, 6.30, 6.55, ratios=(2.6, 2.8, 2.4))
    for a in axes:
        price_bands(a, day.p)

    ax = axes[0]
    ax.plot(HOURS, day.p, color=C_PRICE, lw=1.85, label=r"$p_t$")
    ax.plot(HOURS, sol.lam, color=C_DUAL, lw=1.55, ls="--",
            label=r"$\lambda_t$")
    ax.plot(HOURS, sol.nu, color=C_BUY, lw=1.55, ls="-.",
            label=r"$\nu_t$")
    face(ax, "元/kWh")
    panel_title(ax, "a")
    legend_w(ax, ncol=3)

    ax = axes[1]
    w = (1 / 6) * 0.88
    edge = dict(edgecolor=WHITE, linewidth=0.35)
    ax.bar(HOURS, sol.g * 6, w, color=C_BUY, label=r"购电 $g$", **edge)
    ax.bar(HOURS, sol.d * 6, w, bottom=sol.g * 6, color=C_DIS,
           label=r"放电 $d$", **edge)
    ax.bar(HOURS, -sol.c * 6, w, color=C_CH, label=r"充电 $c$", **edge)
    ax.axhline(0, color=BASE, lw=0.7)
    face(ax, "kW")
    panel_title(ax, "b")
    legend_w(ax, ncol=3)

    ax = axes[2]
    ax.plot(ENDS, sol.S, color=C_SOC, lw=1.85, drawstyle="steps-post")
    ax.axhline(S_MIN, color=MUTED, lw=0.8, ls="--")
    ax.axhline(S_MAX, color=MUTED, lw=0.8, ls="--")
    halo(ax.text(0.8, S_MAX + 180, "10800", fontsize=8, color=MUTED, va="bottom"))
    halo(ax.text(0.8, S_MIN - 70, "1200", fontsize=8, color=MUTED, va="top"))
    ax.set_ylim(0, S_MAX * 1.14)
    face(ax, "kWh")
    panel_title(ax, "c")
    day_axis(ax)
    return save_v2(fig, out_dir, "fig04_题1典型日调度")


def fig07_s0(outcome, out_dir: Path):
    fig, ax = new_plot(6.30, 3.35)
    grid, costs = outcome.s0_grid, np.asarray(outcome.s0_costs)
    z_star, s0 = outcome.sol.cost, float(outcome.sol.S[0])
    z6000 = outcome.baselines["Z^6000"]
    flat = np.flatnonzero(costs <= z_star + 1.0)
    if len(flat):
        ax.axvspan(grid[flat[0]], grid[flat[-1]], color=C_SOC, alpha=0.12,
                   lw=0, label=f"平坦盆地（宽 {grid[flat[-1]] - grid[flat[0]]:.0f} kWh）")
    ax.plot(grid, costs, color=C_PRICE, lw=1.9)
    ax.plot([s0], [z_star], "o", color=C_SOC, ms=8, markeredgecolor=WHITE,
            zorder=3, label=rf"$S_0^\star={s0:.0f}$")
    ax.plot([6000], [z6000], "s", color=C_LOAD, ms=7, markeredgecolor=WHITE,
            zorder=3, label=f"钉死 6000，+{z6000 - z_star:.2f} 元")
    ax.annotate(rf"$Z^\star={z_star:,.2f}$ 元",
                xy=(s0, z_star), xytext=(s0 - 2800, z_star + 28),
                fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    ax.margins(x=0.02, y=0.08)
    ax.set_xlabel(r"$S_0$（kWh）", fontsize=9, color=INK2)
    face(ax, "全天购电费（元，纵轴已放大）")
    legend_w(ax, loc="upper right")
    return save_v2(fig, out_dir, "fig07_S0敏感性")


# ---------------------------------------------------------------------------
# 题 2 / 题 3 / 题 4
# ---------------------------------------------------------------------------

def fig13_ladder(out_dir: Path):
    apply_v2()
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.55), dpi=200,
                             facecolor=WHITE, layout="constrained",
                             gridspec_kw={"width_ratios": (1.08, 1.12)})
    fig.set_facecolor(WHITE)
    pads(fig, wspace=0.22)
    ax = axes[0]
    names = ("完美信息  PI", "事后补救  ex-post", "交付  online", "确定性等价  CE")
    vals = np.array([12223092.57, 13463630.76, 13745816.13, 14254554.72]) / 1e4
    cols = (C_SOC, C_BUY, C_PRICE, MUTED)
    y = np.arange(4)
    ax.barh(y, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 8, i, f"{v:,.1f} 万", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, vals.max() * 1.24)
    ax.set_xlabel("报告期费用（万元）", fontsize=9, color=INK2)
    face(ax)
    ax.grid(axis="y", visible=False)
    panel_title(ax, "a")

    ax = axes[1]
    labels = ("① 功率强制", "② 储能耗尽", "③a 囤电", "③b 配给", "④ 低水位带")
    v6 = np.array([8.71, 18.99, 0.00, 49.06, 23.23])
    nocap = np.array([3.50, 17.80, 41.70, 19.31, 17.69])
    y = np.arange(len(labels))
    ax.barh(y - 0.18, nocap, 0.34, color="#b4b1a8", label="无上界")
    ax.barh(y + 0.18, v6, 0.34, color=C_PRICE, label="交付")
    halo(ax.text(2.2, 2 + 0.18, "交付 0", ha="left", va="center",
                 fontsize=8.5, color=C_PRICE))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 56)
    ax.set_xlabel("紧急电量占比（%）", fontsize=9, color=INK2)
    face(ax)
    ax.grid(axis="y", visible=False)
    panel_title(ax, "b")
    legend_w(ax, loc="lower right")
    return save_v2(fig, out_dir, "fig13_题2费用阶梯与紧急归因")


def fig15_fusion(year, fc, out_dir: Path):
    r_hour = hour_powers(year.r)
    phi, tau = fc.phi, np.array(VINTAGE_HOURS)
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
                if actual < 80:
                    continue
                err_fc[h - 1].append(phi[j, m, h - 1] - actual)
                err_pe[h - 1].append(pers[tgt - 1] - actual)
    h = np.arange(1, 19)
    rmse_fc = np.array([np.sqrt(np.mean(np.square(e))) if e else np.nan
                        for e in err_fc[:18]])
    rmse_pe = np.array([np.sqrt(np.mean(np.square(e))) if e else np.nan
                        for e in err_pe[:18]])
    phi_w = rmse_pe ** 2 / (rmse_pe ** 2 + rmse_fc ** 2)

    fig, axes = new_rows(2, 6.30, 4.55)
    ax = axes[0]
    ax.plot(h, rmse_fc, color=C_PV, lw=1.9, marker="o", ms=3.6,
            label="官方预报")
    ax.plot(h, rmse_pe, color=C_CH, lw=1.9, marker="s", ms=3.6,
            label="持续法")
    diff = rmse_fc - rmse_pe
    cross = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
    if len(cross):
        i = int(cross[0])
        ax.axvline(h[i] + 0.5, color=MUTED, lw=0.85, ls="--")
        halo(ax.text(h[i] + 0.65, np.nanmax(rmse_fc) * 0.90, "交叉 ≈ 9–10 h",
                     fontsize=8.5, color=INK2))
    face(ax, "RMSE（kW）")
    panel_title(ax, "a")
    legend_w(ax)

    ax = axes[1]
    ax.plot(h, phi_w, color=C_PRICE, lw=1.9, marker="o", ms=3.6)
    ax.axhline(0.5, color=MUTED, lw=0.85, ls="--")
    halo(ax.text(1.15, 0.54, "等权 0.5", fontsize=8, color=MUTED, va="bottom"))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.6, 18.6)
    axes[0].set_xlim(0.6, 18.6)
    ax.set_xlabel("预报提前期（h）", fontsize=9, color=INK2)
    face(ax, r"官方权重 $\varphi$")
    panel_title(ax, "b")
    return save_v2(fig, out_dir, "fig15_题3白天预报融合")


def _soc_10min(S0: float, c_bat: np.ndarray, d_bat: np.ndarray) -> np.ndarray:
    """4 小时报表块均匀拆到 10 分钟：保持日初、日末与块电量，不作 7 点折线。"""
    s = np.empty(T + 1)
    s[0] = S0
    for k in range(6):
        dc, dd = c_bat[k] / 24.0, d_bat[k] / 24.0
        for i in range(24):
            t = k * 24 + i
            s[t + 1] = s[t] + dc - dd
    return s


def fig16_revision(year, day1, dates3, g3, a3, bat3, out_dir: Path):
    dev = np.abs(a3[:, 36:] - g3[:, 36:]).sum(axis=1)
    i = int(np.argmax(dev))
    date = dates3[i]
    g_kw, a_kw = g3[i] * 6, a3[i] * 6
    row = bat3.loc[pd.Timestamp(date)]
    s = _soc_10min(float(row["S0"]), row["c"], row["d"])

    fig, axes = new_rows(3, 6.30, 6.85, ratios=(3.0, 1.9, 2.2))
    ax = axes[0]
    price_bands(ax, day1.p)
    ax.axvspan(0, 6, color=GRID, alpha=0.45, lw=0)
    for h in (6, 12, 18):
        ax.axvline(h, color=BASE, lw=0.8, ls="--")
    ax.plot(HOURS, g_kw, color=C_BUY, lw=1.8, drawstyle="steps-mid",
            label=r"计划 $g_t$")
    ax.plot(HOURS, a_kw, color=C_PV, lw=1.8, ls="--",
            drawstyle="steps-mid", label=r"注入 $a_t$")
    face(ax, "kW")
    panel_title(ax, "a", str(date.date()))
    legend_w(ax, [
        Line2D([], [], color=C_BUY, lw=1.8),
        Line2D([], [], color=C_PV, lw=1.8, ls="--"),
        Patch(facecolor=GRID, alpha=0.45, lw=0),
    ], [r"计划 $g_t$", r"注入 $a_t$", r"$\mathcal{B}_0$ 不可修订"])

    ax = axes[1]
    delta = a_kw - g_kw
    w = (1 / 6) * 0.9
    ax.bar(HOURS, np.where(delta > 0, delta, 0.0), w, color=C_LOAD, lw=0)
    ax.bar(HOURS, np.where(delta < 0, delta, 0.0), w, color=C_PRICE, lw=0)
    ax.axhline(0, color=BASE, lw=0.7)
    for h in (6, 12, 18):
        ax.axvline(h, color=BASE, lw=0.8, ls="--")
    face(ax, "kW")
    panel_title(ax, "b")
    legend_w(ax, [
        Patch(facecolor=C_LOAD, lw=0),
        Patch(facecolor=C_PRICE, lw=0),
    ], [r"上调 $\delta^+$", r"下调 $\delta^-$"], loc="upper right")

    ax = axes[2]
    ax.plot(ENDS, s, color=C_SOC, lw=1.85, drawstyle="steps-post")
    ax.axhline(S_MIN, color=MUTED, lw=0.8, ls="--")
    ax.axhline(S_MAX, color=MUTED, lw=0.8, ls="--")
    halo(ax.text(0.4, S_MIN + 140, "1200", fontsize=8, color=MUTED))
    halo(ax.text(0.4, S_MAX - 260, "10800", fontsize=8, color=MUTED))
    ax.set_ylim(0, S_MAX * 1.12)
    face(ax, "kWh")
    panel_title(ax, "c")
    day_axis(ax)
    return save_v2(fig, out_dir, "fig16_题3修订演示-最大偏差日")


def fig17_q2q3(q2: pd.DataFrame, q3: pd.DataFrame, year, out_dir: Path):
    apply_v2()
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.55), dpi=200,
                             facecolor=WHITE, layout="constrained",
                             gridspec_kw={"width_ratios": (1.0, 1.28)})
    fig.set_facecolor(WHITE)
    pads(fig, wspace=0.24)
    ax = axes[0]
    q2_emerg = float(q2["emerg_fee"].sum())
    q2_z = float(q2["z_online"].sum())
    q3_base = float(q3["base_fee"].sum())
    q3_dev = float(q3["dev_fee"].sum())
    q3_emerg = float(q3["emerg_fee"].sum())
    s2 = np.array([q2_z - q2_emerg, 0.0, q2_emerg]) / 1e4
    s3 = np.array([q3_base, q3_dev, q3_emerg]) / 1e4
    cols = (C_BUY, C_CH, C_EMERG)
    labels = ("基础购电", "偏差费", "紧急费")
    b2 = b3 = 0.0
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(0, s2[i], 0.52, bottom=b2, color=c, label=lab)
        ax.bar(1, s3[i], 0.52, bottom=b3, color=c)
        b2 += s2[i]
        b3 += s3[i]
    ax.set_xticks([0, 1], ("问题 2", "问题 3"))
    ax.set_xlim(-0.7, 1.7)
    ax.text(0, b2 + 22, f"{b2:,.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.text(1, b3 + 22, f"{b3:,.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, max(b2, b3) * 1.42)
    face(ax, "报告期费用（万元）")
    ax.grid(axis="x", visible=False)
    panel_title(ax, "a")
    legend_w(ax, loc="upper left")
    ax.set_xlim(-0.7, 1.7)

    ax = axes[1]
    aligned = (q3["z_online"].to_numpy()
               - q2.reindex(q3.index)["z_online"].to_numpy())
    x = np.arange(len(aligned))
    ax.bar(x, np.where(aligned > 0, aligned, 0.0), width=1.0, color=C_LOAD, lw=0)
    ax.bar(x, np.where(aligned < 0, aligned, 0.0), width=1.0, color=C_PRICE, lw=0)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axhline(np.nanmean(aligned), color=MUTED, lw=1.1, ls="--")
    face(ax, "题 3 − 题 2（元/日）")
    panel_title(ax, "b")
    dates = year.dates[WARMUP_DAYS:WARMUP_DAYS + len(aligned)]
    month_axis(ax, dates, len(aligned))
    ticks = list(ax.get_xticks())
    labs = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(ticks[::2], labs[::2])
    legend_w(ax, [
        Patch(facecolor=C_LOAD, lw=0, label="题 3 更贵"),
        Patch(facecolor=C_PRICE, lw=0, label="题 3 更省"),
        Line2D([], [], color=MUTED, ls="--",
               label=f"均值 {np.nanmean(aligned):+,.0f} 元/日"),
    ])
    return save_v2(fig, out_dir, "fig17_题2与题3费用对照")


def fig18_epochs(out_dir: Path):
    fig, ax = new_plot(6.30, 3.15)
    names = [r[0] for r in LADDER_Z]
    vals = np.array([r[1] for r in LADDER_Z]) / 1e4
    y = np.arange(4)
    ax.barh(y, vals, color=C_PRICE, height=0.55)
    for i, v in enumerate(vals):
        extra = ""
        if i:
            extra = f"  ({vals[i] - vals[i - 1]:+.1f})"
        ax.text(v + 3.5, i, f"{v:,.1f} 万{extra}", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(1288, 1388)
    ax.set_xlabel("报告期费用（万元，横轴已截断）", fontsize=9, color=INK2)
    face(ax)
    ax.grid(axis="y", visible=False)
    return save_v2(fig, out_dir, "fig18_题3修订阶梯")


def _next_dow(dates: pd.DatetimeIndex) -> np.ndarray:
    """与 report/q4.py 星期台阶同一口径：公历次日星期，年末不改写。"""
    return (dates + pd.Timedelta(days=1)).dayofweek.to_numpy()


def fig19_weekday(q4: pd.DataFrame, out_dir: Path):
    wd = _next_dow(q4.index)
    s144 = q4["S144"].to_numpy()
    order = ("次日周五", "次日周六", "次日周一", "其他日期")
    keys = (4, 5, 0, None)
    vals = []
    for key in keys:
        mask = wd == key if key is not None else ~np.isin(wd, (4, 5, 0))
        vals.append(float(s144[mask].mean()) if mask.any() else np.nan)
    vals = np.array(vals)
    fig, ax = new_plot(6.30, 3.15)
    y = np.arange(4)
    cols = (C_LOAD, C_SOC, C_SOC, MUTED)
    ax.barh(y, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 80, i, f"{v:,.0f} kWh", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, np.nanmax(vals) * 1.22)
    ax.set_xlabel("日末水位（kWh）", fontsize=9, color=INK2)
    face(ax)
    ax.grid(axis="y", visible=False)
    return save_v2(fig, out_dir, "fig19_题4日末水位星期台阶")


def fig20_price(day: DayData, price, out_dir: Path):
    """日内形态稳定 + 周内水平：读法 R 可行性的数据图。"""
    p, dow = price.p, price.dow
    p_mon = p[dow == 0].mean(axis=0)
    p_fri = p[dow == 4].mean(axis=0)
    daily = p.mean(axis=1)
    by_wd = np.array([float(daily[dow == k].mean()) for k in range(7)])
    rest = float(np.mean([by_wd[k] for k in (0, 1, 2, 3, 6)]))
    weekend = float(np.mean(by_wd[4:6]))
    drop = (rest - weekend) / rest * 100.0

    fig, axes = new_rows(2, 6.30, 4.55, ratios=(2.6, 2.0), sharex=False)
    ax = axes[0]
    ax.axvspan(0.0, 6.0, color=GRID, alpha=0.55, lw=0)
    ax.plot(HOURS, day.p, color=C_PRICE, lw=1.9, label="附件 1（全年均值）")
    ax.plot(HOURS, p_mon, color=C_LOAD, lw=1.45, label="周一")
    ax.plot(HOURS, p_fri, color=C_SOC, lw=1.45, label="周五")
    halo(ax.text(3.0, 0.58, r"$\mathcal{B}_0$",
                 ha="center", fontsize=8.5, color=MUTED))
    halo(ax.text(11.6, 0.08, "午间深谷",
                 ha="center", va="bottom", fontsize=8, color=INK2))
    ax.set_ylim(0.0, float(np.max(p_mon)) * 1.08)
    face(ax, "元/kWh")
    panel_title(ax, "a")
    legend_w(ax, loc="upper left")
    day_axis(ax)

    ax = axes[1]
    names = ("一", "二", "三", "四", "五", "六", "日")
    cols = [C_SOC if k in (4, 5) else C_PRICE for k in range(7)]
    ax.bar(np.arange(7), by_wd, 0.62, color=cols)
    for k, v in enumerate(by_wd):
        ax.text(k, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK2)
    halo(ax.text(4.5, float(by_wd.max()) * 1.12, f"周五/六 低 {drop:.0f}%",
                 ha="center", fontsize=8.5, color=C_SOC))
    ax.set_xticks(np.arange(7), names)
    ax.set_ylim(0, float(by_wd.max()) * 1.28)
    ax.set_xlabel("星期", fontsize=9, color=INK2)
    face(ax, "日均价（元/kWh）")
    panel_title(ax, "b")
    ax.grid(axis="x", visible=False)
    return save_v2(fig, out_dir, "fig20_题4价格形态与周内水平")


def fig21_ablate(out_dir: Path):
    fig, ax = new_plot(6.30, 3.15)
    names = [r[0] for r in Q4_LADDER]
    zs = np.array([r[1] for r in Q4_LADDER])
    d = (zs - zs[2]) / 1e4
    y = np.arange(4)
    cols = (MUTED, MUTED, C_PRICE, MUTED)
    ax.barh(y, d, color=cols, height=0.55)
    ax.axvline(0, color=INK2, lw=0.8)
    for i, v in enumerate(d):
        lab = "0（主交付）" if i == 2 else f"{v:+.2f} 万"
        ax.text(v + (0.12 if v >= 0 else -0.12), i, lab,
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(-5.6, 5.2)
    ax.set_xlabel("相对主交付的全年费用差额（万元）", fontsize=9, color=INK2)
    face(ax)
    ax.grid(axis="y", visible=False)
    return save_v2(fig, out_dir, "fig21_题4价格信息阶梯")


def fig22_q42q43(q42: pd.DataFrame, q43: pd.DataFrame, out_dir: Path):
    apply_v2()
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.55), dpi=200,
                             facecolor=WHITE, layout="constrained",
                             gridspec_kw={"width_ratios": (1.0, 1.28)})
    fig.set_facecolor(WHITE)
    pads(fig, wspace=0.24)
    ax = axes[0]
    z2 = float(q42["z_online"].sum())
    e2 = float(q42["emerg_fee"].sum())
    z3 = float(q43["z_online"].sum())
    d3 = float(q43["dev_fee"].sum())
    e3 = float(q43["emerg_fee"].sum())
    s2 = np.array([z2 - e2, 0.0, e2]) / 1e4
    s3 = np.array([z3 - d3 - e3, d3, e3]) / 1e4
    cols = (C_BUY, C_CH, C_EMERG)
    labels = ("基础购电", "偏差费", "紧急费")
    b2 = b3 = 0.0
    for i, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(0, s2[i], 0.52, bottom=b2, color=c, label=lab)
        ax.bar(1, s3[i], 0.52, bottom=b3, color=c)
        b2 += s2[i]
        b3 += s3[i]
    ax.set_xticks([0, 1], ("计划线", "调整线"))
    ax.set_xlim(-0.7, 1.7)
    ax.text(0, b2 + 22, f"{b2:,.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.text(1, b3 + 22, f"{b3:,.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0, max(b2, b3) * 1.42)
    face(ax, "报告期费用（万元）")
    ax.grid(axis="x", visible=False)
    panel_title(ax, "a")
    legend_w(ax, loc="upper left")
    ax.set_xlim(-0.7, 1.7)

    ax = axes[1]
    aligned = (q43["z_online"].to_numpy()
               - q42.reindex(q43.index)["z_online"].to_numpy())
    x = np.arange(len(aligned))
    ax.bar(x, np.where(aligned > 0, aligned, 0.0), width=1.0, color=C_LOAD, lw=0)
    ax.bar(x, np.where(aligned < 0, aligned, 0.0), width=1.0, color=C_PRICE, lw=0)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axhline(np.nanmean(aligned), color=MUTED, lw=1.1, ls="--")
    face(ax, "调整线 − 计划线（元/日）")
    panel_title(ax, "b")
    dates = q43.index.to_numpy(dtype="datetime64[D]")
    month_axis(ax, dates, len(aligned))
    ticks = list(ax.get_xticks())
    labs = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(ticks[::2], labs[::2])
    legend_w(ax, [
        Patch(facecolor=C_LOAD, lw=0, label="调整线更贵"),
        Patch(facecolor=C_PRICE, lw=0, label="调整线更省"),
        Line2D([], [], color=MUTED, ls="--",
               label=f"均值 {np.nanmean(aligned):+,.0f} 元/日"),
    ], loc="lower left")
    return save_v2(fig, out_dir, "fig22_题4计划线与调整线")


def fig_a2_heat(dates3, g3, a3, year, out_dir: Path):
    diff = a3 - g3
    code = np.zeros_like(diff, dtype=np.uint8)
    code[diff > EPS] = 1
    code[diff < -EPS] = 2
    n = len(code)
    idle, b0 = WHITE, "#eceae4"
    fig, ax = new_plot(6.30, 5.15)
    ax.set_facecolor(b0)
    ax.imshow(code[:, 36:], aspect="auto", interpolation="nearest",
              cmap=ListedColormap([idle, C_LOAD, C_PRICE]), vmin=0, vmax=2,
              extent=(6.0, 24.0, float(n), 0.0))
    ax.axvspan(0.0, 6.0, facecolor=b0, edgecolor="none", zorder=0)
    ax.text(3.0, n / 2, r"$\mathcal{B}_0$  不可修订",
            ha="center", va="center", rotation=90, fontsize=9, color=MUTED)
    for h in (6.0, 12.0, 18.0):
        ax.axvline(h, color=BASE, lw=0.8, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks((0, 6, 12, 18, 24))
    ax.set_xlabel("时刻（h）", fontsize=9, color=INK2)
    month_axis(ax, year.dates[WARMUP_DAYS:WARMUP_DAYS + n], n, on_y=True)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    legend_w(ax, [
        Patch(facecolor=C_LOAD, lw=0),
        Patch(facecolor=C_PRICE, lw=0),
        Patch(facecolor=idle, edgecolor=BASE, linewidth=0.6),
    ], [r"上调 $\delta^+$", r"下调 $\delta^-$", "未改"],
        loc="outside lower center", ncol=3)
    return save_v2(fig, out_dir, "figA2_题3修订指纹")


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

def _report(path: Path, issues: list[str]) -> None:
    flag = "OK" if not issues else f"{len(issues)} issues"
    print(f"  [{flag}] {path.name}")
    for it in issues[:8]:
        print(f"      {it}")


def _only_set(only: list[str] | None) -> set[str] | None:
    if not only:
        return None
    wanted: set[str] = set()
    for token in only:
        t = token.strip()
        if t.lower() in {"q4", "题4"}:
            wanted.update(Q4_FIGS)
            continue
        if t.startswith("fig"):
            wanted.add(t.split("_")[0])
        elif t[:1].isdigit():
            wanted.add(f"fig{t}")
        else:
            wanted.add(t)
    return wanted


def render_all(out_dir: Path | None = None,
               only: list[str] | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else ROOT / "results" / "paper-v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_v2()
    wanted = _only_set(only)

    def go(name: str) -> bool:
        return wanted is None or name in wanted

    day1 = load_day1()
    year = fc = q1 = None
    dates3 = g3 = a3 = bat3 = daily2 = daily3 = None
    if go("fig15") or go("fig16") or go("fig17") or go("figA2"):
        year = load_year()
    if go("fig15"):
        fc = load_forecast()
    if go("fig04") or go("fig07"):
        q1 = run_q1(day1)
        q1.baselines["Z_base"] = float(
            day1.p @ np.maximum(day1.ell - day1.r, 0.0))
    if go("fig16") or go("figA2"):
        q3x = src.Q3_DIR / "result3.xlsx"
        dates3, g3 = src.load_plan(q3x, "计划购电量")
        _, a3 = src.load_plan(q3x, "调整购电量")
        bat3 = src.load_battery(q3x)
    if go("fig17"):
        daily2 = src.load_daily(src.Q2_DIR / "q2_details.csv")
        daily3 = src.load_daily(src.Q3_DIR / "q3_details.csv")
    q42_path, q43_path = Q4_DIR / "q4-2_details.csv", Q4_DIR_43 / "q4-3_details.csv"
    q42 = src.load_daily(q42_path) if q42_path.exists() else None
    q43 = src.load_daily(q43_path) if q43_path.exists() else None
    price = load_price(day1.p) if go("fig20") else None

    steps: list[tuple[str, object]] = []
    if go("fig01"):
        steps.append(("fig01", lambda: fig01_bus(out_dir)))
    if go("fig02"):
        steps.append(("fig02", lambda: fig02_framework(out_dir)))
    if go("fig03"):
        steps.append(("fig03", lambda: fig03_data(day1, out_dir)))
    if go("fig04"):
        steps.append(("fig04", lambda: fig04_schedule(day1, q1.sol, out_dir)))
    if go("fig07"):
        steps.append(("fig07", lambda: fig07_s0(q1, out_dir)))
    if go("fig09"):
        steps.append(("fig09", lambda: fig09_arch(out_dir)))
    if go("fig13"):
        steps.append(("fig13", lambda: fig13_ladder(out_dir)))
    if go("fig14"):
        steps.append(("fig14", lambda: fig14_commit(out_dir)))
    if go("fig15"):
        steps.append(("fig15", lambda: fig15_fusion(year, fc, out_dir)))
    if go("fig16"):
        steps.append(("fig16", lambda: fig16_revision(
            year, day1, dates3, g3, a3, bat3, out_dir)))
    if go("fig17"):
        steps.append(("fig17", lambda: fig17_q2q3(daily2, daily3, year, out_dir)))
    if go("fig18"):
        steps.append(("fig18", lambda: fig18_epochs(out_dir)))
    if go("figA2"):
        steps.append(("figA2", lambda: fig_a2_heat(dates3, g3, a3, year, out_dir)))
    if go("fig20") and price is not None:
        steps.append(("fig20", lambda: fig20_price(day1, price, out_dir)))
    if go("fig21"):
        steps.append(("fig21", lambda: fig21_ablate(out_dir)))
    if go("fig22") and q42 is not None and q43 is not None:
        steps.append(("fig22", lambda: fig22_q42q43(q42, q43, out_dir)))
    if go("fig19") and q42 is not None and "S144" in q42.columns:
        steps.append(("fig19", lambda: fig19_weekday(q42, out_dir)))

    print(f"写出 {len(steps)} 张 → {out_dir}")
    keep: set[str] = set()
    for name, fn in steps:
        path, issues = fn()
        keep.add(path.stem)
        _report(path, issues)
        plt.close("all")
    if wanted is None:
        for p in out_dir.glob("fig*.*"):
            if p.suffix.lower() in {".png", ".pdf"} and p.stem not in keep:
                p.unlink()
    return out_dir
