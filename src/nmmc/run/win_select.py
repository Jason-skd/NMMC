"""issue #6 去超参实验执行层：预报层技能对照、题2 全年费用承重、
题3 融合层光伏对照、双实现交叉验证。

三段（CLI --part 分发）：

  forecast   选窗序列 + 五臂预报技能对照（分钟级）——固定 4/7 基线、
             负载/光伏单独自适应、双自适应、零参数 L3zp×P3zp。
  downstream 题2 全年 rollout 四臂（固定 / 仅负载 / 仅光伏 / 双自适应，
             各约 13 分钟，增量缓存防中断），Z 现金口径承重检验 + 逐族
             归因 + 既有交付基线复现核对。
  q3fc       题3 融合层对照：持续法窗固定 7 vs 逐日 w_r（融合权重与
             点预报同步换窗），逐阶段光伏预报 MAE。

交叉验证（双实现纪律，不写测试）：MAE 曲面为封闭形式重放，与交付层
forecast_load / forecast_pv 逐日重算对拍；自适应 rollout 的逐日点预报
与预报层实验臂逐字节对拍；1-SE 选择抽样直接复算。
"""
from dataclasses import replace
from pathlib import Path
import re

import numpy as np
import pandas as pd

from nmmc.data.day import DayData
from nmmc.data.fcst import YearForecast
from nmmc.data.year import YearData
from nmmc.model.forecast import ForecastConfig, forecast_load, forecast_pv
from nmmc.model.forecast3 import (Q3Config, fusion_weights, hour_powers,
                                  pv_point)
from nmmc.model.winselect import (LOAD_GRID, PV_GRID, select_one_se,
                                  window_schedule, zp_load, zp_pv)
from nmmc.report.common import md_table
from nmmc.report.q2 import WARMUP_DAYS, summarize
from nmmc.run.q2 import rollout

BASE_CFG = ForecastConfig()          # 交付基线 L1×P1×R2，W_L=4 / W_R=7
BASELINE_MD = Path("results/q2/L1-P1-R2-smpc/q2_summary.md")  # 交付基线复现源
ARMS = ("L1w4-P1w7", "L1ad-P1w7", "L1w4-P1ad", "L1ad-P1ad", "L3zp-P3zp")
ARM_DESCR = {
    "L1w4-P1w7": "固定 4/7（交付基线）",
    "L1ad-P1w7": "仅负载自适应 W_L",
    "L1w4-P1ad": "仅光伏自适应 W_R",
    "L1ad-P1ad": "双自适应 W_L+W_R",
    "L3zp-P3zp": "零参数（全历史回归 × 全历史形态）",
}


def arm_forecasts(year: YearData, typ_ell: np.ndarray, typ_r: np.ndarray,
                  schedule: pd.DataFrame
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """五臂全年逐日点预报：每臂 (ell_hat, r_hat) 各 (365, 144)。

    自适应臂逐日 replace(cfg, load_lags/pv_window=选中窗)，与下游
    rollout 的换窗路径同一调用（交叉验证逐字节可对拍的前提）。
    """
    w_l = schedule["w_l"].to_numpy()
    w_r = schedule["w_r"].to_numpy()
    n = len(year.dates)
    out = {arm: (np.empty((n, 144)), np.empty((n, 144))) for arm in ARMS}
    for j in range(n):
        # L1×P1 四臂：固定 / 自适应窗替换（复用交付层 forecast_*）
        for arm in ARMS[:4]:
            cfg = BASE_CFG
            if arm.startswith("L1ad"):
                cfg = replace(cfg, load_lags=int(w_l[j]))
            if arm.endswith("P1ad"):
                cfg = replace(cfg, pv_window=int(w_r[j]))
            out[arm][0][j] = forecast_load(cfg, year, j, typ_ell)
            out[arm][1][j] = forecast_pv(cfg, year, j, typ_r)
        # 零参数臂：全历史回归 × 全历史形态（无任何窗超参）
        out["L3zp-P3zp"][0][j] = zp_load(year, j, typ_ell)
        out["L3zp-P3zp"][1][j] = zp_pv(year, j, typ_r)
    return out


def skill_table(year: YearData,
                fcasts: dict) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """报告期（j ≥ 31）五臂 MAE 技能表：负载 / 光伏 / 净负荷，附相对
    基线差。净负荷为下游链条的真正输入，单列以对齐承重口径。"""
    rep = np.arange(WARMUP_DAYS, len(year.dates))
    net_true = year.ell[rep] - year.r[rep]
    rows, daily = [], []
    base = {}
    for arm in ARMS:
        e_l = float(np.abs(fcasts[arm][0][rep] - year.ell[rep]).mean())
        e_r = float(np.abs(fcasts[arm][1][rep] - year.r[rep]).mean())
        e_n = float(np.abs((fcasts[arm][0][rep] - fcasts[arm][1][rep])
                           - net_true).mean())
        if arm == ARMS[0]:
            base = {"l": e_l, "r": e_r, "n": e_n}
        rows.append((arm, ARM_DESCR[arm], f"{e_l:.3f}", f"{(e_l / base['l'] - 1) * 100:+.2f}%",
                     f"{e_r:.3f}", f"{(e_r / base['r'] - 1) * 100:+.2f}%",
                     f"{e_n:.3f}", f"{(e_n / base['n'] - 1) * 100:+.2f}%"))
        # 逐日明细（供后续作图）
        dl = np.abs(fcasts[arm][0][rep] - year.ell[rep]).mean(axis=1)
        dr = np.abs(fcasts[arm][1][rep] - year.r[rep]).mean(axis=1)
        dn = np.abs((fcasts[arm][0][rep] - fcasts[arm][1][rep]) - net_true).mean(axis=1)
        daily.append(pd.DataFrame({"j": rep, "arm": arm,
                                   "mae_l": dl, "mae_r": dr, "mae_n": dn}))
    df = pd.DataFrame(rows, columns=("arm", "descr", "mae_l", "d_l",
                                     "mae_r", "d_r", "mae_n", "d_n"))
    md = md_table(["臂", "说明", "负载 MAE", "Δ%", "光伏 MAE", "Δ%",
                   "净负荷 MAE", "Δ%"], rows)
    return df, md, pd.concat(daily, ignore_index=True)


def cross_checks(year: YearData, mae_l: np.ndarray, mae_r: np.ndarray,
                 schedule: pd.DataFrame) -> list[str]:
    """双实现对拍：① MAE 曲面抽样（交付层逐日重算）；② 1-SE 选择
    抽样（评估集直接复算）。全部通过返回报告行，不一致直接抛错。"""
    lines = []
    typ = np.zeros(144)  # 占位先验——对拍日均有历史，typ 不会被用到

    # ① 曲面对拍：封闭形式重放 vs forecast_load / forecast_pv 逐日调用
    worst = 0.0
    for i in (80, 150, 250, 364):
        for w in (2, 4, 7, 10):
            gl = LOAD_GRID.index(w) if w in LOAD_GRID else None
            gr = PV_GRID.index(w) if w in PV_GRID else None
            if gl is not None:
                pred = forecast_load(replace(BASE_CFG, load_lags=w), year, i, typ)
                worst = max(worst, abs(float(np.abs(pred - year.ell[i]).mean())
                                       - mae_l[i, gl]))
            if gr is not None:
                pred = forecast_pv(replace(BASE_CFG, pv_window=w), year, i, typ)
                worst = max(worst, abs(float(np.abs(pred - year.r[i]).mean())
                                       - mae_r[i, gr]))
    assert worst < 1e-9, f"MAE 曲面对拍失败: max |diff| = {worst:.3e}"
    lines.append(f"MAE 曲面对拍（16 组抽样 × 交付层重算）：max |diff| = {worst:.3e}")

    # ② 选择对拍：直接从曲面上重算评估集均值 / 标准误与选中窗
    for j0 in (120, 240, 364):
        for name, mae, grid, col in (("l", mae_l, LOAD_GRID, "w_l"),
                                     ("r", mae_r, PV_GRID, "w_r")):
            usable = ~np.isnan(mae[:j0]).any(axis=1)
            sub = mae[:j0][usable]
            mean, se = sub.mean(axis=0), sub.std(axis=0, ddof=1) / np.sqrt(len(sub))
            w = select_one_se(mean, se, grid)
            assert w == int(schedule[col][j0]), \
                f"{name} 选择对拍失败 j={j0}: {w} vs {schedule[col][j0]}"
    lines.append("1-SE 选择对拍（3 日 × 两族直接复算）：逐日一致")
    return lines


DOWN_ARMS = ("L1w4-P1w7", "L1ad-P1w7", "L1w4-P1ad", "L1ad-P1ad")
DOWN_KEYS = (("Z_online", "Z^online（元）", "{:,.2f}"),
             ("Z_ce", "Z^CE 对照（元）", "{:,.2f}"),
             ("执行层增益", "执行层增益（元）", "{:,.2f}"),
             ("Z_expost", "Z^ex-post（元）", "{:,.2f}"),
             ("Z_pi", "Z^PI（元）", "{:,.2f}"),
             ("EVPI", "EVPI（元）", "{:,.2f}"),
             ("负载预报 MAE", "负载预报 MAE（kWh/时段）", "{:.3f}"),
             ("光伏预报 MAE", "光伏预报 MAE（kWh/时段）", "{:.3f}"),
             ("紧急购电费合计", "紧急购电费（元）", "{:,.2f}"),
             ("紧急购电天数", "紧急购电天数", "{:d}"),
             ("弃电率", "弃电率", "{:.2%}"))


def run_downstream(year: YearData, day1: DayData, schedule: pd.DataFrame,
                   out_dir: Path, days: int | None = None
                   ) -> tuple[str, list[str], dict[str, float]]:
    """题2 全年四臂承重：固定 4/7 / 仅负载 / 仅光伏 / 双自适应选窗。
    同管线同种子同 SOC 链，臂间唯一差异是窗长来源；逐臂结果增量落盘
    downstream.csv 防中断（days 不匹配的缓存行忽略）。固定臂兼作既有
    交付基线的复现核对，自适应臂与预报层实验臂逐字节对拍。"""
    cache = out_dir / "downstream.csv"
    done: dict[str, dict] = {}
    if cache.exists():
        old = pd.read_csv(cache)
        old = old[old["days"] == (days or len(year.dates))]
        done = {r["arm"]: r.to_dict() for _, r in old.iterrows()}

    n = len(year.dates)
    w_l = schedule["w_l"].to_numpy()
    w_r = schedule["w_r"].to_numpy()
    # 各臂逐日窗长对：固定臂用常数对（与不传 win_schedule 等价）
    scheds = {"L1w4-P1w7": (np.full(n, 4), np.full(n, 7)),
              "L1ad-P1w7": (w_l, np.full(n, 7)),
              "L1w4-P1ad": (np.full(n, 4), w_r),
              "L1ad-P1ad": (w_l, w_r)}
    fc_all = arm_forecasts(year, day1.ell, day1.r, schedule)
    lines = []
    for arm in DOWN_ARMS:
        if arm not in done:
            res = rollout(year, day1.p, day1.ell, day1.r, BASE_CFG,
                          win_schedule=scheds[arm], days=days)
            s = summarize(res, year, day1.p)

            # 交叉验证：rollout 逐日点预报 vs 预报层实验臂逐字节对拍
            fc = fc_all[arm]
            dl = max(float(np.abs(d.ell_hat - fc[0][d.j]).max()) for d in res)
            dr = max(float(np.abs(d.r_hat - fc[1][d.j]).max()) for d in res)
            assert dl == 0.0 and dr == 0.0, f"rollout 对拍失败: {dl:.3e} / {dr:.3e}"
            lines.append(f"rollout 点预报对拍（{arm} {len(res)} 日逐字节）："
                         "max |diff| = 0")
            if arm == "L1w4-P1w7" and days is None and BASELINE_MD.exists():
                # 固定臂复现核对：与交付基线 q2_summary 的 Z^online 对到分
                m = re.search(r"Z\^online（交付）\s*\|\s*([\d,]+)\.(\d{2})",
                              BASELINE_MD.read_text())
                if m:
                    z_md = float(m.group(1).replace(",", "") + "." + m.group(2))
                    d_z = abs(z_md - s["Z_online"])
                    assert d_z < 0.005, f"基线复现失败: {z_md} vs {s['Z_online']}"
                    lines.append(f"交付基线复现核对：Z^online = {z_md:,.2f} 元"
                                 f"（差 {d_z:.2f}）")
            row = {"arm": arm, "days": days or n,
                   **{k: s[k] for k, _, _ in DOWN_KEYS}}
            done[arm] = row
            pd.DataFrame([row]).to_csv(cache, mode="a",
                                       header=not cache.exists(), index=False)
        else:
            lines.append(f"缓存命中 downstream.csv: {arm}")

    # 汇总表：固定臂为基准列，各自适应臂给"数值 (Δ%)"
    base = done["L1w4-P1w7"]
    deltas = {arm: (done[arm]["Z_online"] - base["Z_online"]) / base["Z_online"]
              for arm in DOWN_ARMS[1:]}
    rows = []
    for key, label, fmt in DOWN_KEYS:
        cells = [fmt.format(base[key])]
        for arm in DOWN_ARMS[1:]:
            v0, v1 = base[key], done[arm][key]
            d = (v1 - v0) / v0 if v0 else 0.0
            cells.append(f"{fmt.format(v1)} ({d:+.2%})")
        rows.append((label, *cells))
    md = md_table(("指标", "固定 4/7", "仅负载 W_L", "仅光伏 W_R", "双自适应"),
                  rows)
    return md, lines, deltas


def q3_forecast_compare(year: YearData, fc: YearForecast,
                        typ_r: np.ndarray, schedule: pd.DataFrame
                        ) -> tuple[str, str]:
    """题3 融合层对照：持续法窗固定 7 vs 逐日 w_r——融合权重（σ 估计）
    与点预报同步换窗，保持"谁历史准听谁"的自洽。W_R 在全日曲线上选
    （P1 族口径），融合持续法为其时段切片，判据为逐阶段预报段 MAE。"""
    r_hour = hour_powers(year.r)
    cfg3 = Q3Config()
    n = len(year.dates)
    mae = np.empty((n, 4, 2))
    for j in range(n):
        w = int(schedule["w_r"][j])
        wt_f = fusion_weights(r_hour, fc.phi, j)
        wt_a = fusion_weights(r_hour, fc.phi, j, window=w)
        for m in range(4):
            h0 = 36 * m
            act = year.r[j][h0:]
            r_f = pv_point(cfg3, year, fc, j, m, typ_r, wt_f)[h0:]
            r_a = pv_point(replace(cfg3, pv_window=w), year, fc, j, m,
                           typ_r, wt_a)[h0:]
            mae[j, m, 0] = np.abs(r_f - act).mean()
            mae[j, m, 1] = np.abs(r_a - act).mean()

    # 报告期聚合：逐阶段（发布后 6h 块）与全天
    rep = slice(WARMUP_DAYS, n)
    rows = []
    for m in range(4):
        v0 = float(mae[rep, m, 0].mean())
        v1 = float(mae[rep, m, 1].mean())
        rows.append((f"阶段 m={m}（{6 * m}:00 起）", f"{v0:.3f}", f"{v1:.3f}",
                     f"{(v1 / v0 - 1) * 100:+.2f}%"))
    v0 = float(mae[rep, :, 0].mean())
    v1 = float(mae[rep, :, 1].mean())
    rows.append(("全阶段平均", f"{v0:.3f}", f"{v1:.3f}", f"{(v1 / v0 - 1) * 100:+.2f}%"))
    verdict = f"{v1 / v0 - 1:+.2%}"
    return md_table(("口径", "固定 W_R=7", "逐日选窗", "Δ"), rows), verdict


def write_summary(out_dir: Path, schedule: pd.DataFrame, skill_md: str,
                  downstream_md: str | None,
                  deltas: dict[str, float] | None, q3_md: str | None,
                  q3_verdict: str | None, checks: list[str]) -> Path:
    """实验总览 summary.md：配置、三段结果与 issue #6 验收对照。"""
    rep = schedule.iloc[WARMUP_DAYS:]
    l_adapt = int((rep["w_l"] != 4).sum())
    r_adapt = int((rep["w_r"] != 7).sum())
    parts = [
        "# 去超参实验（issue #6）：窗口 W_L=4 / W_R=7 改滚动伪样本外选择",
        "",
        "## 配置",
        "",
        f"- 候选网格：W_L ∈ {list(LOAD_GRID)}（同周几步），W_R ∈ {list(PV_GRID)}",
        "- 选择规则：伪样本外 MAE（公共评估集 = 全候选完整窗的过去日），"
        "1-SE 带内取最小窗；评估日不足 21 退回先验 4/7",
        f"- 报告期（j ≥ {WARMUP_DAYS}）偏离先验天数：负载 {l_adapt}/{len(rep)}，"
        f"光伏 {r_adapt}/{len(rep)}",
        "",
        "## 双实现交叉验证",
        "",
        *[f"- {line}" for line in checks],
        "",
        "## 预报层技能（五臂）",
        "",
        skill_md or "（未运行 --part forecast）",
    ]
    if downstream_md:
        parts += ["", "## 题2 全年费用承重（四臂 rollout）", "", downstream_md]
    if q3_md:
        parts += ["", "## 题3 融合层光伏对照", "", q3_md]

    # issue #6 验收对照（事实陈述，判定供讨论）
    rows = [
        ("W_L/W_R 不再写死（实验链）", "rollout 支持 win_schedule 逐日选窗；"
         "交付链采纳待议"),
        ("选择只用历史（伪样本外）", "结构成立 + 曲面/选择双对拍通过"),
        ("全年回测不劣于固定 4/7",
         f"Z^online Δ：仅负载 {deltas['L1ad-P1w7']:+.2%} / "
         f"仅光伏 {deltas['L1w4-P1ad']:+.2%} / "
         f"双自适应 {deltas['L1ad-P1ad']:+.2%}" if deltas else "（未运行）"),
        ("逐日选窗序列诊断输出", "windows.csv / mae_surface_*.csv / "
         "forecast_daily.csv"),
    ] if deltas else [
        ("全年回测不劣于固定 4/7", "（未运行 downstream）"),
    ]
    parts += ["", "## issue #6 验收对照", "",
              md_table(("验收项", "事实"), rows)]
    if q3_verdict:
        parts += ["", f"题3 融合层全阶段平均 Δ = {q3_verdict}（固定 7 为基准）"]
    path = out_dir / "summary.md"
    path.write_text("\n".join(parts) + "\n")
    return path
