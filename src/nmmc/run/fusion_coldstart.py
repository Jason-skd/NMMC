"""融合冷启动实验：删 min_days=8 人工阈值的年初段诊断（issue #8 去超参⑤）。

背景：反方差融合原以 min_days=8 护栏划分冷启动——不足 8 天官方直用，
第 9 天起突然全融合。方差估计量无偏，样本少带来的是噪声不是偏差；
删阈值改恒融合后，信任度由样本量自然决定：j ∈ {0, 1} 无可用样本
（day 0 无持续法）自动全 φ = 1 退化为官方直用，第 2 日起自由度 ≥ 2
的提前期先行融合，权重噪声随样本累积自平均。

Part A 权重与预报质量诊断（纯 numpy，秒级）——
    A1 逐日冷启动权重 φ_j(h)，j = 0..view_days−1；可用样本数闭式
       n_j(h) = (j−1)⁺·v(h)，v(h) = 覆盖该提前期的 vintage 数
       （h ∈ 1–6 / 7–12 / 13–18 / 19–24 → v = 4 / 3 / 2 / 1）
    A2 受影响日（j = 2..7，旧规则官方直用 vs 新规则融合）四阶段
       点预报 MAE 对照；j = 8..13 两规则一致，作控制组
Part B 全年回测承重检验（重）——交付配置全年 rollout 对照 main 基线
    （results/q3/F1-L1-LU1-S-b/q3_details.csv）：报告期 Z/Σg/紧急费
    逐项对照 + 逐日 Δz；暖机期逐日 z 水平表（受影响日前 8 天全在
    31 天暖机内，2/1 起仅 SOC 链携带微扰）。
判据：报告期 |ΔZ|/Z(基线) 在噪声量级、年初段无异常值 ⇒ 阈值删除成立。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data.fcst import YearForecast
from nmmc.data.year import YearData
from nmmc.model.forecast3 import (Q3Config, fusion_weights, hour_powers,
                                  pv_point)
from nmmc.model.lp import T
from nmmc.model.q2 import KAPPA
from nmmc.model.q3 import VINTAGE_HOURS
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.run.q3 import DayResult, rollout

AFFECTED = range(2, 8)     # 旧规则官方直用、新规则开始融合的日索引
CONTROL = range(8, 14)     # 两规则一致的对照日索引（j ≥ 8 样本即满 8 天）


def lead_vintages() -> np.ndarray:
    """逐提前期覆盖的 vintage 数 v(h)：h ∈ 1–6/7–12/13–18/19–24 → 4/3/2/1。"""
    return np.array([sum(tau + h <= 24 for tau in VINTAGE_HOURS)
                     for h in range(1, 25)], float)


def weights_table(year: YearData, fc: YearForecast,
                  view_days: int) -> pd.DataFrame:
    """A1：逐日冷启动融合权重宽表（行 j，列 h01..h24，值 φ_j(h)）。"""
    r_hour = hour_powers(year.r)
    rows = []
    for j in range(view_days):
        w = fusion_weights(r_hour, fc.phi, j)
        rows.append({"j": j, **{f"h{h:02d}": w[h - 1] for h in range(1, 25)}})
    return pd.DataFrame(rows)


def point_quality(cfg: Q3Config, year: YearData, fc: YearForecast,
                  typ_r: np.ndarray) -> pd.DataFrame:
    """A2：受影响日与对照日的光伏点预报质量（只对预报段 t ≥ 6τ_m 计误差）。

    旧规则冷启动语义（官方直用）以全 1 权重复现——pv_point 在 φ ≡ 1
    时数值上严格等于官方降尺度直用。
    """
    r_hour = hour_powers(year.r)
    ones = np.ones(24)
    rows = []
    for j in list(AFFECTED) + list(CONTROL):
        w = fusion_weights(r_hour, fc.phi, j)
        errs = {"official": 0.0, "fused": 0.0}
        for m in range(4):
            seg = slice(36 * m, T)
            actual = year.r[j][seg]
            for kind, weights in (("official", ones), ("fused", w)):
                r_hat = pv_point(cfg, year, fc, j, m, typ_r, weights)
                errs[kind] += float(np.abs(r_hat[seg] - actual).mean())
        rows.append({"j": j,
                     "group": "affected" if j in AFFECTED else "control",
                     "mae_official": errs["official"] / 4,
                     "mae_fused": errs["fused"] / 4})
    return pd.DataFrame(rows)


def backtest_compare(year: YearData, fc: YearForecast, p: np.ndarray,
                     typ_ell: np.ndarray, typ_r: np.ndarray, cfg: Q3Config,
                     baseline: Path,
                     days: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part B：新代码全年 rollout 对照 main 基线（报告期逐日 + 暖机段）。

    Args:
        year / fc / p / typ_ell / typ_r / cfg / days: 同 rollout。
        baseline: main 基线明细 CSV（q3_details.csv，报告期逐日现金口径）。

    Returns:
        (报告期逐日对照表，暖机期逐日水平表)。
    """
    res: list[DayResult] = rollout(year, fc, p, typ_ell, typ_r, cfg, days=days)
    base = pd.read_csv(baseline)

    # 报告期逐日对照：新旧现金费用 z、计划电量与紧急费同口径对齐
    rows = [{
        "date": str(year.dates[d.j]), "z_new": d.z,
        "g_new": float(d.g.sum()),
        "emerg_new": float(KAPPA * (p * d.u).sum()),
        "soc_new": float(d.S.mean()),
    } for d in res if d.j >= WARMUP_DAYS]
    daily = (pd.DataFrame(rows)
             .merge(base[["date", "z_online", "g_sum", "emerg_fee"]],
                    on="date", how="left")
             .rename(columns={"z_online": "z_base", "g_sum": "g_base",
                              "emerg_fee": "emerg_base"}))
    daily["dz"] = daily["z_new"] - daily["z_base"]

    # 暖机期逐日水平（前 10 天）：受影响日集中在此段，观察有无异常值
    warmup = pd.DataFrame([{
        "j": d.j, "date": str(year.dates[d.j]), "z": d.z,
        "soc_end": float(d.S[T]),
    } for d in res if d.j < min(10, len(res))])
    return daily, warmup


def write_summary(out_dir: Path, weights: pd.DataFrame | None,
                  quality: pd.DataFrame | None,
                  daily: pd.DataFrame | None,
                  warmup: pd.DataFrame | None) -> Path:
    """汇总 summary.md：A 权重/质量表 + B 回测对照表 + 判据结论。"""
    v = lead_vintages()
    lines = [
        "# 融合冷启动实验（删 min_days=8 阈值，恒融合）",
        "",
        "- 机制：j ∈ {0, 1} 无可用样本自动 φ ≡ 1（官方直用）；第 2 日起",
        "  自由度 ≥ 2 的提前期融合",
        f"- 样本数闭式 n_j(h) = (j−1)⁺·v(h)，v = "
        f"{v[0]:g}/{v[6]:g}/{v[12]:g}/{v[18]:g}"
        "（h ∈ 1–6 / 7–12 / 13–18 / 19–24）",
    ]
    if weights is not None:
        # 代表提前期 h = 1/6/12/18/24 与已融合提前期数（φ<1）逐日展开
        lines += ["", "## Part A1｜逐日冷启动权重 φ_j(h)"
                  "（全表见 coldstart_weights.csv）", "",
                  "| j | φ(h=1) | φ(h=6) | φ(h=12) | φ(h=18) | φ(h=24)"
                  " | 融合提前期数 |", "|---|---|---|---|---|---|---|"]
        for _, r in weights.iterrows():
            fused = int((r[[f"h{h:02d}" for h in range(1, 25)]] < 1.0).sum())
            lines.append(
                f"| {int(r['j'])} | {r['h01']:.3f} | {r['h06']:.3f}"
                f" | {r['h12']:.3f} | {r['h18']:.3f} | {r['h24']:.3f}"
                f" | {fused}/24 |")
    if quality is not None:
        lines += ["", "## Part A2｜点预报 MAE 对照（kWh/时段，四阶段平均）",
                  "", "| j | 组 | 官方直用 | 融合 | Δ |", "|---|---|---|---|---|"]
        for _, r in quality.iterrows():
            d_pct = (r["mae_fused"] / r["mae_official"] - 1) * 100
            lines.append(f"| {int(r['j'])} | {r['group']} "
                         f"| {r['mae_official']:.2f} | {r['mae_fused']:.2f}"
                         f" | {d_pct:+.1f}% |")
    if daily is not None:
        m = daily.dropna(subset=["z_base"])
        z_b, z_n = float(m["z_base"].sum()), float(m["z_new"].sum())
        g_b, g_n = float(m["g_base"].sum()), float(m["g_new"].sum())
        e_b, e_n = float(m["emerg_base"].sum()), float(m["emerg_new"].sum())
        i_max = m["dz"].abs().idxmax()
        worst = m.loc[i_max]
        lines += ["", f"## Part B｜全年回测对照（报告期 {len(m)} 天，"
                  "基线 = main results/q3）", "",
                  "| 指标 | 基线（min_days=8） | 恒融合 | Δ | Δ% |",
                  "|---|---|---|---|---|",
                  f"| Z（元） | {z_b:,.0f} | {z_n:,.0f} | {z_n - z_b:+,.0f}"
                  f" | {(z_n / z_b - 1) * 100:+.3f}% |",
                  f"| Σg（kWh） | {g_b:,.0f} | {g_n:,.0f} | {g_n - g_b:+,.0f}"
                  f" | {(g_n / g_b - 1) * 100:+.3f}% |",
                  f"| 紧急费（元） | {e_b:,.0f} | {e_n:,.0f}"
                  f" | {e_n - e_b:+,.0f} | {(e_n / e_b - 1) * 100:+.3f}% |",
                  "",
                  "## 判据", "",
                  f"- 报告期 ΔZ/Z(基线) = **{(z_n / z_b - 1) * 100:+.3f}%**",
                  f"- 逐日 |Δz| 最大 = {abs(worst['dz']):,.0f} 元"
                  f"（{worst['date']}，占当日 z 的 "
                  f"{abs(worst['dz']) / worst['z_base'] * 100:.2f}%）"
                  "——SOC 链携带的微扰量级",
                  f"- 暖机期日 z 范围 [{warmup['z'].min():,.0f},"
                  f"{warmup['z'].max():,.0f}] 元"
                  f"（报告期日 z 范围 [{m['z_base'].min():,.0f},"
                  f"{m['z_base'].max():,.0f}] 元，同量级即无异常）",
                  "",
                  ]
        if warmup is not None:
            lines += ["", "## 暖机期逐日 z（前 10 天，前 8 天为受影响日）",
                      "", "| j | 日期 | z（元） | 日末 SOC |",
                      "|---|---|---|---|"]
            for _, r in warmup.iterrows():
                lines.append(f"| {int(r['j'])} | {r['date']} | {r['z']:,.0f}"
                             f" | {r['soc_end']:,.0f} |")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
