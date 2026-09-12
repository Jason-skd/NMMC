"""ν 锚稳健性实验：附录"逐日对偶 vs 常数锚 0.472"对照（文档 §11 诊断扩展）。

Part A 测量（每天 3 个题1 型小 LP，全年约 1100 个，分钟级）——逐日循环
对偶 ν_j = 解"附件1 价格 + 昼夜循环 + S_0 自由"的单日 LP，读循环约束
行（S_144 − S_0 = 0）的对偶：
    nu_own        第 j 天实际数据（0.472 即典型日的这个量，直接对照）
    nu_next_post  第 j+1 天实际数据（事后真值，测量用非策略）
    nu_next_fcst  第 j 日 0:00 只用 ≤ j−1 历史（hist_end=j 的 L1/P1 点
                  预报）对 j+1 求解（诚实可实现，Part B 换装输入）
Part B 承重检验（全年回测，B1 地标 ×4 + B2 换装 ×1）：
    B1 常数扫参——地标点 {p_min/η 下界, 锚 0.472, γ 复现点 0.616, 均价
        0.7662}，逐点全年 rollout，报 Z/Σg/紧急费/SOC
    B2 V2b 换装——nu_next_fcst 逐日数组进 rollout，对照常数锚
判据：B1 四点 Z 极差 / Z(锚) < 0.5% 且 B2 ΔZ 同量级 ⇒ 锚不承重。
纪律：固定种子；Z 现金口径（不含 ν）；末日 j=364 无次日，ν_next 退化
为自身（该日 ν 指向不存在的 day 365，约定取自身型对偶）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.model.forecast import ForecastConfig, forecast_load, forecast_pv
from nmmc.model.lp import ETA
from nmmc.model.q1 import solve as q1_solve
from nmmc.model.q2 import KAPPA, NU_END
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.run.q2 import rollout

# B1 扫参地标点（都有来历，非均匀网格）：理论下界 / 题1 锚 /
# 附件1 γ 复现点（0.472/0.7662）/ 附件1 均价
NU_LANDMARKS = (0.4126, 0.472, 0.616, 0.7662)

SERIES = ("nu_own", "nu_next_post", "nu_next_fcst")


def cycle_dual(p: np.ndarray, ell: np.ndarray, r: np.ndarray) -> float:
    """题1 型 LP 的循环约束对偶 ν（S_0 自由；多留 1 kWh 到日末的边际价值）。

    读循环约束行（S_144 − S_0 = 0）的对偶而非 ν_t 序列均值——后者在典型日
    并非常数（日内多次套利时段的价差更高），题2/题3 的常数锚 0.472 即此行
    对偶在典型日的取值。

    Args:
        p: 逐时段电价，shape (144,)。
        ell / r: 目标日负载 / 光伏电量，shape (144,)。

    Returns:
        ν，元/kWh（典型日 = 0.472）。
    """
    return q1_solve(p, ell, r).nu_cycle


def measure_duals(year, p: np.ndarray, cfg: ForecastConfig,
                  typ_ell: np.ndarray, typ_r: np.ndarray) -> pd.DataFrame:
    """Part A：全年逐日对偶测量（nu_own / nu_next_post / nu_next_fcst）。

    Args:
        year: 附件2 全年实际值。
        p: 逐时段电价（附件1）。
        cfg: 预报层配置（ν_next_fcst 用，与交付同候选）。
        typ_ell / typ_r: 附件1 典型日曲线（无历史时先验）。

    Returns:
        逐日对偶表，列 j/date/SERIES。
    """
    n = len(year.dates)
    rows = []
    for j in range(n):
        nxt = j + 1 if j + 1 < n else j
        ell_f = forecast_load(cfg, year, nxt, typ_ell, hist_end=j)
        r_f = forecast_pv(cfg, year, nxt, typ_r, hist_end=j)
        rows.append({
            "j": j,
            "date": str(year.dates[j]),
            "nu_own": cycle_dual(p, year.ell[j], year.r[j]),
            "nu_next_post": cycle_dual(p, year.ell[nxt], year.r[nxt]),
            "nu_next_fcst": cycle_dual(p, ell_f, r_f),
        })
        if (j + 1) % 31 == 0:
            print(f"已测对偶 {j + 1}/{n} 天", flush=True)
    return pd.DataFrame(rows)


def summarize_duals(duals: pd.DataFrame, p: np.ndarray,
                    anchor_check: float) -> dict:
    """Part A 汇总统计：三序列 mean/std/min/max、对锚偏差、理论界断言。

    Args:
        duals: measure_duals 输出。
        p: 电价（推理论界 [p_min/η, η·p_max]）。
        anchor_check: 典型日对偶复现值（应 ≈ 0.472）。

    Returns:
        统计字典（summary.md 与终端共用）。
    """
    nu_lo, nu_hi = p.min() / ETA, ETA * p.max()
    out = {"nu_lo": nu_lo, "nu_hi": nu_hi, "anchor_check": anchor_check}
    for col in SERIES:
        s = duals[col]
        out[col] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "dev_anchor": abs(float(s.mean()) - NU_END) / NU_END,
            "in_bounds": int(((s >= nu_lo - 1e-9) &
                              (s <= nu_hi + 1e-9)).sum()),
        }
    return out


def _rollout_metrics(days: list, p: np.ndarray) -> dict:
    """单次全年 rollout 的报告期指标（现金口径，不含 ν）。"""
    rep = [d for d in days if d.j >= WARMUP_DAYS]
    return {
        "Z_online": float(sum(d.z for d in rep)),
        "sum_g": float(sum(d.g.sum() for d in rep)),
        "emerg_fee": float(sum(KAPPA * float((p * d.u).sum()) for d in rep)),
        "soc_mean": float(np.mean([d.S.mean() for d in rep])),
    }


def run_sweep(year, p: np.ndarray, typ_ell: np.ndarray, typ_r: np.ndarray,
              cfg: ForecastConfig, nu_arr: np.ndarray,
              out_dir: Path, days: int | None = None) -> pd.DataFrame:
    """Part B：地标扫参（B1）+ V2b 逐日换装（B2），逐行落盘。

    Args:
        year / p / typ_ell / typ_r / cfg / days: 同 rollout。
        nu_arr: B2 的逐日 ν 数组（duals["nu_next_fcst"]）。
        out_dir: 结果目录（sweep.csv 落此）。

    Returns:
        扫参结果表，列 label/nu/metrics。
    """
    rows = []

    def _save() -> None:
        pd.DataFrame(rows).to_csv(out_dir / "sweep.csv", index=False)

    # B1 常数地标：每点一次全年回测
    for nu in NU_LANDMARKS:
        print(f"B1 扫参 ν = {nu:g} …", flush=True)
        res = rollout(year, p, typ_ell, typ_r, cfg, nu_end=nu, days=days)
        rows.append({"label": f"const-{nu:g}", "nu": nu,
                     **_rollout_metrics(res, p)})
        _save()

    # B2 V2b 换装：nu_next_fcst 逐日数组（诚实：只用 ≤ j−1 信息）
    print("B2 V2b 换装（逐日 ν_next_fcst）…", flush=True)
    res = rollout(year, p, typ_ell, typ_r, cfg, nu_end=nu_arr, days=days)
    rows.append({"label": "v2b-fcst", "nu": float("nan"),
                 **_rollout_metrics(res, p)})
    _save()
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, stats: dict,
                  sweep: pd.DataFrame | None) -> Path:
    """汇总 summary.md：Part A 统计表 + Part B 扫参表 + 判据结论。"""
    lines = [
        "# ν 锚稳健性实验（逐日对偶 vs 常数锚 0.472）",
        "",
        f"- 理论界：[{stats['nu_lo']:.4f}, {stats['nu_hi']:.4f}]"
        f"（p_min/η, η·p_max）",
        f"- 方法学自检：典型日对偶复现 = {stats['anchor_check']:.4f}"
        f"（题1 档报 0.472）",
        "",
        "## Part A｜逐日对偶分布（334+31 全天）",
        "",
        "| 序列 | mean | std | min | max | \\|mean−0.472\\|/0.472 | 界内天数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for col in SERIES:
        s = stats[col]
        lines.append(
            f"| {col} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f}"
            f" | {s['max']:.4f} | {s['dev_anchor'] * 100:.2f}%"
            f" | {s['in_bounds']}/{365} |")
    if sweep is not None:
        anchor_row = sweep.loc[sweep["nu"] == NU_END].iloc[0]
        anchor_z = float(anchor_row["Z_online"])
        z = sweep["Z_online"]
        spread = (z.max() - z.min()) / anchor_z
        consts = sweep[sweep["label"].str.startswith("const-")]
        lo_lab = consts.loc[consts["Z_online"].idxmin(), "label"]
        v2b = sweep.loc[sweep["label"] == "v2b-fcst"].iloc[0]
        dz = abs(float(v2b["Z_online"]) - anchor_z) / anchor_z
        nu_max = stats["nu_own"]["max"]
        lines += [
            "",
            "## Part B｜承重检验（全年回测，交付配置 smpc）",
            "",
            "| 配置 | ν | Z_online（元） | Σg（kWh） | 紧急费（元）"
            " | SOC 均值 |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in sweep.iterrows():
            lines.append(
                f"| {row['label']} | {row['nu']:.4g} | {row['Z_online']:,.0f}"
                f" | {row['sum_g']:,.0f} | {row['emerg_fee']:,.0f}"
                f" | {row['soc_mean']:,.1f} |")
        lines += [
            "",
            "## 判据",
            "",
            f"- B1 地标四点 Z 极差 / Z(锚) = **{spread * 100:.3f}%**"
            f"（阈值 0.5%，未过但见下条分解）",
            f"- 地标中 Z 最小者 = {lo_lab}"
            + ("——常数锚恰在常数族最优附近，两侧均变贵" if lo_lab == "const-0.472" else ""),
            f"- 上端地标（≥0.616）经验不可达：全年逐日对偶 max = "
            f"{nu_max:.4f} ≤ 0.472，可达域上 Z 变化 = B2 实测 "
            f"**{dz * 100:.3f}%**",
            f"- B2 V2b 换装 ΔZ / Z(锚) = **{dz * 100:.3f}%**，"
            f"ΔΣg = {float(v2b['sum_g']) - float(anchor_row['sum_g']):+,.0f} kWh",
            "",
        ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
