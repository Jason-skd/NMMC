"""题3 诊断：把全部题2↔题3 横向对照重锚到题2 v6 口径（u ≤ (n−g)⁺）。

只读 results/ 下的逐日明细 CSV，不重跑模型。输出诊断文档所需的全部数字。
用法：.venv/bin/python experiments/q3/rebaseline_v6.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
Q3 = ROOT / "results/q3/F1-L1-LU1-S-b/q3_details.csv"
Q2_V6 = ROOT / "results/q2_v6/L1-P1-R2-smpc/q2_details.csv"
Q2_V5 = ROOT / "results/q2_v6/L1-P1-R2-smpc-nocap/q2_details.csv"
PAPER = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
NU_END = 0.472


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q3 = pd.read_csv(Q3, parse_dates=["date"])
    v6 = pd.read_csv(Q2_V6, parse_dates=["date"])
    v5 = pd.read_csv(Q2_V5, parse_dates=["date"])
    for name, df in (("q3", q3), ("v6", v6), ("v5", v5)):
        assert len(df) == 334, f"{name} 报告期 {len(df)} 天，应为 334"
    assert (q3.date.values == v6.date.values).all(), "日期未对齐"
    return q3, v6, v5


def money(x: float) -> str:
    return f"{x:,.2f}"


def main() -> None:
    q3, v6, v5 = load()
    print("=" * 72)
    print("1. 纵向倒序：题3 vs 题2 v6（新基准）与 v5（旧基准）")
    print("=" * 72)
    for tag, q2 in (("v6 (u≤(n−g)⁺, 定稿)", v6), ("v5 (u 无上界, 旧档)", v5)):
        d = q3.z_online.values - q2.z_online.values
        print(f"\n-- 题2 {tag}")
        print(f"   Z^Q2      = {money(q2.z_online.sum())}")
        print(f"   Z^Q3      = {money(q3.z_online.sum())}")
        print(f"   ΔZ        = {money(d.sum())}  "
              f"({d.sum() / q2.z_online.sum() * 100:+.2f}%)")
        print(f"   题3更贵天数 = {(d > 0).sum()} / {len(d)}")
        print(f"   逐日 ΔZ:  中位 {np.median(d):+,.1f}  "
              f"p10 {np.percentile(d, 10):+,.1f}  p90 {np.percentile(d, 90):+,.1f}  "
              f"最坏 {d.max():+,.1f} ({q3.date[d.argmax()].date()})  "
              f"最好 {d.min():+,.1f} ({q3.date[d.argmin()].date()})")
        print(f"   题3更贵日合计 {money(d[d > 0].sum())} / "
              f"题3更省日合计 {money(d[d < 0].sum())}")
        # 年末 SOC 修正（C3-16）
        soc_gap = q2.S144.values[-1] - q3.S144.values[-1]
        print(f"   年末 SOC: 题2 {q2.S144.values[-1]:,.0f} vs 题3 "
              f"{q3.S144.values[-1]:,.0f} kWh；ν^end 补差 "
              f"{money(soc_gap * NU_END)}")
        print(f"   SOC 修正后 ΔZ = {money(d.sum() + soc_gap * NU_END)}")

    print("\n" + "=" * 72)
    print("2. 费用结构（题3 三块 vs 题2 v6 两块）")
    print("=" * 72)
    rows = [
        ("基础购电费 Σp·a（题3）/ Σp·g（题2）",
         q3.base_fee.sum(), v6.z_online.sum() - v6.emerg_fee.sum()),
        ("偏差费 ½Σp|a−g|", q3.dev_fee.sum(), 0.0),
        ("紧急购电费 κΣp·u", q3.emerg_fee.sum(), v6.emerg_fee.sum()),
        ("合计 Z", q3.z_online.sum(), v6.z_online.sum()),
    ]
    for name, a, b in rows:
        print(f"   {name:38s} 题3 {money(a):>16s}   题2v6 {money(b):>16s}   "
              f"Δ {money(a - b):>14s}")
    print(f"   偏差费/紧急费省额 = {money(q3.dev_fee.sum())} / "
          f"{money(v6.emerg_fee.sum() - q3.emerg_fee.sum())}  → 净亏 "
          f"{money(q3.dev_fee.sum() - (v6.emerg_fee.sum() - q3.emerg_fee.sum()))}")
    print(f"   购电量：题3 Σa {q3.a_sum.sum():,.0f} kWh / Σg {q3.g_sum.sum():,.0f}；"
          f"题2v6 Σg {v6.g_sum.sum():,.0f} kWh")

    print("\n" + "=" * 72)
    print("3. 下调 δ− 主导（性质9 / C3-8）")
    print("=" * 72)
    dp, dm = q3.dp_sum.values, q3.dm_sum.values
    print(f"   Σδ+ = {dp.sum():,.1f} kWh   Σδ− = {dm.sum():,.1f} kWh   "
          f"δ+/δ− = {dp.sum() / dm.sum():.4f}")
    print(f"   有下调天数 {(dm > 1e-6).sum()}/334；有上调天数 {(dp > 1e-6).sum()}/334")
    print(f"   逐日 δ−/(Σg) 中位 {np.median(dm / q3.g_sum) * 100:.2f}%  "
          f"p90 {np.percentile(dm / q3.g_sum, 90) * 100:.2f}%")
    print(f"   过量计划 Σ(g−a) = {(q3.g_sum - q3.a_sum).sum():,.0f} kWh "
          f"= Σg 的 {(q3.g_sum - q3.a_sum).sum() / q3.g_sum.sum() * 100:.2f}%")
    # 空仓砍单：日末 SOC 低却仍在砍
    lowsoc = q3.S144.values < 3000
    print(f"   日末 SOC<3000 的 {lowsoc.sum()} 天里，仍有下调的 "
          f"{(lowsoc & (dm > 1e-6)).sum()} 天，Σδ− = {dm[lowsoc].sum():,.0f} kWh")

    print("\n" + "=" * 72)
    print("4. 隔夜库存（问题三），题2 用 v6")
    print("=" * 72)
    for tag, df in (("题2 v6", v6), ("题2 v5", v5), ("题3", q3)):
        print(f"   {tag:8s} 日初均值 {df.S0.mean():8,.0f}  日末均值 "
              f"{df.S144.mean():8,.0f}  日末最低 {df.S144.min():8,.0f}  "
              f"首日初 {df.S0.values[0]:,.0f} → 末日末 {df.S144.values[-1]:,.0f}  "
              f"净变化 {df.S144.values[-1] - df.S0.values[0]:+,.0f}")
    print(f"   题2v6 日初 > 题3 日初的天数：{(v6.S0.values > q3.S0.values).sum()}/334")

    print("\n" + "=" * 72)
    print("5. 紧急购电（题2 用 v6）")
    print("=" * 72)
    for tag, df in (("题2 v6", v6), ("题2 v5", v5), ("题3", q3)):
        u = df.u_sum.values
        print(f"   {tag:8s} Σu {u.sum():10,.0f} kWh  紧急费 "
              f"{money(df.emerg_fee.sum()):>14s}  发生天数 {(u > 1e-6).sum():3d}/334")
    print(f"   题3/题2v6 电量比 {q3.u_sum.sum() / v6.u_sum.sum():.3f}；"
          f"费用比 {q3.emerg_fee.sum() / v6.emerg_fee.sum():.3f}")

    print("\n" + "=" * 72)
    print("6. 下界与 C9（跨题可比性）")
    print("=" * 72)
    for tag, df in (("题2 v6", v6), ("题3", q3)):
        print(f"   {tag:8s} Z^PI {money(df.z_pi.sum()):>16s}  Z^ex-post "
              f"{money(df.z_expost.sum()):>16s}  Z^online−Z^ex-post "
              f"{money(df.z_online.sum() - df.z_expost.sum()):>14s}")
    print(f"   Z^PI 跨题差 {money(q3.z_pi.sum() - v6.z_pi.sum())}（应为 0）")
    r = np.corrcoef(q3.z_pi.values - v6.z_pi.values,
                    q3.S0.values - v6.S0.values)[0, 1]
    print(f"   corr(ΔZ^PI, ΔS0) = {r:+.3f}  → PI 被日初 SOC 链污染")
    print(f"   题3 实时信息损失 Z^online−Z^ex-post = "
          f"{money(q3.z_online.sum() - q3.z_expost.sum())}"
          f"（题2v6 {money(v6.z_online.sum() - v6.z_expost.sum())}）")

    print("\n" + "=" * 72)
    print("7. 四个指定日（题2 用 v6）")
    print("=" * 72)
    hdr = (f"   {'日期':<12}{'题2v6 Z':>12}{'题3 Z':>12}{'ΔZ':>11}"
           f"{'题3 S0':>9}{'题2 S0':>9}{'δ−':>10}{'δ+':>8}{'题3 u':>9}{'题2 u':>9}")
    print(hdr)
    for date in PAPER:
        i = q3.index[q3.date == date][0]
        print(f"   {date:<12}{v6.z_online[i]:>12,.0f}{q3.z_online[i]:>12,.0f}"
              f"{q3.z_online[i] - v6.z_online[i]:>+11,.0f}{q3.S0[i]:>9,.0f}"
              f"{v6.S0[i]:>9,.0f}{q3.dm_sum[i]:>10,.0f}{q3.dp_sum[i]:>8,.0f}"
              f"{q3.u_sum[i]:>9,.0f}{v6.u_sum[i]:>9,.0f}")

    print("\n" + "=" * 72)
    print("8. 预报技能（口径核对）")
    print("=" * 72)
    for tag, df in (("题2 v6", v6), ("题3", q3)):
        print(f"   {tag:8s} 负载 MAE {df.mae_ell.mean():.2f}  "
              f"光伏 MAE(0:00 点预报) {df.mae_r.mean():.2f} kWh/时段")

    print("\n" + "=" * 72)
    print("9. ΔZ 与什么相关（归因排序）")
    print("=" * 72)
    d = q3.z_online.values - v6.z_online.values
    for name, x in (("题3 偏差费", q3.dev_fee.values),
                    ("题3 Σδ−", q3.dm_sum.values),
                    ("题3 日初 SOC 劣势 (S0_q2−S0_q3)",
                     v6.S0.values - q3.S0.values),
                    ("题3 弃电量", q3.w_sum.values),
                    ("题3 紧急量", q3.u_sum.values),
                    ("题2 紧急量", v6.u_sum.values),
                    ("当日 Σ净负荷代理 (a_sum)", q3.a_sum.values)):
        print(f"   corr(ΔZ, {name:34s}) = "
              f"{np.corrcoef(d, x)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
