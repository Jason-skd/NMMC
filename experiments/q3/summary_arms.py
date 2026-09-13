"""题3 诊断：把交付臂、题2 v6/v5 与全部反事实臂摆到同一张表上。

各臂口径声明（读表前必看）：
  Q2-v6 / Q2-v5   题2 交付 / 口径消融（u 有 / 无上界）。
  Q3-交付          仓库 results/q3/F1-L1-LU1-S-b，阶段0 LP 带别名缺陷。
  CF-A / CF-B      0:00 计划换成题2 两阶段 SAA（非预期性）；A 行使 6/12/18
                   改单，B 冻结 a ≡ g。
  CF-D / CF-E      0:00 计划换成修正变量布局的同款阶段0 LP；D 改单，E 冻结。
  E0（仓库）        results/q3-epochs/E0，= 缺陷计划 + 冻结（稻草人对照）。
全部 CF 臂与 Q3-交付 共用同一预报层、情景器、内环随机 MPC、SOC 链与 rng 流，
u 一律无上界（v5 口径）——差别只在"0:00 那张 LP"与"是否改单"。

用法：.venv/bin/python experiments/q3/summary_arms.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results/q3-diag"
WARMUP = 31
NU_END = 0.472

ARMS = (
    ("Q2-v6", ROOT / "results/q2_v6/L1-P1-R2-smpc/q2_details.csv", "q2"),
    ("Q2-v5", ROOT / "results/q2_v6/L1-P1-R2-smpc-nocap/q2_details.csv", "q2"),
    ("Q3-交付", ROOT / "results/q3/F1-L1-LU1-S-b/q3_details.csv", "q3"),
    ("CF-A 题2计划+改单", DIAG / "cfA.csv", "cf"),
    ("CF-B 题2计划+不改单", DIAG / "cfB.csv", "cf"),
    ("CF-D 修正计划+改单", DIAG / "cfD.csv", "cf"),
    ("CF-E 修正计划+不改单", DIAG / "cfE.csv", "cf"),
)


def load(path: Path, kind: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if kind == "cf":
        df = df.iloc[WARMUP:].reset_index(drop=True)
    for col in ("dev_fee", "dp_sum", "dm_sum", "a_sum", "base_fee",
                "night_c", "night_d"):
        if col not in df:
            df[col] = 0.0 if col != "a_sum" else df["g_sum"]
    if kind == "q2":
        df["base_fee"] = df.z_online - df.emerg_fee
    return df


def main() -> None:
    tab = {}
    for name, path, kind in ARMS:
        df = load(path, kind)
        if df is None:
            print(f"（缺 {path.name}，跳过 {name}）")
            continue
        assert len(df) == 334, f"{name} 报告期 {len(df)} 天"
        tab[name] = df

    ref = tab["Q2-v6"].z_online.sum()
    q3 = tab["Q3-交付"].z_online.sum()
    print("=" * 104)
    print("全臂对照（报告期 334 天，元）")
    print("=" * 104)
    print(f"   {'臂':<22}{'Z^online':>14}{'vs Q2-v6':>12}{'vs Q3交付':>12}"
          f"{'基础购电':>13}{'偏差费':>11}{'紧急费':>11}{'Z^ex-post':>14}")
    for name, df in tab.items():
        z = df.z_online.sum()
        print(f"   {name:<22}{z:>14,.0f}{z - ref:>+12,.0f}{z - q3:>+12,.0f}"
              f"{df.base_fee.sum():>13,.0f}{df.dev_fee.sum():>11,.0f}"
              f"{df.emerg_fee.sum():>11,.0f}{df.z_expost.sum():>14,.0f}")

    print("\n" + "=" * 104)
    print("电量、调整量与库存")
    print("=" * 104)
    print(f"   {'臂':<22}{'Σg':>13}{'Σa':>13}{'Σ(g−a)/Σg':>11}{'Σδ+':>11}"
          f"{'Σδ−':>12}{'δ+/δ−':>8}{'Σu':>10}{'Σw':>12}")
    for name, df in tab.items():
        og = (df.g_sum.sum() - df.a_sum.sum()) / df.g_sum.sum()
        r = (df.dp_sum.sum() / df.dm_sum.sum()
             if df.dm_sum.sum() > 1e-6 else np.nan)
        print(f"   {name:<22}{df.g_sum.sum():>13,.0f}{df.a_sum.sum():>13,.0f}"
              f"{og:>10.1%}{df.dp_sum.sum():>11,.0f}{df.dm_sum.sum():>12,.0f}"
              f"{r:>8.3f}{df.u_sum.sum():>10,.0f}{df.w_sum.sum():>12,.0f}")

    print("\n" + "=" * 104)
    print("隔夜库存（保险指纹）与夜间行为")
    print("=" * 104)
    print(f"   {'臂':<22}{'日初均值':>10}{'日末均值':>10}{'日末最低':>10}"
          f"{'末日末':>10}{'净变化':>10}{'0-4h充':>9}{'0-4h放':>9}"
          f"{'夜净':>9}{'紧急天数':>9}")
    for name, df in tab.items():
        nc = df.night_c.mean() if df.night_c.sum() else np.nan
        nd = df.night_d.mean() if df.night_d.sum() else np.nan
        print(f"   {name:<22}{df.S0.mean():>10,.0f}{df.S144.mean():>10,.0f}"
              f"{df.S144.min():>10,.0f}{df.S144.values[-1]:>10,.0f}"
              f"{df.S144.values[-1] - df.S0.values[0]:>+10,.0f}"
              f"{nc:>9,.0f}{nd:>9,.0f}{nc - nd:>+9,.0f}"
              f"{(df.u_sum > 1e-6).sum():>9d}")

    print("\n" + "=" * 104)
    print("逐日胜负（对 Q2-v6；ν^end 年末 SOC 修正后）")
    print("=" * 104)
    v6 = tab["Q2-v6"]
    for name, df in tab.items():
        if name.startswith("Q2"):
            continue
        d = df.z_online.values - v6.z_online.values
        corr = (v6.S144.values[-1] - df.S144.values[-1]) * NU_END
        print(f"   {name:<22} ΔZ {d.sum():>+11,.0f}  更贵 {(d > 0).sum():3d}/334"
              f"  中位 {np.median(d):>+8,.0f}  SOC 修正后 {d.sum() + corr:>+11,.0f}")
    print("\n   关键差值：")
    if "CF-B 题2计划+不改单" in tab and "CF-A 题2计划+改单" in tab:
        b = tab["CF-B 题2计划+不改单"].z_online.sum()
        a = tab["CF-A 题2计划+改单"].z_online.sum()
        print(f"     改单权在非预期性计划下的价值 = CF-B − CF-A = "
              f"{b - a:+,.0f} 元（正=改单有用）")
        print(f"     别名缺陷的全年代价（下界）= Q3交付 − CF-B = "
              f"{q3 - b:+,.0f} 元（{(q3 - b) / b:+.2%}）")
    if "CF-D 修正计划+改单" in tab and "CF-E 修正计划+不改单" in tab:
        dd = tab["CF-D 修正计划+改单"].z_online.sum()
        ee = tab["CF-E 修正计划+不改单"].z_online.sum()
        print(f"     修正计划下改单权的价值 = CF-E − CF-D = {ee - dd:+,.0f} 元")
        print(f"     别名缺陷的全年代价（同计划器口径）= Q3交付 − CF-D = "
              f"{q3 - dd:+,.0f} 元（{(q3 - dd) / dd:+.2%}）")
        print(f"     修正后是否仍纵向倒序：CF-D − Q2v6 = {dd - ref:+,.0f} 元 "
              f"→ {'仍倒序' if dd > ref else '倒序消失'}")


if __name__ == "__main__":
    main()
