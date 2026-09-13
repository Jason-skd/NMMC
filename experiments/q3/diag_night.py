"""题3 诊断：夜间 B_0 的块内错配与紧急购电、弃电的分类计量（细粒度）。

要解开的表面矛盾：B_0（0:00–6:00）整块 Σg 比题2 高 26%，SOC 在 0:00→6:00
净增 +4,875 kWh，可 0:00–4:00 却是净放电、且 45% 的紧急电量落在 B_0 的谷价
时段。若整块买得更多，为什么前半夜还要抽电池、还要按 5 倍价买谷电？

用法：.venv/bin/python experiments/q3/diag_night.py
"""
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/q3-diag/slots.npz"
Q2X = ROOT / "results/q2_v6/L1-P1-R2-smpc/result2.xlsx"
WARMUP, T, ETA = 31, 144, 0.9
E_MAX_PER, S_MIN, S_MAX = 833.3333333333334, 1200.0, 10800.0


def q2_plan() -> np.ndarray:
    ws = openpyxl.load_workbook(Q2X, read_only=True)["计划购电量"]
    return np.array([[float(v or 0.0) for v in r] for r in
                     ws.iter_rows(min_row=2, min_col=2, max_col=145,
                                  values_only=True)])


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    p = z["p"]
    sl = slice(WARMUP, None)
    g, a, net = z["g"][sl], z["a"][sl], z["net"][sl]
    c, d, u, w, S = z["c"][sl], z["d"][sl], z["u"][sl], z["w"][sl], z["S"][sl]
    n_day = len(g)
    g2 = q2_plan()

    print("=" * 92)
    print("§1 逐 2 小时：计划、真值净负荷与电池净流（日均，kWh）")
    print("=" * 92)
    print(f"   {'时段':<12}{'均价':>7}{'净负荷⁺':>10}{'题3 g':>9}{'题3 a':>9}"
          f"{'题2 g':>9}{'Q3 电池净':>11}{'Q2 g−n⁺':>10}{'Q3 g−n⁺':>10}{'Σu':>8}")
    for k in range(12):
        b = slice(12 * k, 12 * (k + 1))
        npos = np.maximum(net[:, b], 0).sum() / n_day
        bat = (ETA * c[:, b].sum(1) - d[:, b].sum(1) / ETA).mean()
        print(f"   {k * 2:02d}:00-{k * 2 + 2:02d}:00 {p[b].mean():>7.3f}"
              f"{npos:>10,.0f}{g[:, b].sum() / n_day:>9,.0f}"
              f"{a[:, b].sum() / n_day:>9,.0f}{g2[:, b].sum() / n_day:>9,.0f}"
              f"{bat:>+11,.0f}"
              f"{g2[:, b].sum() / n_day - npos:>+10,.0f}"
              f"{g[:, b].sum() / n_day - npos:>+10,.0f}"
              f"{u[:, b].sum() / n_day:>8,.1f}")

    print("\n" + "=" * 92)
    print("§2 B_0 块内错配：0:00–2:00 欠、4:00–6:00 超")
    print("=" * 92)
    early, late = slice(0, 12), slice(24, 36)
    for lab, b in (("0:00-2:00", early), ("2:00-4:00", slice(12, 24)),
                   ("4:00-6:00", late)):
        npos = np.maximum(net[:, b], 0).sum()
        print(f"   {lab}: 题3 Σg/Σn⁺ = {g[:, b].sum() / npos:5.3f}   "
              f"题2 Σg/Σn⁺ = {g2[:, b].sum() / npos:5.3f}   "
              f"Σu = {u[:, b].sum():8,.0f} kWh   "
              f"充电顶格时段 {(c[:, b] > E_MAX_PER - 1e-6).sum():5,}")
    print(f"\n   4:00–6:00 充电顶格占该段全部时段的 "
          f"{(c[:, late] > E_MAX_PER - 1e-6).mean():.1%}"
          f"（题3）——夜里把补货全挤到最后两小时，功率成为瓶颈")
    print(f"   SOC：日初 {S[:, 0].mean():,.0f} → 4:00 {S[:, 24].mean():,.0f}"
          f" → 6:00 {S[:, 36].mean():,.0f} → 24:00 {S[:, 144].mean():,.0f} kWh")
    lowest = S[:, :37].min(axis=1)
    print(f"   夜间（0:00–6:00）最低 SOC：均值 {lowest.mean():,.0f}，"
          f"贴底(≤1210) 的天数 {(lowest <= S_MIN + 10).sum()}/{n_day}")

    print("\n" + "=" * 92)
    print("§3 紧急购电归因（按电量，题2 v6 §7.6 的五分类口径平移）")
    print("=" * 92)
    pos = u > 1e-6
    Sn = S[:, 1:]
    tot = u.sum()
    dmax = d > E_MAX_PER - 1e-6
    floor = Sn <= S_MIN + 10.0
    rise = np.diff(S, axis=1) > 1e-6
    cats = (("①功率强制（放电顶格）", pos & dmax & ~floor),
            ("②储能耗尽（SOC 贴底）", pos & floor),
            ("③a 囤电型（当步 SOC 上升）", pos & rise),
            ("④其余（低水位带 / 配给）",
             pos & ~dmax & ~floor & ~rise))
    for lab, m_ in cats:
        print(f"   {lab:<26}{u[m_].sum() / tot:>7.2%}"
              f"   {m_.sum():>6,} 个时段")
    b0 = slice(0, 36)
    print(f"\n   落在 B_0（不可修订）的紧急电量 {u[:, b0].sum() / tot:.1%}"
          f"，其中谷价段 "
          f"{u[:, b0][:, p[b0] < 0.5].sum() / max(u[:, b0].sum(), 1e-9):.1%}")
    print(f"   谷价段紧急电量的等效单价 5×p̄ = "
          f"{5 * float(p[p < 0.5].mean()):.3f} 元/kWh"
          f"（同一晚 4:00–6:00 的正常计划电价只要 {float(p[24:36].mean()):.3f}）")
    print(f"   全年紧急费 {5 * float(p @ u.sum(0)):,.0f} 元；其中 B_0 段 "
          f"{5 * float(p[b0] @ u[:, b0].sum(0)):,.0f} 元")

    print("\n" + "=" * 92)
    print("§4 弃电分类（按电量，不按时段数——时段数会被大量微量弃电淹没）")
    print("=" * 92)
    wp = w > 1e-6
    full = Sn > S_MAX - 1.0
    cmax = c > E_MAX_PER - 1e-6
    wtot = w.sum()
    # 当日之后 SOC 是否还会贴顶：判断"存不下"是否为跨时段约束
    later_full = np.zeros_like(full)
    for i in range(len(full)):
        later_full[i] = np.flip(np.maximum.accumulate(
            np.flip(full[i].astype(float)))) > 0.5
    for lab, m_ in (("SOC 当步贴顶", wp & full),
                    ("充电功率顶格（SOC 未满）", wp & cmax & ~full),
                    ("两者皆非但当日晚些 SOC 会贴顶", wp & ~full & ~cmax
                     & later_full),
                    ("两者皆非且当日 SOC 再未贴顶", wp & ~full & ~cmax
                     & ~later_full)):
        print(f"   {lab:<32}{w[m_].sum() / wtot:>7.2%}   "
              f"{w[m_].sum():>12,.0f} kWh   {m_.sum():>6,} 个时段")
    print(f"\n   题3 Σw {wtot:,.0f} kWh（弃电率 "
          f"{wtot / (z['r_hat0'][sl].sum() * 0 + (z['net'][sl] * 0).sum() + 1) if False else 0:.0f}"
          f"）——弃电率见 q3_summary.md 6.96%")

    print("\n" + "=" * 92)
    print("§5 δ− 的钱去哪了：按时段类型分摊 99 万偏差费")
    print("=" * 92)
    dm = np.maximum(g - a, 0.0)
    dp = np.maximum(a - g, 0.0)
    fee_dm = 0.5 * float(p @ dm.sum(0))
    fee_dp = 0.5 * float(p @ dp.sum(0))
    print(f"   下调违约金 ½Σp·δ− = {fee_dm:,.0f} 元"
          f"（{fee_dm / (fee_dm + fee_dp):.1%}）")
    print(f"   上调加价     ½Σp·δ+ = {fee_dp:,.0f} 元"
          f"（{fee_dp / (fee_dm + fee_dp):.1%}）")
    print(f"   合计偏差费 {fee_dm + fee_dp:,.0f} 元（q3_summary.md: 990,517）")
    hit = dm > 1e-6
    for lab, m_ in (("当步在弃电", hit & wp),
                    ("SOC 贴顶", hit & full),
                    ("充电功率顶格", hit & cmax),
                    ("以上皆非（纯粹计划过量）",
                     hit & ~wp & ~full & ~cmax)):
        share = 0.5 * float(p @ np.where(m_, dm, 0).sum(0)) / fee_dm
        print(f"     {lab:<24}{share:>7.1%} 的下调违约金"
              f"   δ− {np.where(m_, dm, 0).sum():>12,.0f} kWh")


if __name__ == "__main__":
    main()
