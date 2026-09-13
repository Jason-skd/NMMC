"""题3 诊断：逐时段离线分析（读 dump_slots.py 落盘的 npz + 题2 v6 result2）。

回答的问题（对应诊断档条目）：
  §A 计划的时段构成：题3 的 g 与题2 v6 的 g 在谷/平/峰的分布差；单价差。
  §B 下调 δ− 落在哪：按修订块 B_m、按价段、按当步 SOC/弃电状态分层。
  §C 阶段0 计划的隐含分位：g 相对情景净负荷 n^(s) 与情景试探注入 a^(s)
      的经验分位——检验"报童分位"预言（性质7）与"g = a^(s) 中位数"的
      LP 一阶条件。
  §D 夜间（B_0）行为：充放电、g 覆盖率、紧急购电。
  §E 紧急购电上界 u ≤ (n−a)⁺ 在实测路径上是否已自动满足（口径同步的代价）。
  §F 弃电与 SOC 贴边的共存诊断。

用法：.venv/bin/python experiments/q3/diag_plan.py
"""
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/q3-diag/slots.npz"
Q2X = ROOT / "results/q2_v6/L1-P1-R2-smpc/result2.xlsx"
WARMUP = 31
T = 144
ETA = 0.9
E_MAX_PER = 833.3333333333334      # 5000 kW × (1/6) h
S_MIN, S_MAX = 1200.0, 10800.0
BLOCKS = ("B0 0:00-6:00", "B1 6:00-12:00", "B2 12:00-18:00", "B3 18:00-24:00")


def price_seg(p: np.ndarray) -> dict[str, np.ndarray]:
    """按 report.q2.PRICE_BANDS 的谷/平/峰阈值给时段掩码（电价是连续曲线，
    不是三档，必须按 0.5 / 1.0 元的带界分段，与题2 报表同口径）。"""
    return {"谷": p < 0.5, "平": (p >= 0.5) & (p < 1.0), "峰": p >= 1.0}


def q2_plan() -> np.ndarray:
    """题2 v6 交付计划 g，shape (334, 144)。"""
    ws = openpyxl.load_workbook(Q2X, read_only=True)["计划购电量"]
    rows = []
    for r in ws.iter_rows(min_row=2, min_col=2, max_col=145,
                          values_only=True):
        rows.append([float(v or 0.0) for v in r])
    return np.array(rows)


def q2_night() -> tuple[np.ndarray, np.ndarray]:
    """题2 v6 的 0:00–4:00 块充/放电量（电池侧），各 shape (334,)。"""
    ws = openpyxl.load_workbook(Q2X, read_only=True)["充放电量"]
    ch, di = [], []
    for i, r in enumerate(ws.iter_rows(min_row=2, min_col=2, max_col=4,
                                       values_only=True)):
        if i % 6 == 0:                      # 每日第一块 = 0:00–4:00
            ch.append(float(r[1] or 0.0))
            di.append(float(r[2] or 0.0))
    return np.array(ch), np.array(di)


def bar(x: np.ndarray, width: int = 46) -> str:
    """把非负序列画成粗糙的 ASCII 竖条（诊断可读性用）。"""
    m = max(x.max(), 1e-9)
    lv = " ▁▂▃▄▅▆▇█"
    idx = np.clip((x / m * 8).round().astype(int), 0, 8)
    step = max(1, len(x) // width)
    return "".join(lv[idx[i]] for i in range(0, len(x), step))


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    p = z["p"]
    sl = slice(WARMUP, None)
    g, a, net = z["g"][sl], z["a"][sl], z["net"][sl]
    c, d, u, w, S = z["c"][sl], z["d"][sl], z["u"][sl], z["w"][sl], z["S"][sl]
    commit = z["commit"][sl]
    ascn = {k: z[f"ascn_{k}"][sl] for k in ("lo", "med", "hi")}
    scn0 = {k: z[f"scn0_{k}"][sl] for k in ("lo", "med", "hi")}
    dates = z["dates"][sl]
    n_day = len(g)
    seg = price_seg(p)
    g2 = q2_plan()
    assert g2.shape == g.shape, f"题2 计划 {g2.shape} 与题3 {g.shape} 不同形"
    dp = np.maximum(a - g, 0.0)
    dm = np.maximum(g - a, 0.0)

    print("=" * 78)
    print(f"§A 计划的时段构成（报告期 {n_day} 天；题2 = v6 交付）")
    print("=" * 78)
    hdr = f"   {'价段':<5}{'题3 Σg':>13}{'题3 Σa':>13}{'题2 Σg':>13}"
    print(hdr + f"{'Σa−Σg(Q2)':>13}{'均价Q3(a)':>11}{'均价Q2':>9}")
    for name, mask in (("谷", seg["谷"]), ("平", seg["平"]), ("峰", seg["峰"])):
        q3g, q3a, q2g = g[:, mask].sum(), a[:, mask].sum(), g2[:, mask].sum()
        print(f"   {name:<5}{q3g:>13,.0f}{q3a:>13,.0f}{q2g:>13,.0f}"
              f"{q3a - q2g:>+13,.0f}{p[mask][0]:>11.4f}{p[mask][0]:>9.4f}")
    print(f"   {'合计':<5}{g.sum():>13,.0f}{a.sum():>13,.0f}{g2.sum():>13,.0f}"
          f"{a.sum() - g2.sum():>+13,.0f}"
          f"{float(p @ a.sum(0)) / a.sum():>11.4f}"
          f"{float(p @ g2.sum(0)) / g2.sum():>9.4f}")
    print(f"\n   题3 实际购电 Σa 比题2 少 {g2.sum() - a.sum():,.0f} kWh，"
          f"基础购电费却多 {float(p @ a.sum(0)) - float(p @ g2.sum(0)):,.0f} 元")
    print(f"   → 单价差 {float(p @ a.sum(0)) / a.sum() - float(p @ g2.sum(0)) / g2.sum():+.4f} "
          f"元/kWh（{(float(p @ a.sum(0)) / a.sum()) / (float(p @ g2.sum(0)) / g2.sum()) - 1:+.2%}）")
    print(f"\n   逐时段日均（题3 a / 题2 g / 题3 g），0:00→24:00：")
    for nm, arr in (("Q3 a", a.mean(0)), ("Q3 g", g.mean(0)),
                    ("Q2 g", g2.mean(0)), ("净负荷", net.mean(0))):
        print(f"     {nm:<7}|{bar(np.maximum(arr, 0))}| max {arr.max():6.0f}")

    print("\n" + "=" * 78)
    print("§B 下调 δ− 落在哪")
    print("=" * 78)
    print(f"   Σδ+ {dp.sum():,.0f} / Σδ− {dm.sum():,.0f} kWh   "
          f"δ+/δ− = {dp.sum() / dm.sum():.4f}")
    print(f"\n   按修订块：")
    print(f"     {'块':<15}{'Σδ+':>12}{'Σδ−':>13}{'δ+/δ−':>9}"
          f"{'Σg':>13}{'δ−/Σg':>9}")
    for m in range(4):
        b = slice(36 * m, 36 * (m + 1))
        print(f"     {BLOCKS[m]:<15}{dp[:, b].sum():>12,.0f}"
              f"{dm[:, b].sum():>13,.0f}"
              f"{dp[:, b].sum() / max(dm[:, b].sum(), 1e-9):>9.3f}"
              f"{g[:, b].sum():>13,.0f}"
              f"{dm[:, b].sum() / max(g[:, b].sum(), 1e-9) * 100:>8.1f}%")
    print(f"\n   按价段：")
    print(f"     {'价段':<5}{'Σδ+':>12}{'Σδ−':>13}{'δ− 占计划':>11}"
          f"{'δ− 的钱':>14}")
    for name, mask in (("谷", seg["谷"]), ("平", seg["平"]), ("峰", seg["峰"])):
        print(f"     {name:<5}{dp[:, mask].sum():>12,.0f}"
              f"{dm[:, mask].sum():>13,.0f}"
              f"{dm[:, mask].sum() / max(g[:, mask].sum(), 1e-9) * 100:>10.1f}%"
              f"{0.5 * float(p[mask] @ dm[:, mask].sum(0)):>14,.0f}")
    # 砍单当步的物理状态：SOC 是否贴顶 / 是否在弃电 / 是否功率顶格
    hit = dm > 1e-6
    full = S[:, 1:] > S_MAX - 1.0
    curt = w > 1e-6
    cmax = c > E_MAX_PER - 1e-6
    print(f"\n   有下调的 {hit.sum():,} 个时段里：SOC 贴顶 "
          f"{(hit & full).sum() / hit.sum():.1%}，当步在弃电 "
          f"{(hit & curt).sum() / hit.sum():.1%}，充电功率顶格 "
          f"{(hit & cmax).sum() / hit.sum():.1%}，"
          f"三者皆无 {(hit & ~full & ~curt & ~cmax).sum() / hit.sum():.1%}")
    print(f"   下调时段的 SOC 分布：p10 {np.percentile(S[:, 1:][hit], 10):,.0f}  "
          f"中位 {np.median(S[:, 1:][hit]):,.0f}  p90 "
          f"{np.percentile(S[:, 1:][hit], 90):,.0f} kWh（上限 {S_MAX:,.0f}）")

    print("\n" + "=" * 78)
    print("§C 阶段0 计划的隐含分位（性质7 / LP 一阶条件）")
    print("=" * 78)
    # (1) g 相对阶段0 情景试探注入 a^(s) 的位置：LP 一阶条件应给中位数
    print("   (1) g 与阶段0 逐情景试探注入 a^(s) 的关系（理论：g = 中位数）")
    print(f"       {'块':<15}{'g>med(a^s) 占比':>16}{'g−med(a^s) 日均':>17}"
          f"{'g−p90(a^s) 日均':>17}")
    for m in range(4):
        b = slice(36 * m, 36 * (m + 1))
        if m == 0:
            print(f"       {BLOCKS[m]:<15}{'—（a≡g，无 δ）':>16}")
            continue
        gt = (g[:, b] > ascn["med"][:, b] + 1e-6).mean()
        print(f"       {BLOCKS[m]:<15}{gt:>15.1%}"
              f"{(g[:, b] - ascn['med'][:, b]).sum(1).mean():>17,.0f}"
              f"{(g[:, b] - ascn['hi'][:, b]).sum(1).mean():>17,.0f}")
    # (2) g 覆盖情景净负荷的经验分位（报童口径）
    print("\n   (2) g 对阶段0 情景净负荷 n^(s) 的覆盖（g ≥ n^(s) 的情景占比）")
    print(f"       {'块':<15}{'g≥med(n^s)':>13}{'g≥p90(n^s)':>13}"
          f"{'Σg/Σmed(n^s)':>15}")
    for m in range(4):
        b = slice(36 * m, 36 * (m + 1))
        pos = net[:, b].sum(1) > 0            # 只看净负荷为正的块，避免除零
        print(f"       {BLOCKS[m]:<15}"
              f"{(g[:, b] >= scn0['med'][:, b]).mean():>12.1%}"
              f"{(g[:, b] >= scn0['hi'][:, b]).mean():>12.1%}"
              f"{g[:, b].sum() / max(np.maximum(scn0['med'][:, b], 0).sum(), 1e-9):>15.3f}")
    # (3) 最终 a 相对真实净负荷；与题2 的 g 对比
    print("\n   (3) 计划 / 最终注入相对真值净负荷（Σ 比值，报告期）")
    npos = np.maximum(net, 0.0)
    for m in range(4):
        b = slice(36 * m, 36 * (m + 1))
        den = max(npos[:, b].sum(), 1e-9)
        print(f"       {BLOCKS[m]:<15} Σg/Σn⁺ {g[:, b].sum() / den:6.3f}"
              f"   Σa/Σn⁺ {a[:, b].sum() / den:6.3f}"
              f"   题2 Σg/Σn⁺ {g2[:, b].sum() / den:6.3f}")

    print("\n" + "=" * 78)
    print("§D 夜间（B_0，0:00–6:00 不可修订）")
    print("=" * 78)
    q2ch, q2di = q2_night()
    n24 = slice(0, 24)
    print(f"   0:00–4:00 日均（电池侧）：题3 充 {(ETA * c[:, n24].sum(1)).mean():,.0f}"
          f" / 放 {(d[:, n24].sum(1) / ETA).mean():,.0f}"
          f" → 净 {(ETA * c[:, n24].sum(1) - d[:, n24].sum(1) / ETA).mean():+,.0f} kWh")
    print(f"   {'':<21}题2v6 充 {q2ch.mean():,.0f} / 放 {q2di.mean():,.0f}"
          f" → 净 {(q2ch - q2di).mean():+,.0f} kWh")
    nd = (d[:, n24].sum(1) / ETA) > (ETA * c[:, n24].sum(1))
    print(f"   0:00–4:00 净放电的天数：题3 {nd.sum()}/{n_day}，"
          f"题2v6 {((q2di - q2ch) > 0).sum()}/{n_day}")
    b0 = slice(0, 36)
    print(f"   B_0 计划覆盖：题3 Σg {g[:, b0].sum():,.0f} kWh，"
          f"题2v6 Σg {g2[:, b0].sum():,.0f} kWh"
          f"（差 {g[:, b0].sum() - g2[:, b0].sum():+,.0f}，"
          f"{g[:, b0].sum() / g2[:, b0].sum() - 1:+.1%}）")
    print(f"   B_0 紧急购电：Σu {u[:, b0].sum():,.0f} kWh / 全天 {u.sum():,.0f}"
          f"（占 {u[:, b0].sum() / max(u.sum(), 1e-9):.1%}）")
    print(f"   B_0 末（6:00）SOC：题3 均值 {S[:, 36].mean():,.0f}  "
          f"日初均值 {S[:, 0].mean():,.0f} → 夜间净变化 "
          f"{(S[:, 36] - S[:, 0]).mean():+,.0f} kWh")

    print("\n" + "=" * 78)
    print("§E 紧急购电上界 u ≤ (n−a)⁺ 在实测路径上的状态（口径同步代价）")
    print("=" * 78)
    cap = np.maximum(net - a, 0.0)
    viol = u - cap
    pos = u > 1e-6
    print(f"   有 u 的时段 {pos.sum():,} 个；max(u − (n−a)⁺) = {viol.max():.3e} kWh")
    print(f"   u∧c 时段 {((u > 1e-6) & (c > 1e-6)).sum()}；"
          f"u∧w 时段 {((u > 1e-6) & (w > 1e-6)).sum()}")
    print("   → 实测轨迹已满足新口径上界（由 c·d=0 与 u·w=0 可证），"
          "同步 v6 口径不改交付轨迹；改的是情景块与规划层的影子价格。")
    print(f"   紧急电量按价段：谷 {u[:, seg['谷']].sum() / u.sum():.1%} / "
          f"平 {u[:, seg['平']].sum() / u.sum():.1%} / "
          f"峰 {u[:, seg['峰']].sum() / u.sum():.1%}")
    print(f"   紧急发生时的 SOC：p10 {np.percentile(S[:, 1:][pos], 10):,.0f}  "
          f"中位 {np.median(S[:, 1:][pos]):,.0f}  p90 "
          f"{np.percentile(S[:, 1:][pos], 90):,.0f}（下限 {S_MIN:,.0f}）")
    print(f"   紧急时段中 SOC 贴底(≤1210) 占比 "
          f"{(S[:, 1:][pos] <= S_MIN + 10).mean():.1%}；"
          f"放电功率顶格占比 {(d[pos] > E_MAX_PER - 1e-6).mean():.1%}")

    print("\n" + "=" * 78)
    print("§F 弃电 / SOC 贴边共存")
    print("=" * 78)
    print(f"   Σw {w.sum():,.0f} kWh；弃电时段 {(w > 1e-6).sum():,}"
          f"（占 {(w > 1e-6).mean():.1%}）")
    for lab, msk in (("SOC 贴顶", S[:, 1:] > S_MAX - 1.0),
                     ("充电功率顶格", c > E_MAX_PER - 1e-6),
                     ("两者皆非", (S[:, 1:] <= S_MAX - 1.0)
                      & (c <= E_MAX_PER - 1e-6))):
        m_ = (w > 1e-6) & msk
        print(f"     弃电时段中 {lab:<8} {m_.sum() / max((w > 1e-6).sum(), 1):>6.1%}")
    print(f"   SOC 贴底(≤1210) 时段 {(S[:, 1:] <= S_MIN + 10).sum():,}"
          f"（占 {(S[:, 1:] <= S_MIN + 10).mean():.1%}）；"
          f"贴顶 {(S[:, 1:] > S_MAX - 1).sum():,}"
          f"（{(S[:, 1:] > S_MAX - 1).mean():.1%}）")

    print("\n" + "=" * 78)
    print("§G 逐阶段承诺的漂移（0:00 计划 → 6/12/18 三次改单）")
    print("=" * 78)
    print(f"   {'块':<15}{'g(0:00)':>12}{'a(6:00后)':>12}{'a(12:00后)':>12}"
          f"{'a(18:00后)':>12}{'最终 a':>12}")
    for m in range(4):
        b = slice(36 * m, 36 * (m + 1))
        row = [commit[:, k, b].sum() / n_day for k in range(4)]
        print(f"   {BLOCKS[m]:<15}{g[:, b].sum() / n_day:>12,.0f}"
              + "".join(f"{v:>12,.0f}" for v in row[1:])
              + f"{a[:, b].sum() / n_day:>12,.0f}")
    print("   （每列为该阶段结束后对该块的承诺日均值；B_m 只在 m 阶段定稿）")

    print("\n" + "=" * 78)
    print("§H 四个指定日的计划形状（Σ 按 6 小时块，kWh）")
    print("=" * 78)
    print(f"   {'日期':<12}{'臂':<8}" + "".join(f"{b[:2]:>10}" for b in BLOCKS))
    for date in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"):
        i = int(np.flatnonzero(dates == date)[0])
        for nm, arr in (("Q3 g", g[i]), ("Q3 a", a[i]), ("Q2 g", g2[i]),
                        ("净负荷⁺", np.maximum(net[i], 0))):
            print(f"   {date if nm == 'Q3 g' else '':<12}{nm:<8}"
                  + "".join(f"{arr[36 * m:36 * (m + 1)].sum():>10,.0f}"
                            for m in range(4)))


if __name__ == "__main__":
    main()
