"""题 2/3 交付轨迹按附件 4 实现价重开账单（Z^mis，不重新优化）。

三格分解（题4 档 §12.1）：
    Z^Q4 − Z^Q = (Z^mis − Z^Q) + (Z^Q4 − Z^mis)
                 价格效应（同一策略换账单）   策略适应价值

g / a 从 result*.xlsx 逐时段读取，计划项（含题 3 偏差费）精确重开。
紧急购电表只有段合计：用交付 csv 的紧急费 × (实现价·û)/(原价·û)
换价，û 为段内均匀摊回；原价下闭合到交付 Z。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data import ROOT
from nmmc.data.day import load_day1
from nmmc.data.price import load_price
from nmmc.data.year import load_year
from nmmc.model.lp import T
from nmmc.report.common import md_table
from nmmc.report.q2 import WARMUP_DAYS

OUT = ROOT / "results/gap-checks"
Q2_XLSX = ROOT / "results/q2_v6/L1-P1-R2-smpc/result2.xlsx"
Q2_CSV = ROOT / "results/q2_v6/L1-P1-R2-smpc/q2_details.csv"
Q3_XLSX = ROOT / "results/q3_v7/F1-L1-LU1-S-b/result3.xlsx"
Q3_CSV = ROOT / "results/q3_v7/F1-L1-LU1-S-b/q3_details.csv"
Q42_CSV = ROOT / "results/q4/4-2/PR1-V2-r/q4-2_details.csv"
Q43_CSV = ROOT / "results/q4/4-3/PR1-V2-r/q4-3_details.csv"


def _slots(xlsx: Path, sheet: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """购电表：日期列 + 144 个时段。"""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    dates = pd.to_datetime(df.iloc[:, 0])
    arr = df.iloc[:, 1:1 + T].to_numpy(dtype=float)
    if arr.shape[1] != T:
        raise ValueError(f"{xlsx} {sheet}: 期望 {T} 列，得到 {arr.shape[1]}")
    return dates, arr


def _parse_seg(label: str) -> tuple[int, int]:
    """segment_label 的逆：'13:00-13:30' → (i_a, i_b) 0 基含端点。"""
    left, right = str(label).split("-")

    def minutes(token: str) -> int:
        hour, minute = (int(x) for x in token.split(":"))
        return hour * 60 + minute

    i_a = minutes(left) // 10
    i_b = minutes(right) // 10 - 1
    if not (0 <= i_a <= i_b < T):
        raise ValueError(f"无法解析紧急段标签: {label!r}")
    return i_a, i_b


def _emergency_u(xlsx: Path, dates: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
    """紧急购电表摊回 (n_days, 144)。段内均匀；返回诊断。"""
    raw = pd.read_excel(xlsx, sheet_name="紧急购电量")
    raw.iloc[:, 0] = raw.iloc[:, 0].ffill()
    u = np.zeros((len(dates), T))
    index = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    n_seg = n_single = n_skip = 0
    for row in raw.itertuples(index=False):
        date, label, energy = row[0], row[1], row[2]
        if date is None or label is None or energy is None:
            continue
        key = pd.Timestamp(date).normalize()
        if key not in index:
            n_skip += 1
            continue
        i_a, i_b = _parse_seg(str(label))
        width = i_b - i_a + 1
        u[index[key], i_a:i_b + 1] += float(energy) / width
        n_seg += 1
        n_single += int(width == 1)
    stats = {
        "n_seg": n_seg, "n_single": n_single, "n_skip": n_skip,
        "single_share": n_single / max(n_seg, 1),
    }
    return u, stats


def _js(dates: pd.DatetimeIndex, year) -> np.ndarray:
    """xlsx 日期 → 附件 2 的 0 基日索引。"""
    lookup = {pd.Timestamp(d).normalize(): int(j)
              for j, d in enumerate(year.dates)}
    out = np.array([lookup[pd.Timestamp(d).normalize()] for d in dates])
    if out[0] != WARMUP_DAYS:
        raise ValueError(f"报告期起始日索引 {out[0]}，期望 {WARMUP_DAYS}")
    return out


def _line(name: str, xlsx: Path, csv: Path, q4_csv: Path, adjusted: bool,
          year, price, typ_p: np.ndarray) -> dict:
    """一条交付线：原账单 / 重开账单 / 题 4 对照。"""
    dates, g = _slots(xlsx, "计划购电量")
    a = _slots(xlsx, "调整购电量")[1] if adjusted else g
    u, estats = _emergency_u(xlsx, dates)
    js = _js(dates, year)
    daily = pd.read_csv(csv)
    q4 = pd.read_csv(q4_csv)
    if len(daily) != len(dates) or len(q4) != len(dates):
        raise ValueError(f"{name}: 行数 csv={len(daily)} xlsx={len(dates)} "
                         f"q4={len(q4)}")

    z_old = z_mis = z_plan_old = z_plan_mis = z_emerg_old = z_emerg_mis = 0.0
    z_ratio_guard = 0.0
    rows = []
    for i, j in enumerate(js):
        p_old, p_new = typ_p, price.p[j]
        emerg_csv = float(daily.emerg_fee.iloc[i])
        if adjusted:
            plan_old = float(p_old @ a[i] + 0.5 * (p_old @ np.abs(a[i] - g[i])))
            plan_new = float(p_new @ a[i] + 0.5 * (p_new @ np.abs(a[i] - g[i])))
        else:
            plan_old = float(p_old @ g[i])
            plan_new = float(p_new @ g[i])
        # 紧急费：csv 是真值；段内均匀 u 只用来估实现价/原价比
        den = float(p_old @ u[i])
        if emerg_csv > 0 and den > 1e-12:
            ratio = float(p_new @ u[i]) / den
            emerg_new = emerg_csv * ratio
        else:
            ratio, emerg_new = 1.0, emerg_csv
        old = plan_old + emerg_csv
        new = plan_new + emerg_new
        z_old += old
        z_mis += new
        z_plan_old += plan_old
        z_plan_mis += plan_new
        z_emerg_old += emerg_csv
        z_emerg_mis += emerg_new
        z_ratio_guard += abs(old - float(daily.z_online.iloc[i]))
        rows.append({
            "date": str(pd.Timestamp(dates.iloc[i]).date()),
            "j": int(j),
            "z_old": old, "z_mis": new,
            "z_q4": float(q4.z_online.iloc[i]),
            "plan_old": plan_old, "plan_mis": plan_new,
            "emerg_old": emerg_csv, "emerg_mis": emerg_new,
            "emerg_px_ratio": ratio,
        })

    z_q = float(daily.z_online.sum())
    z_q4 = float(q4.z_online.sum())
    return {
        "line": name,
        "Z_Q": z_q,
        "Z_mis": z_mis,
        "Z_Q4": z_q4,
        "价格效应": z_mis - z_q,
        "策略适应": z_q4 - z_mis,
        "总变化": z_q4 - z_q,
        "计划_旧": z_plan_old,
        "计划_mis": z_plan_mis,
        "紧急_旧": z_emerg_old,
        "紧急_mis": z_emerg_mis,
        "原账单闭合残差": z_ratio_guard,
        **estats,
        "daily": rows,
    }


def main() -> None:
    """写出 q4_mis.csv / q4_mis_daily.csv / 一段 markdown。"""
    day1 = load_day1()
    year = load_year()
    price = load_price(day1.p)
    specs = [
        ("Q4-2", Q2_XLSX, Q2_CSV, Q42_CSV, False),
        ("Q4-3", Q3_XLSX, Q3_CSV, Q43_CSV, True),
    ]
    summaries = [_line(name, xlsx, csv, q4, adj, year, price, day1.p)
                 for name, xlsx, csv, q4, adj in specs]
    OUT.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in row.items() if k != "daily"} for row in summaries]
    pd.DataFrame(slim).to_csv(OUT / "q4_mis.csv", index=False, float_format="%.6f")
    pd.DataFrame([r for row in summaries for r in row["daily"]]).assign(
        line=np.repeat([s["line"] for s in summaries],
                       [len(s["daily"]) for s in summaries])
    ).to_csv(OUT / "q4_mis_daily.csv", index=False, float_format="%.6f")

    def money(x: float) -> str:
        return f"{x:,.2f} 元"

    lines = [
        "# 题 4 三格分解（Z^mis = 题 2/3 轨迹 × 附件 4 实现价）",
        "",
        "同一 (g, a) 逐时段按实现价重开；紧急费 = 交付紧急费 × "
        "段内摊回电量的实现价/原价比（原价下闭合到交付 Z）。",
        "",
        md_table(
            ["线", "Z^Q（原账单）", "Z^mis（换价）", "Z^Q4（重优化）",
             "价格效应", "策略适应", "总变化", "原账单闭合残差"],
            [(s["line"], money(s["Z_Q"]), money(s["Z_mis"]), money(s["Z_Q4"]),
              money(s["价格效应"]), money(s["策略适应"]), money(s["总变化"]),
              f"{s['原账单闭合残差']:.4f}")
             for s in summaries]),
        "",
        md_table(
            ["线", "计划旧 / mis", "紧急旧 / mis", "单槽段占比"],
            [(s["line"],
              f"{s['计划_旧']:,.2f} / {s['计划_mis']:,.2f}",
              f"{s['紧急_旧']:,.2f} / {s['紧急_mis']:,.2f}",
              f"{100 * s['single_share']:.1f}%（{s['n_single']}/{s['n_seg']}）")
             for s in summaries]),
        "",
        "价格效应 = Z^mis − Z^Q；策略适应 = Z^Q4 − Z^mis。"
        "总变化 = Z^Q4 − Z^Q。序关系倒序如实保留。",
        "",
    ]
    (OUT / "q4_mis.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "q4_mis.md").read_text(encoding="utf-8"))
    for s in summaries:
        print(f"{s['line']}: Z_Q={s['Z_Q']:,.2f}  Z_mis={s['Z_mis']:,.2f}  "
              f"Z_Q4={s['Z_Q4']:,.2f}  效应={s['价格效应']:,.2f}  "
              f"适应={s['策略适应']:,.2f}  闭合残差={s['原账单闭合残差']:.4f}")


if __name__ == "__main__":
    main()
