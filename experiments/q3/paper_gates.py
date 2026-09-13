"""题3 论文闸门 C3-5 / C3-6 / C3-8：从 v7 / v6 交付表定量，不重跑全年。

C3-5 台阶：B_0 vs B_1..3 的计划覆盖率 Σg/Σn⁺，以及 Σg^{Q3} vs Σg^{Q2}。
C3-6 分块紧急量：B_0 与题 2 同级；B_1–B_3 应明显下降。
C3-8 方向不对称：δ⁺/δ⁻ 与下调落在哪些修订块——不把 ≥2 写成已验证预言。

用法：.venv/bin/python experiments/q3/paper_gates.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from nmmc.data.year import load_year
from nmmc.model.lp import T
from nmmc.report.common import md_table
from nmmc.report.q2 import WARMUP_DAYS

ROOT = Path(__file__).resolve().parents[2]
Q3_XLSX = ROOT / "results/q3_v7/F1-L1-LU1-S-b/result3.xlsx"
Q2_XLSX = ROOT / "results/q2_v6/L1-P1-R2-smpc/result2.xlsx"
Q3_CSV = ROOT / "results/q3_v7/F1-L1-LU1-S-b/q3_details.csv"
Q2_CSV = ROOT / "results/q2_v6/L1-P1-R2-smpc/q2_details.csv"
OUT = ROOT / "results/gap-checks"
BLOCKS = ("B0 0–6h", "B1 6–12h", "B2 12–18h", "B3 18–24h")


def parse_segment(label: str) -> tuple[int, int]:
    """把 '08:50-09:10' 译回 0 基时段闭区间。"""
    a, b = str(label).strip().split("-")

    def minutes(text: str) -> int:
        h, m = text.split(":")
        return int(h) * 60 + int(m)

    i_a = minutes(a) // 10
    i_b = minutes(b) // 10 - 1
    return i_a, i_b


def load_slot_matrix(path: Path, sheet: str) -> np.ndarray:
    """官方购电表：行=报告期日，列=144 时段。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = []
    for row in ws.iter_rows(min_row=2, max_col=1 + T, values_only=True):
        if row[0] is None:
            continue
        rows.append([float(x or 0.0) for x in row[1:1 + T]])
    wb.close()
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] != T:
        raise ValueError(f"{path} {sheet} 列数 {arr.shape[1]} ≠ {T}")
    return arr


def block_sums(mat: np.ndarray) -> np.ndarray:
    """(n, 144) → (n, 4) 六个小时一块。"""
    return np.stack([mat[:, 36 * k:36 * (k + 1)].sum(axis=1) for k in range(4)],
                    axis=1)


def emergency_blocks_aligned(path: Path, dates) -> np.ndarray:
    """按报告期日期对齐的 4 块紧急电量。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["紧急购电量"]
    by_date: dict = {}
    cur = None
    for date, label, kwh, *_ in ws.iter_rows(min_row=2, max_col=3,
                                             values_only=True):
        if date is not None:
            cur = pd.Timestamp(date).normalize()
        if cur is None or label is None or kwh is None:
            continue
        i_a, i_b = parse_segment(str(label))
        vec = by_date.setdefault(cur, np.zeros(T))
        n_slot = max(i_b - i_a + 1, 1)
        vec[i_a:i_b + 1] += float(kwh) / n_slot
    wb.close()
    out = np.zeros((len(dates), 4))
    for i, d in enumerate(dates):
        key = pd.Timestamp(d).normalize()
        if key in by_date:
            out[i] = block_sums(by_date[key][None, :])[0]
    return out


def main() -> None:
    """计算 C3-5/6/8 并写 results/gap-checks/q3_paper_gates.md。"""
    year = load_year()
    net = year.ell[WARMUP_DAYS:] - year.r[WARMUP_DAYS:]
    n_plus = np.maximum(net, 0.0)
    dates = year.dates[WARMUP_DAYS:]
    g3 = load_slot_matrix(Q3_XLSX, "计划购电量")
    a3 = load_slot_matrix(Q3_XLSX, "调整购电量")
    g2 = load_slot_matrix(Q2_XLSX, "计划购电量")
    if g3.shape[0] != net.shape[0]:
        raise ValueError(f"Q3 行 {g3.shape[0]} vs 报告期 {net.shape[0]}")

    g3_b, n_b = block_sums(g3), block_sums(n_plus)
    g2_b = block_sums(g2)
    cover3 = g3_b.sum(0) / np.maximum(n_b.sum(0), 1e-9)
    cover2 = g2_b.sum(0) / np.maximum(n_b.sum(0), 1e-9)

    u3 = emergency_blocks_aligned(Q3_XLSX, dates)
    u2 = emergency_blocks_aligned(Q2_XLSX, dates)

    dp = np.maximum(a3 - g3, 0.0)
    dm = np.maximum(g3 - a3, 0.0)
    dp_b, dm_b = block_sums(dp), block_sums(dm)
    ratio = float(dp.sum() / max(float(dm.sum()), 1e-9))

    q3 = pd.read_csv(Q3_CSV)
    q2 = pd.read_csv(Q2_CSV)
    sum_g3, sum_g2 = float(q3.g_sum.sum()), float(q2.g_sum.sum())

    OUT.mkdir(parents=True, exist_ok=True)
    cover_tbl = md_table(
        ["块", "Q3 Σg/Σn⁺", "Q2 Σg/Σn⁺", "Q3 紧急 kWh", "Q2 紧急 kWh",
         "紧急比 Q3/Q2", "Q3 δ⁺", "Q3 δ⁻"],
        [(BLOCKS[k],
          f"{cover3[k]:.3f}", f"{cover2[k]:.3f}",
          f"{u3[:, k].sum():,.0f}", f"{u2[:, k].sum():,.0f}",
          f"{u3[:, k].sum() / max(u2[:, k].sum(), 1e-9):.3f}",
          f"{dp_b[:, k].sum():,.0f}", f"{dm_b[:, k].sum():,.0f}")
         for k in range(4)])
    step = cover3[0] - cover3[1:].mean()
    c36a = u3[:, 1:].sum() / max(u2[:, 1:].sum(), 1e-9)
    c36b = u3[:, 0].sum() / max(u2[:, 0].sum(), 1e-9)
    lines = [
        "# 题 3 论文闸门（v7 交付表，未重跑）",
        "",
        "## C3-5 台阶与 Σg",
        "",
        f"- Σg Q3 = {sum_g3:,.0f} kWh，Σg Q2 = {sum_g2:,.0f} kWh，"
        f"比 = {sum_g3 / sum_g2:.3f}"
        f"（判据 Σg^Q3 < Σg^Q2：{'过' if sum_g3 < sum_g2 else '不过'}）",
        f"- B0 覆盖率 {cover3[0]:.3f}，B1–B3 均值 {cover3[1:].mean():.3f}，"
        f"6:00 台阶（B0 − 其后均值）= {step:+.3f}"
        f"（预言 B0 更高：{'过' if step > 0 else '不过 / 方向相反'}）",
        "",
        cover_tbl,
        "",
        "## C3-6 分块紧急量",
        "",
        f"- C3-6a B1–B3 紧急电量比题 2 = {c36a:.3f}"
        f"（应明显 < 1：{'过' if c36a < 0.7 else '弱 / 不过'}）",
        f"- C3-6b B0 紧急电量比题 2 = {c36b:.3f}"
        f"（不超过题 2 的 1.5 倍即合格，允许更好："
        f"{'过' if c36b <= 1.5 else '变差超 50%'}）",
        "",
        "## C3-8 方向不对称",
        "",
        f"- δ⁺/δ⁻ = {ratio:.3f}（文档建议阈 ≥ 2："
        f"{'过' if ratio >= 2 else '不过，下调更多'}）",
        f"- δ⁺ = {dp.sum():,.0f} kWh，δ⁻ = {dm.sum():,.0f} kWh",
        "- 下调主要落在修订块："
        + "，".join(f"{BLOCKS[k]} {100 * dm_b[:, k].sum() / dm.sum():.1f}%"
                    for k in range(4)),
        "",
        "性质 9「宁存不退 ⇒ 上调远多于下调」在 v7 交付上不成立，"
        "不能把 ≥2 写成已验证结论。下调集中在午后/晚间块，与日内预报"
        "把光伏/负载往下修、计划偏多有关。",
        "",
    ]
    path = OUT / "q3_paper_gates.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame({
        "block": BLOCKS,
        "cover_q3": cover3, "cover_q2": cover2,
        "u_q3": u3.sum(0), "u_q2": u2.sum(0),
        "dp": dp_b.sum(0), "dm": dm_b.sum(0),
    }).to_csv(OUT / "q3_paper_gates.csv", index=False, float_format="%.6f")
    print(path.read_text(encoding="utf-8"))
    print(f"已写 {path}")


if __name__ == "__main__":
    main()
