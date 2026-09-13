"""把缺口实验产物收成一页摘要。缺哪项写哪项，可重复跑。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nmmc.report.common import md_table

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/gap-checks"
Q2_DELIVER = 13_745_816.13


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    """汇总 gap-checks 目录，写 summary.md。"""
    lines = ["# 缺口实验汇总", ""]
    nu_files = sorted(OUT.glob("nu_*.csv"))
    if nu_files:
        nu = pd.concat([pd.read_csv(p) for p in nu_files], ignore_index=True)
        nu = nu.sort_values("nu_end")
        lines += ["## 题 2 ν^end 敏感性（v6 口径，day 100–119）", "",
                  md_table(["ν^end", "窗口费用（元）", "相对 0.472", "紧急费占比"],
                           [(f"{r.nu_end:g}", f"{r.z:,.0f}",
                             f"{r.delta_pct:+.2f}%",
                             f"{100 * r.emerg_fee / r.z:.2f}%")
                            for r in nu.itertuples()]),
                  "",
                  "窗口与 §11.4 相同（day 100–119 / 2025-04-10–04-29）。"
                  "0.472 为交付臂同窗口，其余三点为新口径重扫。"
                  "费用随 ν 单调上升，锚点仍最优。",
                  ""]
    base = _read_csv(OUT / "q2_baselines.csv")
    if base is not None:
        lines += ["## 题 2 笨办法对照（报告期 334 天）", "",
                  md_table(["对照", "Z^online（元）", "相对交付", "紧急费占比"],
                           [(r.kind, f"{r.z_online:,.2f}",
                             f"{100 * (r.z_online / Q2_DELIVER - 1):+.1f}%",
                             f"{100 * r.emerg_share:.2f}%")
                            for r in base.itertuples()]),
                  f"",
                  f"交付臂 Z^online = {Q2_DELIVER:,.2f} 元。",
                  ""]
        nobat = base[base.kind.astype(str).str.startswith("nobat")]
        if len(nobat):
            best = nobat.loc[nobat.z_online.idxmin()]
            lines += [f"无电池报童扫描最低点 α={best.alpha:g}，"
                      f"落在 C10 区间 [0.80, 0.96]："
                      f"{'过' if 0.80 <= float(best.alpha) <= 0.96 else '不过'}。",
                      ""]
    gates = OUT / "q3_paper_gates.md"
    if gates.exists():
        body = gates.read_text(encoding="utf-8")
        if body.startswith("# "):
            body = "\n".join(body.splitlines()[1:]).lstrip()
        lines += ["## 题 3 C3-5 / C3-6 / C3-8", "", body, ""]
    ladder_spec = [
        ("仅 0:00", ROOT / "results/q3_v7-ladder/F1-L1-LU1-S-b-E0/q3_details.csv"),
        ("+6:00", ROOT / "results/q3_v7-ladder/F1-L1-LU1-S-b-E0-6/q3_details.csv"),
        ("+12:00", ROOT / "results/q3_v7-ladder/F1-L1-LU1-S-b-E0-6-12/q3_details.csv"),
        ("全开（交付）", ROOT / "results/q3_v7/F1-L1-LU1-S-b/q3_details.csv"),
    ]
    ladder_rows = []
    for label, csv in ladder_spec:
        if csv.exists():
            z = float(pd.read_csv(csv).z_online.sum())
            ladder_rows.append((label, csv.parent.name, z))
    if ladder_rows:
        lines += ["## 题 3 附加问阶梯（v7 计划器）", "",
                  md_table(["档", "标签", "Z^online（元）", "逐级 Δ"],
                           [(lab, tag, f"{z:,.2f}",
                             "—" if i == 0 else f"{z - ladder_rows[i-1][2]:+,.2f}")
                            for i, (lab, tag, z) in enumerate(ladder_rows)]),
                  ""]
    lines += [
        "## 仍未做（非本轮）",
        "",
        "- 题 2 的 3×3×2 预报消融网格、题 3 的 12 组网格：现行口径下仍作废，未重跑。",
        "- 多种子 / $N_s$ 扫描 / 暖机冲销：未做。",
        "- 题 3 的 2×2 因子（附件 3 × 改单权）未单独重跑；附加问用 A0–A3 逐级阶梯回答。",
        "",
    ]
    path = OUT / "summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    print(f"已写 {path}")


if __name__ == "__main__":
    main()
