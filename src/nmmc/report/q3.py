"""题3 结算与交付：自检量、result3 四表、三块费用分解、调整统计、摘要。

交付口径（文档 §9）：计划购电量表填 g（全天购电量 Σg、全天购电费为名义
计划费 Σp·g）；调整购电量表填全天最终注入 a（B_0 段 = g），全天购电费为
实际结算购电费 Σ[p·a + ½p|a−g|]（含偏差费、不含紧急费）；充放电量与
紧急购电量同题2（真实路径、电池侧、区间合并）。诊断量对应文档 §12。
"""
import datetime as dt
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from nmmc.data import ATTACH
from nmmc.data.year import YearData
from nmmc.model.lp import ETA, T
from nmmc.model.q2 import KAPPA
from nmmc.model.q3 import beta_check
from nmmc.report.common import PAPER_SLOTS, md_table, segment_label
from nmmc.report.q2 import (BLOCK_LABELS, EPS_EMERG, PAPER_DATES,
                            WARMUP_DAYS, emergency_segments, report_days)
from nmmc.run.q3 import DayResult

TEMPLATE = ATTACH / "附件5" / "result3.xlsx"


def _py_date(day: DayResult, year: YearData) -> dt.date:
    """DayResult 的 0 基日索引转 Python 日期。"""
    return year.dates[day.j].astype(object)


def day_checks(day: DayResult, year: YearData) -> dict[str, float | int]:
    """单日自检量（文档 §12.4 的 C3-7/C6/C7/C8/C9）。

    Returns:
        dict：前缀偏差（a 与 g 在 B_0 段应逐格相等）、对账与能量守恒残差、
        互斥乘积、u∧c 计数、序关系违例标志。
    """
    net = year.ell[day.j] - year.r[day.j]
    d_soc = day.S[T] - day.S[0]
    bat = ETA * day.c.sum() - day.d.sum() / ETA
    losses = (1 - ETA) * day.c.sum() + (1 / ETA - 1) * day.d.sum()
    c7 = abs((day.a + day.u - net - day.w).sum() - (d_soc + losses))
    tol = 1e-4
    # C9 序关系（同题2 档 §7.1 推广）：Z^PI ≤ Z^ex-post ≤ min(Z^online,
    # Z^CE)——事后最优对两臂内环策略都是可行下界
    bad_order = int(day.z_pi > day.z_expost + tol
                    or day.z_expost > day.z + tol
                    or day.z_expost > day.z_ce + tol)
    return {
        "c3_7": float(np.abs(day.a[:36] - day.g[:36]).max()),
        "c6": float(abs(bat - d_soc)),
        "c7": float(c7),
        "cd_max": float((day.c * day.d).max()),
        "uw_max": float((day.u * day.w).max()),
        "u_and_c": int(((day.u > EPS_EMERG) & (day.c > EPS_EMERG)).sum()),
        "c9_bad": bad_order,
    }


def summarize(days: list[DayResult], year: YearData, p: np.ndarray
              ) -> dict[str, float | int]:
    """汇总报告期诊断量（文档 §12.3）：三块费用分解、EVPI、调整统计
    （方向比、行使率）、紧急购电、弃电、SOC 水平与预报技能。

    Args:
        days: 全部回测日（本函数只取报告期）。
        year: 全年数据（真值与 SOC 链自检用）。
        p: 逐时段电价。
    """
    rep = report_days(days)
    if not rep:
        raise ValueError("报告期为空（days 须超过暖机期 31 天）")
    checks = [day_checks(d, year) for d in rep]
    u_all = np.array([d.u for d in rep])
    base_fee = [float(p @ d.a) for d in rep]
    dev_fee = [0.5 * float(p @ np.abs(d.a - d.g)) for d in rep]
    emerg_fee = [KAPPA * float(p @ d.u) for d in rep]
    dp_all = np.concatenate([np.maximum(d.a - d.g, 0)[36:] for d in rep])
    dm_all = np.concatenate([np.maximum(d.g - d.a, 0)[36:] for d in rep])
    adj_all = dp_all + dm_all
    err_l = np.concatenate([np.abs(d.ell_hat - year.ell[d.j]) for d in rep])
    err_r = np.concatenate([np.abs(d.r_hat - year.r[d.j]) for d in rep])
    z_online = float(sum(d.z for d in rep))
    z_ce = float(sum(d.z_ce for d in rep))
    return {
        "天数": len(rep),
        "Z_online": z_online,
        "Z_ce": z_ce,
        "执行层增益": z_ce - z_online,
        "Z_expost": float(sum(d.z_expost for d in rep)),
        "Z_pi": float(sum(d.z_pi for d in rep)),
        "EVPI": z_online - float(sum(d.z_pi for d in rep)),
        "日前信息损失": float(sum(d.z_expost - d.z_pi for d in rep)),
        "实时信息损失": float(sum(d.z - d.z_expost for d in rep)),
        "基础购电费": float(sum(base_fee)),
        "偏差费": float(sum(dev_fee)),
        "偏差费占比": float(sum(dev_fee) / z_online),
        "紧急购电费合计": float(sum(emerg_fee)),
        "紧急费占比": float(sum(emerg_fee) / z_online),
        "δ+ 合计": float(dp_all.sum()),
        "δ− 合计": float(dm_all.sum()),
        "方向比 δ+/δ−": float(dp_all.sum() / dm_all.sum())
        if dm_all.sum() > 1e-9 else float("inf"),
        "调整行使率": float((adj_all > EPS_EMERG).mean()),
        "C3-7 前缀偏差 max": float(max(c["c3_7"] for c in checks)),
        "C6 对账残差 max": float(max(c["c6"] for c in checks)),
        "C7 能量守恒残差 max": float(max(c["c7"] for c in checks)),
        "C8 max(c·d)": float(max(c["cd_max"] for c in checks)),
        "C8 max(u·w)": float(max(c["uw_max"] for c in checks)),
        "u∧c 时段数": int(sum(c["u_and_c"] for c in checks)),
        "C9 序关系违例天数": int(sum(c["c9_bad"] for c in checks)),
        "C3-3 结算式恒等残差": beta_check(),
        "紧急购电天数": int((u_all > EPS_EMERG).any(axis=1).sum()),
        "紧急购电时段数": int((u_all > EPS_EMERG).sum()),
        "弃电量合计": float(sum(d.w.sum() for d in rep)),
        "弃电率": float(sum(d.w.sum() for d in rep)
                        / max(float(year.r[[d.j for d in rep]].sum()), 1e-9)),
        "SOC 均值": float(np.mean([d.S for d in rep])),
        "SOC 范围": (float(min(d.S.min() for d in rep)),
                     float(max(d.S.max() for d in rep))),
        "负载预报 MAE": float(err_l.mean()),
        "光伏预报 MAE": float(err_r.mean()),
    }


def metrics_row(label: str, epochs: tuple[int, ...], summary: dict
                ) -> dict[str, float | int | str]:
    """summarize 输出压平为聚合 CSV 行（snake 键，供消融/阶梯汇总）。"""
    return {
        "label": label,
        "epochs": "-".join(str(6 * m) for m in epochs),
        "days": summary["天数"],
        "z_online": summary["Z_online"],
        "z_ce": summary["Z_ce"],
        "smpc_gain": summary["执行层增益"],
        "z_expost": summary["Z_expost"],
        "z_pi": summary["Z_pi"],
        "evpi": summary["EVPI"],
        "base_fee": summary["基础购电费"],
        "dev_fee": summary["偏差费"],
        "dev_share": round(summary["偏差费占比"], 6),
        "emerg_fee": summary["紧急购电费合计"],
        "emerg_share": round(summary["紧急费占比"], 6),
        "dp_sum": summary["δ+ 合计"],
        "dm_sum": summary["δ− 合计"],
        "curtail_kwh": summary["弃电量合计"],
        "curtail_rate": round(summary["弃电率"], 6),
        "soc_mean": summary["SOC 均值"],
    }


class AggregateCsv:
    """消融网格 / 逐级阶梯聚合 CSV（ablation.csv、ladder.csv 同一格式）。"""

    def __init__(self, filename: str, rows: list[dict]):
        self.filename = filename
        self.rows = rows

    def export(self, out_dir: Path) -> Path:
        """写出聚合 CSV 并返回路径。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / self.filename
        pd.DataFrame(self.rows).to_csv(path, index=False,
                                       float_format="%.4f")
        return path


class Result3Xlsx:
    """按官方模板行序填写 result3.xlsx（四表，行号对位不改标签）。"""

    def __init__(self, template: Path = TEMPLATE):
        self.template = template

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """写出 result3.xlsx（计划 / 调整 / 充放电 / 紧急购电）并返回路径。"""
        rep = report_days(days)
        wb = openpyxl.load_workbook(self.template)
        for name, arr, fee in (
                ("计划购电量", [d.g for d in rep],
                 [float(p @ d.g) for d in rep]),
                ("调整购电量", [d.a for d in rep],
                 [float(p @ d.a) + 0.5 * float(p @ np.abs(d.a - d.g))
                  for d in rep])):
            ws = wb[name]
            for i, (day, row) in enumerate(zip(rep, arr)):
                r = 2 + i
                for t, val in enumerate(row):
                    ws.cell(row=r, column=2 + t, value=float(val))
                ws.cell(row=r, column=146, value=float(row.sum()))
                ws.cell(row=r, column=147, value=float(fee[i]))
        ws = wb["充放电量"]
        ws.delete_rows(2, ws.max_row - 1)
        for day in rep:
            row = ws.max_row + 1
            for k in range(6):
                seg = slice(24 * k, 24 * (k + 1))
                ws.cell(row=row + k, column=2, value=BLOCK_LABELS[k])
                ws.cell(row=row + k, column=3,
                        value=float(ETA * day.c[seg].sum()))
                ws.cell(row=row + k, column=4,
                        value=float(day.d[seg].sum() / ETA))
            ws.cell(row=row, column=1, value=_py_date(day, year))
            ws.cell(row=row, column=5, value=dt.time(0, 0))
            ws.cell(row=row, column=6, value=float(day.S[0]))
            ws.cell(row=row + 1, column=5, value="24:00")
            ws.cell(row=row + 1, column=6, value=float(day.S[T]))
        ws = wb["紧急购电量"]
        ws.delete_rows(2, ws.max_row - 1)
        for day in rep:
            segs = emergency_segments(day.u)
            row = ws.max_row + 1
            for k, (i_a, i_b, e) in enumerate(segs):
                ws.cell(row=row + k, column=2,
                        value=segment_label(i_a, i_b))
                ws.cell(row=row + k, column=3, value=e)
            if segs:
                ws.cell(row=row, column=1, value=_py_date(day, year))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result3.xlsx"
        wb.save(path)
        return path


class DetailsCsv:
    """报告期逐日明细 CSV：日期、三层费用、偏差量、紧急/弃电、SOC。"""

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """写出 q3_details.csv 并返回路径。"""
        rows = []
        for day in report_days(days):
            dev = np.abs(day.a - day.g)
            rows.append({
                "date": str(_py_date(day, year)),
                "z_online": day.z, "z_ce": day.z_ce,
                "z_expost": day.z_expost, "z_pi": day.z_pi,
                "g_sum": float(day.g.sum()), "a_sum": float(day.a.sum()),
                "base_fee": float(p @ day.a),
                "dev_fee": 0.5 * float(p @ dev),
                "dp_sum": float(dev[day.a > day.g].sum()),
                "dm_sum": float(dev[day.a < day.g].sum()),
                "u_sum": float(day.u.sum()),
                "emerg_fee": float(KAPPA * (p @ day.u)),
                "w_sum": float(day.w.sum()),
                "S0": float(day.S[0]), "S144": float(day.S[T]),
                "mae_ell": float(np.abs(day.ell_hat - year.ell[day.j]).mean()),
                "mae_r": float(np.abs(day.r_hat - year.r[day.j]).mean()),
            })
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q3_details.csv"
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
        return path


class SummaryMd:
    """生成题3 结果摘要 markdown（配置、三块费用、调整统计、自检、
    紧急购电、表 1 计划/调整并列、口径说明）。"""

    def __init__(self, cfg_label: str, summary: dict):
        self.cfg_label = cfg_label
        self.summary = summary

    def export(self, days: list[DayResult], year: YearData, p: np.ndarray,
               out_dir: Path) -> Path:
        """写出 q3_summary.md 并返回路径。"""
        s = self.summary
        money = lambda x: f"{x:,.2f} 元"
        core = md_table(["指标", "数值"], [
            ("Z^online（交付）", money(s["Z_online"])),
            ("Z^CE（确定性等价对照，α^rt=0.65）", money(s["Z_ce"])),
            ("执行层增益 Z^CE−Z^online", money(s["执行层增益"])),
            ("Z^ex-post", money(s["Z_expost"])),
            ("Z^PI", money(s["Z_pi"])),
            ("EVPI", money(s["EVPI"])),
            ("日前信息损失", money(s["日前信息损失"])),
            ("实时信息损失", money(s["实时信息损失"])),
            ("基础购电费 Σp·a", money(s["基础购电费"])),
            ("偏差费 ½Σp|a−g| / 占比", f"{money(s['偏差费'])} / "
             f"{s['偏差费占比'] * 100:.2f}%"),
            ("紧急购电费 / 占比", f"{money(s['紧急购电费合计'])} / "
             f"{s['紧急费占比'] * 100:.2f}%"),
            ("弃电量 / 弃电率", f"{s['弃电量合计']:,.2f} kWh / "
             f"{s['弃电率'] * 100:.2f}%"),
            ("SOC 均值 / 范围", f"{s['SOC 均值']:,.1f} kWh / "
             f"{s['SOC 范围'][0]:,.1f} ~ {s['SOC 范围'][1]:,.1f}"),
        ])
        adj = md_table(["指标", "数值"], [
            ("δ+ 合计（上调）", f"{s['δ+ 合计']:,.1f} kWh"),
            ("δ− 合计（下调）", f"{s['δ− 合计']:,.1f} kWh"),
            ("方向比 δ+/δ−", f"{s['方向比 δ+/δ−']:,.1f}"
             if np.isfinite(s["方向比 δ+/δ−"]) else "∞"),
            ("调整行使率（t≥36 时段）", f"{s['调整行使率'] * 100:.1f}%"),
        ])
        checks = md_table(["检查项", "数值"], [
            (k, f"{s[k]:.6f}" if isinstance(s[k], float) else s[k])
            for k in ("C3-3 结算式恒等残差", "C3-7 前缀偏差 max",
                      "C6 对账残差 max", "C7 能量守恒残差 max",
                      "C8 max(c·d)", "C8 max(u·w)", "u∧c 时段数",
                      "C9 序关系违例天数")])
        tab_rows = []
        for date in PAPER_DATES:
            day = next((d for d in days if str(_py_date(d, year)) == date),
                       None)
            if day is None:
                continue
            tab_rows.append((date,
                             *[f"{day.g[t - 1]:.2f}" for t in PAPER_SLOTS],
                             *[f"{day.a[t - 1]:.2f}" for t in PAPER_SLOTS],
                             f"{day.g.sum():.2f}", f"{day.a.sum():.2f}",
                             f"{day.z:.2f}"))
        table1 = md_table(
            ["日期"] + [f"计划 t={t}" for t in PAPER_SLOTS]
            + [f"调整 t={t}" for t in PAPER_SLOTS]
            + ["全天计划", "全天调整", "全天购电费"], tab_rows)
        text = f"""# 问题 3 求解结果（{self.cfg_label}）

## 费用分层（报告期 {s['天数']} 天）

{core}

## 调整统计

{adj}

## 自检（文档 §12.4）

{checks}

## 紧急购电统计

- 发生天数 {s['紧急购电天数']} / {s['天数']}，共 {s['紧急购电时段数']} 个时段
- 逐日明细见 q3_details.csv

## 论文表 1（四个指定日期，计划 / 调整并列；全天购电费含偏差与紧急费）

{table1}

## 计算口径

- Z = Σ[p·a + ½p|a−g|] + κΣp·u（现金费用，真实路径 u）；Z̃ 含末端估值，不上报。
- Z^CE 与 Z^online 同 g、同 committed、同日初 SOC，只差内环未来表示
  （当前阶段情景分布 vs 单条 α^rt=0.65 分位路径）——执行层增益即
  "分位压缩把概率压成必然"的代价（题 2 结论在题 3 的同款计量，文档 §12.3）。
  SOC 链按交付臂推进，CE 臂不入跨日反馈。
- 计划表全天购电费为名义计划费 Σp·g；调整表为实际结算购电费（含偏差费、
  不含紧急费）——两列口径不同，分别声明（文档 §9.2）。
- 调整购电量表 B_0 段（0:00–6:00）填计划值（无修订机会，最终量即计划量）。
- 充放电量按电池侧填报（ηc、d/η）；紧急购电按极大连续段合并（ε=10⁻⁶ kWh）。
- 暖机期 1 月完整滚动但不计入报告；跨日以真实路径 SOC 链衔接。
- result3.xlsx 按官方模板行号对位填写，不改动模板标签。
"""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q3_summary.md"
        path.write_text(text, encoding="utf-8")
        return path
