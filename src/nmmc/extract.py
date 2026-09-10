"""结果导出：统一 Extractor 协议，每个产物一个提取器。

表1/表2 行与诊断量也在此提供——它们是从最优解"提取信息"的公共层，
命令行打印与文件导出共用。
"""
from pathlib import Path
from typing import Protocol

import numpy as np
import openpyxl
import pandas as pd

from nmmc.data import DayData
from nmmc.lp import ETA, Solution, T

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs" / "附件" / "附件5" / "result1.xlsx"

PAPER_SLOTS = [6 * h + 1 for h in (10, 12, 14, 16, 18, 20)]  # t = 61..121


class Extractor(Protocol):
    """导出器协议：从最优解导出一份产物。

    每个实现类负责一种产物（xlsx / csv / png / md），CLI 组装提取器列表
    依次运行；结构化满足协议即可，无需显式继承。
    """

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """写出产物并返回文件路径。"""
        ...


def interval_label(t: int) -> str:
    """返回时段 t 的物理区间标签。

    Args:
        t: 时段编号，1..144（标签为时段末点，t=1 即 [00:00, 00:10)）。

    Returns:
        区间标签字符串，如 "00:00-00:10"。
    """
    def hm(m):
        return f"{m // 60:02d}:{m % 60:02d}"
    return f"{hm((t - 1) * 10)}-{hm(t * 10)}"


def table1_rows(sol: Solution) -> list[tuple[str, float]]:
    """组装论文表 1 行：6 个指定时段购电量 + 全天购电量 / 购电费。

    Returns:
        (标签, 数值) 列表；数值单位 kWh，末两行为元。
    """
    rows = [(interval_label(t), sol.g[t - 1]) for t in PAPER_SLOTS]
    rows += [("全天购电量", sol.g.sum()), ("全天购电费", sol.cost)]
    return rows


def table2_rows(sol: Solution) -> list[tuple[str, float, float]]:
    """组装论文表 2 行：6 个 4h 块的 (充电量, 放电量)，电池侧口径。

    Returns:
        (块标签, 充电量 η·Σc, 放电量 Σd/η) 列表，kWh。
    """
    rows = []
    for k in range(6):
        seg = slice(24 * k, 24 * (k + 1))
        rows.append((f"{4 * k}:00-{4 * (k + 1)}:00",
                     ETA * sol.c[seg].sum(), sol.d[seg].sum() / ETA))
    return rows


def diagnostics(sol: Solution, data: DayData) -> dict:
    """计算诊断量：基准费用、弃电、互斥、损耗、能量平衡、SOC 与对偶检查。

    Args:
        sol: LP 最优解。
        data: 当日输入数据。

    Returns:
        dict：键为检查项名，值为数值或 (min, max) 元组。
    """
    out = {
        "Z_base 无电池直买": float(data.p @ np.maximum(data.ell - data.r, 0.0)),
        "弃电量 Σw": float(sol.w.sum()),
        "逐时段 w ≤ r": bool((sol.w <= data.r + 1e-9).all()),
        "充放互斥 max(c·d)": float((sol.c * sol.d).max()),
        "全天损耗": float((1 - ETA) * sol.c.sum() + (1 / ETA - 1) * sol.d.sum()),
        "能量平衡残差": float(
            np.abs(sol.g + data.r + sol.d - data.ell - sol.c - sol.w).max()),
        "SOC 范围": (float(sol.S.min()), float(sol.S.max())),
        "循环误差 |S144-S0|": float(abs(sol.S[T] - sol.S[0])),
    }
    if sol.lam is not None:
        buying = sol.g > 1e-9
        out["对偶 λ=p 偏差(购电时段)"] = float(
            np.abs(sol.lam[buying] - data.p[buying]).max())
        out["对偶 λ 范围"] = (float(sol.lam.min()), float(sol.lam.max()))
        out["对偶 ν 范围"] = (float(sol.nu.min()), float(sol.nu.max()))
    return out


def _fmt(val) -> str:
    """把诊断值格式化为字符串：元组转 "a ~ b"，其余保留 4 位小数。"""
    if isinstance(val, tuple):
        return f"{val[0]:.4f} ~ {val[1]:.4f}"
    return f"{val:.4f}"


def _md_table(headers: list[str], rows: list[tuple]) -> str:
    """把表头与数据行渲染为 markdown 表格文本。"""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


class Result1Xlsx:
    """按官方模板行序填写 result1.xlsx（模板行 i ↔ 时段 t=i，不改标签）。

    Attributes:
        template: result1.xlsx 官方模板路径。
    """

    def __init__(self, template: Path = TEMPLATE):
        self.template = template

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """写出 result1.xlsx（计划购电量 + 充放电量 + 储电量）并返回路径。"""
        wb = openpyxl.load_workbook(self.template)
        ws = wb["计划购电量"]
        for i, val in enumerate(sol.g):
            ws.cell(row=2 + i, column=2, value=float(val))
        ws = wb["充放电量"]
        for k, (_, ch_bat, dis_bat) in enumerate(table2_rows(sol)):
            ws.cell(row=2 + k, column=2, value=float(ch_bat))
            ws.cell(row=2 + k, column=3, value=float(dis_bat))
        ws.cell(row=2, column=5, value=float(sol.S[0]))   # 0:00 储电量
        ws.cell(row=3, column=5, value=float(sol.S[T]))   # 24:00 储电量
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "result1.xlsx"
        wb.save(path)
        return path


class DetailsCsv:
    """逐时段调度明细 CSV：t、时间段、g/c/d/w、S、λ/ν。"""

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """写出 q1_details.csv 并返回路径。"""
        df = pd.DataFrame({
            "t": np.arange(1, T + 1),
            "时间段": [interval_label(t) for t in range(1, T + 1)],
            "g": sol.g, "c": sol.c, "d": sol.d, "w": sol.w,
            "S": sol.S[1:], "lam": sol.lam if sol.lam is not None else "",
            "nu": sol.nu if sol.nu is not None else "",
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q1_details.csv"
        df.to_csv(path, index=False, float_format="%.6f")
        return path


class SummaryMd:
    """生成结果摘要 markdown（核心指标、表 1/表 2、数值验证、口径说明）。

    Attributes:
        diag: diagnostics() 的输出；空则验证节为空表。
        baselines: 对照基准（Z^6000 / S_0 决策化收益 / Z^rule / 规则节省比例）；
            空则只报主结果。
    """

    def __init__(self, diag: dict | None = None,
                 baselines: dict[str, float] | None = None):
        self.diag = diag or {}
        self.baselines = baselines or {}

    def extract(self, data: DayData, sol: Solution, out_dir: Path) -> Path:
        """写出 q1_summary.md 并返回路径。"""
        z_base = self.diag.get("Z_base 无电池直买", float("nan"))
        saving = z_base - sol.cost
        core = [("最优购电费用 Z*", f"{sol.cost:.4f} 元"),
                ("全天购电量", f"{sol.g.sum():.4f} kWh"),
                ("0:00 / 24:00 储电量", f"{sol.S[0]:.4f} kWh"),
                ("无电池直买基准 Z_base", f"{z_base:.4f} 元"),
                ("相对无电池节省", f"{saving:.4f} 元（{saving / z_base * 100:.2f}%）")]
        for key, label in (("Z^6000", "固定 S_0=6000 的费用 Z^6000"),
                           ("S_0 决策化收益", "S_0 决策化收益"),
                           ("Z^rule", "阈值规则基准 Z^rule"),
                           ("规则节省比例", "相对规则策略节省")):
            if key in self.baselines:
                val = self.baselines[key]
                unit = " %" if key == "规则节省比例" else " 元"
                core.append((label, f"{val:.4f}{unit}"))

        t1 = _md_table(["时间段", "购电量（kWh）"],
                       [(lbl, f"{val:.4f}") for lbl, val in table1_rows(sol)])
        t2 = _md_table(["时间段", "充电量（kWh，电池侧）", "放电量（kWh，电池侧）"],
                       [(lbl, f"{ch:.4f}", f"{dis:.4f}")
                        for lbl, ch, dis in table2_rows(sol)])
        checks = _md_table(["检查项", "数值"],
                           [(k, _fmt(v)) for k, v in self.diag.items()])

        text = f"""# 问题 1 求解结果

## 核心结果

{_md_table(["指标", "数值"], core)}

## 表 1 指定时段购电量

{t1}

## 表 2 储能充放电量（电池侧）

{t2}

0:00 储电量 = 24:00 储电量 = **{sol.S[0]:.4f} kWh**。

## 数值验证

{checks}

## 计算口径

- 附件功率乘 Δ=1/6 后，以 kWh 进入模型。
- g/c/d/w 为母线侧电量；表 2 按电池侧填报 ηc_t 与 d_t/η。
- S_0 为自由决策变量（假设③），另以固定 6000 kWh 重解作敏感性对照。
- λ_t 为供需平衡约束对偶（母线边际价值）；ν_t 为 SOC 递推对偶取"储能边际价值"正号约定（文档 §7）。
- 阈值规则基准：电价低于 35% 分位或光伏盈余时允许充电、高于 65% 分位时允许放电，许可范围内仍求解 LP。
- result1.xlsx 按官方模板行序对位填写，不改动模板标签。
"""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "q1_summary.md"
        path.write_text(text, encoding="utf-8")
        return path
