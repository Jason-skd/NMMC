"""问题3 命令行入口。

用法：
    uv run prob3                                   # 默认 F1×L1×LU1×S-b 全年
    uv run prob3 --fusion PB0 --pool S-c           # 消融网格某一组
    uv run prob3 --epochs 0,6,12                   # 逐级增量（文档 §12.2）
    uv run prob3 --ladder --output-dir results/q3-epochs
                                                  # 四档阶梯 + ladder.csv
    uv run prob3 --grid --jobs 4 --output-dir results/q3-ablation
                                                  # 12 组消融 + ablation.csv
    uv run prob3 --days 45                         # 快速冒烟（含 31 天暖机）
    uv run prob3 --nu-end 0.6                      # ν^end 敏感性扫描
                                                  # （内环为随机 MPC，无 α^rt；
                                                  #  CE 对照臂固定 0.65 冻结）
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import DAYS, load_year
from nmmc.model.forecast3 import Q3Config
from nmmc.model.q2 import KAPPA, NU_END
from nmmc.report.plot.q2 import ForecastQuality
from nmmc.report.plot.q3 import (CostWaterfall, EpochsLadder, FusionWeights,
                                 Q2Q3Delta, RevisionDemo, RevisionHeatmap)
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.report.q3 import (TEMPLATE, AggregateCsv, DetailsCsv, Result3Xlsx,
                            SummaryMd, metrics_row, summarize)
from nmmc.run.q3 import rollout

GRID_AXES = (("F1", "PB0"), ("LU1", "LU0"), ("S-a", "S-b", "S-c"))
LADDER_LEVELS = ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3))


def parse_epochs(text: str) -> tuple[int, ...]:
    """把 '0,6,12'（修订时刻，小时）解析为 0 基阶段号元组。"""
    try:
        hours = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"epochs 含非整数：{text!r}") from exc
    if not hours or hours[0] != 0 or set(hours) - {0, 6, 12, 18}:
        raise argparse.ArgumentTypeError(
            f"epochs 须为 0,6,12,18 的含 0 子集，收到 {text!r}")
    return tuple(h // 6 for h in hours)


def build_parser() -> argparse.ArgumentParser:
    """构建 prob3 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="C 题问题3：分块两阶段场景 LP + 嵌套 MPC 全年滚动回测")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件2.xlsx",
                        help="附件2 数据路径")
    parser.add_argument("--forecast", type=Path, default=ATTACH / "附件3.xlsx",
                        help="附件3 光伏预报路径")
    parser.add_argument("--attach1", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径（电价与典型日先验）")
    parser.add_argument("--template", type=Path, default=TEMPLATE,
                        help="result3.xlsx 官方模板路径")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/q3"), help="结果输出根目录")
    parser.add_argument("--load", choices=["L1", "L2", "L3"], default="L1",
                        help="负载日前基线候选（随题2 消融胜者）")
    parser.add_argument("--fusion", choices=["F1", "PB0"], default="F1",
                        help="光伏融合候选：F1 反方差融合 / PB0 官方直用")
    parser.add_argument("--load-update", choices=["LU1", "LU0"], default="LU1",
                        help="负载日内更新：LU1 比例缩放 / LU0 静态")
    parser.add_argument("--pool", choices=["S-a", "S-b", "S-c"], default="S-b",
                        help="残差池：S-b 提前期桶分池 / S-a 按阶段整条 / S-c 混池")
    parser.add_argument("--nu-end", type=float, default=NU_END,
                        help="日末储电量估值 ν^end")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--days", type=int, default=DAYS,
                        help="滚动天数（含暖机期）")
    parser.add_argument("--epochs", type=parse_epochs, default=(0, 1, 2, 3),
                        help="行使修订权的时刻（小时，含 0 子集），如 0,6,12")
    parser.add_argument("--grid", action="store_true",
                        help="跑完 12 组消融（F1/PB0 × LU × S）并汇总 ablation.csv")
    parser.add_argument("--ladder", action="store_true",
                        help="按 {0}/{0,6}/{0,6,12}/{0,6,12,18} 逐级滚动并汇总 ladder.csv")
    parser.add_argument("--jobs", type=int, default=1,
                        help="并行进程数（grid/ladder 模式）")
    parser.add_argument("--q2-details", type=Path,
                        default=Q2Q3Delta.DEFAULT_Q2,
                        help="题2 明细 CSV（q3_vs_q2 费用差图的数据源）")
    return parser


def _config_label(args: argparse.Namespace, epochs: tuple[int, ...],
                  fusion: str | None = None, lu: str | None = None,
                  pool: str | None = None) -> str:
    """结果目录/摘要标签：候选轴 + 非全集时的 epochs 后缀。"""
    base = (f"{fusion or args.fusion}-{args.load}-{lu or args.load_update}"
            f"-{pool or args.pool}")
    if tuple(epochs) != (0, 1, 2, 3):
        base += "-E" + "-".join(str(6 * m) for m in epochs)
    return base


def _payload(args: argparse.Namespace, label: str, epochs: tuple[int, ...],
             fusion: str | None = None, lu: str | None = None,
             pool: str | None = None, out_dir: Path | None = None
             ) -> dict:
    """grid/ladder 工作单元：自足（子进程内重载数据）。"""
    return {
        "input": str(args.input), "forecast": str(args.forecast),
        "attach1": str(args.attach1), "template": str(args.template),
        "load": args.load, "fusion": fusion or args.fusion,
        "load_update": lu or args.load_update, "pool": pool or args.pool,
        "seed": args.seed, "nu_end": args.nu_end,
        "days": args.days, "epochs": list(epochs), "label": label,
        "out_dir": str(args.output_dir if out_dir is None else out_dir),
    }


def _run_one(payload: dict) -> dict:
    """单个配置的完整流程：加载 → 滚动 → 导出三件套 → 返回聚合行。"""
    day1 = load_day1(Path(payload["attach1"]))
    year = load_year(Path(payload["input"]))
    fc = load_forecast(Path(payload["forecast"]))
    cfg = Q3Config(load=payload["load"], fusion=payload["fusion"],
                   load_update=payload["load_update"], pool=payload["pool"],
                   seed=payload["seed"])
    label = payload["label"]
    epochs = tuple(payload["epochs"])
    print(f"[{label}] 滚动 {payload['days']} 天（epochs={epochs}）…",
          flush=True)
    days = rollout(year, fc, day1.p, day1.ell, day1.r, cfg,
                   nu_end=payload["nu_end"],
                   days=payload["days"], epochs=epochs)
    summary = summarize(days, year, day1.p)
    out_dir = Path(payload["out_dir"])
    for path in (Result3Xlsx(Path(payload["template"])).export(
                     days, year, day1.p, out_dir),
                 DetailsCsv().export(days, year, day1.p, out_dir),
                 SummaryMd(label, summary).export(days, year, day1.p,
                                                  out_dir)):
        print(f"已导出 {path}", flush=True)
    print(f"[{label}] Z={summary['Z_online']:,.2f} 元  "
          f"偏差={summary['偏差费']:,.2f}  "
          f"紧急={summary['紧急购电费合计']:,.2f} "
          f"({summary['紧急费占比'] * 100:.2f}%)", flush=True)
    return metrics_row(label, epochs, summary)


def _run_payloads(payloads: list[dict], jobs: int) -> list[dict]:
    """按输入顺序执行工作单元；jobs > 1 时进程池并行。"""
    if jobs <= 1:
        return [_run_one(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_run_one, payloads))


def main(argv: list[str] | None = None) -> None:
    """prob3 主流程：加载 → 全年滚动回测 → 打印诊断 → 导出 result3 与摘要。

    --grid / --ladder 模式下逐配置导出三件套并汇总聚合 CSV。
    """
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    fc = load_forecast(args.forecast)
    cfg = Q3Config(load=args.load, fusion=args.fusion,
                   load_update=args.load_update, pool=args.pool,
                   seed=args.seed)
    label = _config_label(args, args.epochs)

    if args.grid:
        payloads = []
        for fusion, lu, pool in product(*GRID_AXES):
            sub = _config_label(args, args.epochs, fusion=fusion, lu=lu,
                                pool=pool)
            payloads.append(_payload(args, sub, args.epochs, fusion=fusion,
                                     lu=lu, pool=pool,
                                     out_dir=args.output_dir / sub))
        print(f"消融网格 {len(payloads)} 组 × {args.days} 天"
              f"（epochs={args.epochs}，jobs={args.jobs}）", flush=True)
        rows = _run_payloads(payloads, args.jobs)
        path = AggregateCsv("ablation.csv", rows).export(args.output_dir)
        print(f"消融表: {path}")
        return

    if args.ladder:
        payloads = []
        for level in LADDER_LEVELS:
            tag = "E" + "-".join(str(6 * m) for m in level)
            sub = _config_label(args, level)
            payloads.append(_payload(args, sub, level,
                                     out_dir=args.output_dir / tag))
        print(f"逐级阶梯 {len(payloads)} 档 × {args.days} 天"
              f"（jobs={args.jobs}）", flush=True)
        rows = _run_payloads(payloads, args.jobs)
        prev = None
        for row in rows:
            row["level"] = "E" + row["epochs"]
            row["delta_z"] = ("" if prev is None
                              else round(row["z_online"] - prev, 4))
            prev = row["z_online"]
        path = AggregateCsv("ladder.csv", rows).export(args.output_dir)
        print(f"阶梯表: {path}")
        fig = EpochsLadder(rows).export(args.output_dir)
        print(f"已导出 {fig}")
        return

    print(f"配置 {label}，ν^end = {args.nu_end:g}，κ = {KAPPA:g}，"
          f"滚动 {args.days} 天（暖机 {WARMUP_DAYS} 天）")
    days = rollout(year, fc, day1.p, day1.ell, day1.r, cfg,
                   nu_end=args.nu_end,
                   days=args.days, epochs=args.epochs)

    summary = summarize(days, year, day1.p)
    print("== 费用分层（报告期）==")
    for key in ("Z_online", "Z_ce", "执行层增益", "Z_expost", "Z_pi",
                "EVPI", "日前信息损失", "实时信息损失",
                "基础购电费", "偏差费", "紧急购电费合计"):
        print(f"  {key}: {summary[key]:,.2f} 元")
    print("== 调整统计 ==")
    for key in ("δ+ 合计", "δ− 合计", "方向比 δ+/δ−", "调整行使率"):
        print(f"  {key}: {summary[key]}")
    print("== 自检 ==")
    for key in ("C3-3 结算式恒等残差", "C3-7 前缀偏差 max",
                "C6 对账残差 max", "C7 能量守恒残差 max", "C8 max(c·d)",
                "C8 max(u·w)", "u∧c 时段数", "C9 序关系违例天数"):
        print(f"  {key}: {summary[key]}")
    print("== 紧急购电 ==")
    print(f"  天数 {summary['紧急购电天数']}/{summary['天数']}，"
          f"时段数 {summary['紧急购电时段数']}，弃电率 "
          f"{summary['弃电率'] * 100:.2f}%")

    out_dir = args.output_dir / label
    exports = (
        Result3Xlsx(args.template).export(days, year, day1.p, out_dir),
        DetailsCsv().export(days, year, day1.p, out_dir),
        SummaryMd(label, summary).export(days, year, day1.p, out_dir),
        RevisionDemo().export(days, year, day1.p, out_dir),
        CostWaterfall().export(days, year, day1.p, out_dir),
        FusionWeights().export(year, fc, out_dir),
        ForecastQuality(prefix="q3", subject="问题 3").export(
            days, year, day1.p, out_dir),
        RevisionHeatmap().export(days, year, day1.p, out_dir),
        Q2Q3Delta(args.q2_details).export(days, year, day1.p, out_dir),
    )
    for path in exports:
        print(f"已导出 {path}")


if __name__ == "__main__":
    main()
