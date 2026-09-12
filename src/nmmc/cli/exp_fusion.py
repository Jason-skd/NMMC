"""融合冷启动实验命令行入口（issue #8 去超参⑤：删 min_days=8 阈值）。

用法：
    uv run expfusion --part a              # A 权重 + 预报质量诊断（秒级）
    uv run expfusion --part b --days 45    # B 回测冒烟（含 31 天暖机）
    uv run expfusion                       # 全部（B 为全年回测，重）
"""
import argparse
from pathlib import Path

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import load_year
from nmmc.model.forecast3 import Q3Config
from nmmc.run.fusion_coldstart import (backtest_compare, point_quality,
                                       weights_table, write_summary)


def build_parser() -> argparse.ArgumentParser:
    """构建 expfusion 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="融合冷启动实验：删 min_days=8 阈值后的年初段诊断")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all",
                        help="实验部分：a 权重/质量诊断 / b 回测对照 / all 全部")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件2.xlsx",
                        help="附件2 数据路径")
    parser.add_argument("--forecast", type=Path, default=ATTACH / "附件3.xlsx",
                        help="附件3 光伏预报路径")
    parser.add_argument("--attach1", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径（电价与典型日先验）")
    parser.add_argument("--baseline", type=Path,
                        default=Path("results/q3/F1-L1-LU1-S-b/q3_details.csv"),
                        help="main 基线明细 CSV（Part B 对照）")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/fusion-coldstart"),
                        help="结果输出目录")
    parser.add_argument("--view-days", type=int, default=14,
                        help="A1 权重诊断的天数窗（自 j=0 起）")
    parser.add_argument("--load", choices=["L1", "L2", "L3"], default="L1",
                        help="负载日前基线候选（随题2 消融胜者）")
    parser.add_argument("--load-update", choices=["LU1", "LU0"],
                        default="LU1", help="负载日内更新候选")
    parser.add_argument("--pool", choices=["S-a", "S-b", "S-c"],
                        default="S-b", help="残差池候选")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--days", type=int, default=None,
                        help="回测天数（含暖机期）；缺省全年，冒烟用 45")
    return parser


def main(argv: list[str] | None = None) -> None:
    """expfusion 主流程：Part A 诊断 → Part B 回测对照 → 摘要。"""
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    fc = load_forecast(args.forecast)
    cfg = Q3Config(load=args.load, load_update=args.load_update,
                   pool=args.pool, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Part A：逐日权重表 + 受影响日点预报质量（纯 numpy，秒级）
    weights = quality = None
    if args.part in {"a", "all"}:
        weights = weights_table(year, fc, args.view_days)
        weights.to_csv(args.output_dir / "coldstart_weights.csv", index=False)
        quality = point_quality(cfg, year, fc, day1.r)
        quality.to_csv(args.output_dir / "point_quality.csv", index=False)
        print(f"已导出 coldstart_weights.csv 与 point_quality.csv", flush=True)

    # Part B：全年回测对照（重；逐日明细与暖机段表落盘）
    daily = warmup = None
    if args.part in {"b", "all"}:
        daily, warmup = backtest_compare(year, fc, day1.p, day1.ell, day1.r,
                                         cfg, args.baseline, days=args.days)
        daily.to_csv(args.output_dir / "backtest_daily.csv", index=False)
        print(f"已导出 backtest_daily.csv", flush=True)

    path = write_summary(args.output_dir, weights, quality, daily, warmup)
    print(f"已导出 {path}")


if __name__ == "__main__":
    main()
