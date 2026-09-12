"""ν 锚稳健性实验命令行入口（附录对照：逐日对偶 vs 常数锚 0.472）。

用法：
    uv run expnu --part duals          # Part A：全年逐日对偶测量（分钟级）
    uv run expnu --part sweep          # Part B：地标扫参 + V2b 换装（5 次全年）
    uv run expnu                       # 全部（duals 缓存命中则跳过重测）
    uv run expnu --days 40             # 扫参冒烟（含 31 天暖机）
"""
import argparse
from pathlib import Path

import pandas as pd

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig
from nmmc.run.nu_anchor import (cycle_dual, measure_duals, run_sweep,
                                summarize_duals, write_summary)


def build_parser() -> argparse.ArgumentParser:
    """构建 expnu 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="ν 锚稳健性实验：逐日对偶 vs 常数锚 0.472 对照")
    parser.add_argument("--part", choices=["duals", "sweep", "all"],
                        default="all", help="实验部分：duals 测量 / sweep "
                        "承重检验 / all 全部")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件2.xlsx",
                        help="附件2 数据路径")
    parser.add_argument("--attach1", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径（电价与典型日先验）")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/nu-anchor"),
                        help="结果输出目录")
    parser.add_argument("--load", choices=["L1", "L2", "L3"], default="L1",
                        help="负载点预报候选")
    parser.add_argument("--pv", choices=["P1", "P2", "P3"], default="P1",
                        help="光伏点预报候选")
    parser.add_argument("--resid", choices=["R1", "R2"], default="R2",
                        help="残差场景配置")
    parser.add_argument("--ns", type=int, default=25,
                        help="R1 抽样情景数 N_s")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--days", type=int, default=None,
                        help="回测天数（含暖机期）；缺省全年，冒烟用 40")
    return parser


def main(argv: list[str] | None = None) -> None:
    """expnu 主流程：Part A 对偶测量（可缓存）→ Part B 承重检验 → 摘要。

    Args:
        argv: 命令行参数；缺省取 sys.argv。
    """
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    cfg = ForecastConfig(load=args.load, pv=args.pv, resid=args.resid,
                         ns=args.ns, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    duals_path = args.output_dir / "duals.csv"

    # 方法学自检：典型日对偶应复现题1 档的 0.472（锚的来历核对）
    anchor_check = cycle_dual(day1.p, day1.ell, day1.r)
    print(f"典型日对偶复现 = {anchor_check:.4f}（题1 档报 0.472）", flush=True)

    # Part A：duals.csv 存在且只需 sweep 时缓存命中，否则测量并落盘
    if args.part in {"duals", "all"} or not duals_path.exists():
        duals = measure_duals(year, day1.p, cfg, day1.ell, day1.r)
        duals.to_csv(duals_path, index=False)
        print(f"已导出 {duals_path}", flush=True)
    else:
        duals = pd.read_csv(duals_path)
        print(f"缓存命中 {duals_path}", flush=True)

    stats = summarize_duals(duals, day1.p, anchor_check)

    # Part B：扫参 + 换装（每次全年回测，逐行落盘防中断）
    sweep = None
    if args.part in {"sweep", "all"}:
        sweep = run_sweep(year, day1.p, day1.ell, day1.r, cfg,
                          duals["nu_next_fcst"].to_numpy(),
                          args.output_dir, days=args.days)
        print(f"已导出 {args.output_dir / 'sweep.csv'}", flush=True)

    path = write_summary(args.output_dir, stats, sweep)
    print(f"已导出 {path}")


if __name__ == "__main__":
    main()
