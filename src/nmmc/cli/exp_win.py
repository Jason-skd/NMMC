"""去超参实验（issue #6）命令行入口：窗口 W_L=4 / W_R=7 改滚动伪样本外
选择（1-SE 规则），另备零参数路线对照。

用法：
    uv run expwin --part forecast     # 选窗序列 + 五臂预报技能（分钟级）
    uv run expwin --part downstream   # 题2 全年双臂 rollout（各约 13 分钟）
    uv run expwin --part q3fc         # 题3 融合层光伏对照（分钟级）
    uv run expwin                     # 全部（downstream 可用缓存）
    uv run expwin --part downstream --days 45   # 承重冒烟（含 31 天暖机）
"""
import argparse
from pathlib import Path

import pandas as pd

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import load_year
from nmmc.model.winselect import LOAD_GRID, PV_GRID, window_schedule
from nmmc.run.win_select import (arm_forecasts, cross_checks, q3_forecast_compare,
                                 run_downstream, skill_table, write_summary)


def build_parser() -> argparse.ArgumentParser:
    """构建 expwin 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="去超参实验（issue #6）：窗长逐日滚动伪样本外选择")
    parser.add_argument("--part", choices=["forecast", "downstream", "q3fc", "all"],
                        default="all", help="实验部分；all 全部")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件2.xlsx",
                        help="附件2 数据路径")
    parser.add_argument("--attach1", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径（电价与典型日先验）")
    parser.add_argument("--forecast", type=Path, default=ATTACH / "附件3.xlsx",
                        help="附件3 光伏预报路径（q3fc 用）")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/win-select"),
                        help="结果输出目录")
    parser.add_argument("--days", type=int, default=None,
                        help="downstream 回测天数（含暖机期）；缺省全年，"
                             "冒烟用 45")
    return parser


def main(argv: list[str] | None = None) -> None:
    """expwin 主流程：选窗序列与交叉验证（必跑）→ 三段分发 → 总览摘要。"""
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 选窗序列（含两族 MAE 曲面）+ 双实现交叉验证——所有段共用
    schedule, mae_l, mae_r = window_schedule(year)
    schedule.to_csv(args.output_dir / "windows.csv", index=False)
    pd.DataFrame(mae_l, columns=[f"w{w}" for w in LOAD_GRID]).to_csv(
        args.output_dir / "mae_surface_load.csv", index=False, float_format="%.4f")
    pd.DataFrame(mae_r, columns=[f"w{w}" for w in PV_GRID]).to_csv(
        args.output_dir / "mae_surface_pv.csv", index=False, float_format="%.4f")
    print(f"已导出 {args.output_dir}/windows.csv 与两张 MAE 曲面", flush=True)
    checks = cross_checks(year, mae_l, mae_r, schedule)
    for line in checks:
        print(f"[对拍] {line}", flush=True)

    skill_md = downstream_md = q3_md = None
    deltas: dict[str, float] | None = None
    q3_verdict: str | None = None
    if args.part in {"forecast", "all"}:
        fcasts = arm_forecasts(year, day1.ell, day1.r, schedule)
        _, skill_md, daily = skill_table(year, fcasts)
        daily.to_csv(args.output_dir / "forecast_daily.csv", index=False)
        rep = schedule.iloc[31:]
        print(f"报告期偏离先验：负载 {int((rep['w_l'] != 4).sum())}/{len(rep)} 天，"
              f"光伏 {int((rep['w_r'] != 7).sum())}/{len(rep)} 天", flush=True)
        print("== 预报层技能（报告期 MAE，kWh/时段）==")
        print(skill_md, flush=True)

    if args.part in {"downstream", "all"}:
        downstream_md, lines, deltas = run_downstream(
            year, day1, schedule, args.output_dir, days=args.days)
        for line in lines:
            print(f"[对拍] {line}", flush=True)
        print("== 题2 全年费用承重（四臂）==")
        print(downstream_md, flush=True)

    if args.part in {"q3fc", "all"}:
        fc = load_forecast(args.forecast)
        q3_md, q3_verdict = q3_forecast_compare(year, fc, day1.r, schedule)
        print(f"== 题3 融合层光伏对照：全阶段平均 Δ = {q3_verdict} ==")
        print(q3_md, flush=True)

    path = write_summary(args.output_dir, schedule, skill_md,
                         downstream_md, deltas, q3_md, q3_verdict, checks)
    print(f"已导出 {path}")


if __name__ == "__main__":
    main()
