"""问题3 命令行入口。

用法：
    uv run prob3                                   # 默认 F1×L1×LU1×S-b 全年
    uv run prob3 --fusion PB0 --pool S-c           # 消融网格某一组
    uv run prob3 --days 45                         # 快速冒烟（含 31 天暖机）
    uv run prob3 --alpha-rt 0.6 --nu-end 0.6       # 敏感性扫描
"""
import argparse
from pathlib import Path

from nmmc.data import ATTACH, load_day1
from nmmc.q2.data import ALPHA_RT, DAYS, KAPPA, NU_END, WARMUP_DAYS, load_year
from nmmc.q3.data import load_forecast
from nmmc.q3.execute import rollout
from nmmc.q3.forecast import Q3Config
from nmmc.q3.report import (TEMPLATE, DetailsCsv, Result3Xlsx, SummaryMd,
                            summarize)


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
    parser.add_argument("--pool", choices=["S-b", "S-a", "S-c"], default="S-b",
                        help="残差池：S-b 提前期桶分池 / S-a 按阶段整条 / S-c 混池")
    parser.add_argument("--alpha-rt", type=float, default=ALPHA_RT,
                        help="MPC 参考路径分位 α^rt")
    parser.add_argument("--nu-end", type=float, default=NU_END,
                        help="日末储电量估值 ν^end")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--days", type=int, default=DAYS,
                        help="滚动天数（含暖机期）")
    return parser


def main(argv: list[str] | None = None) -> None:
    """prob3 主流程：加载 → 全年滚动回测 → 打印诊断 → 导出 result3 与摘要。

    Args:
        argv: 命令行参数；缺省取 sys.argv。
    """
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    fc = load_forecast(args.forecast)
    cfg = Q3Config(load=args.load, fusion=args.fusion,
                   load_update=args.load_update, pool=args.pool,
                   seed=args.seed)
    label = f"{cfg.fusion}-{cfg.load}-{cfg.load_update}-{cfg.pool}-a{args.alpha_rt:g}"
    print(f"配置 {label}，ν^end = {args.nu_end:g}，κ = {KAPPA:g}，"
          f"滚动 {args.days} 天（暖机 {WARMUP_DAYS} 天）")
    days = rollout(year, fc, day1.p, day1.ell, day1.r, cfg,
                   nu_end=args.nu_end, alpha_rt=args.alpha_rt,
                   days=args.days)

    summary = summarize(days, year, day1.p)
    print("== 费用分层（报告期）==")
    for key in ("Z_online", "Z_expost", "Z_pi", "EVPI",
                "日前信息损失", "实时信息损失",
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
    for path in (Result3Xlsx(args.template).export(days, year, day1.p, out_dir),
                 DetailsCsv().export(days, year, day1.p, out_dir),
                 SummaryMd(label, summary).export(days, year, day1.p,
                                                  out_dir)):
        print(f"已导出 {path}")


if __name__ == "__main__":
    main()
