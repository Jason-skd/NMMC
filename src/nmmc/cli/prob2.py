"""问题2 命令行入口。

用法：
    uv run prob2                                   # 默认 L1×P1×R2 全年
    uv run prob2 --load L3 --pv P3 --resid R1      # 消融网格某一组
    uv run prob2 --days 45                         # 快速冒烟（含 31 天暖机）
    uv run prob2 --alpha-rt 0.8 --nu-end 0.6       # 敏感性扫描
"""
import argparse
from pathlib import Path

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.year import ALPHA_RT, DAYS, KAPPA, NU_END, WARMUP_DAYS, load_year
from nmmc.model.forecast import ForecastConfig
from nmmc.report.q2 import (TEMPLATE, DetailsCsv, Result2Xlsx, SummaryMd,
                            summarize)
from nmmc.run.q2 import rollout


def build_parser() -> argparse.ArgumentParser:
    """构建 prob2 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="C 题问题2：两阶段场景 LP + 在线 MPC 全年滚动回测")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件2.xlsx",
                        help="附件2 数据路径")
    parser.add_argument("--attach1", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径（电价与典型日先验）")
    parser.add_argument("--template", type=Path, default=TEMPLATE,
                        help="result2.xlsx 官方模板路径")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/q2"), help="结果输出根目录")
    parser.add_argument("--load", choices=["L1", "L2", "L3"], default="L1",
                        help="负载点预报候选")
    parser.add_argument("--pv", choices=["P1", "P2", "P3"], default="P1",
                        help="光伏点预报候选")
    parser.add_argument("--resid", choices=["R1", "R2"], default="R2",
                        help="残差场景配置：R1 28 日放回抽 ns 条 / R2 近窗 12")
    parser.add_argument("--ns", type=int, default=25,
                        help="R1 抽样情景数 N_s")
    parser.add_argument("--alpha-rt", type=float, default=ALPHA_RT,
                        help="MPC 参考路径分位 α^rt")
    parser.add_argument("--nu-end", type=float, default=NU_END,
                        help="日末储电量估值 ν^end")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--days", type=int, default=DAYS,
                        help="滚动天数（含暖机期）")
    return parser


def main(argv: list[str] | None = None) -> None:
    """prob2 主流程：加载 → 全年滚动回测 → 打印诊断 → 导出 result2 与摘要。

    Args:
        argv: 命令行参数；缺省取 sys.argv。
    """
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    cfg = ForecastConfig(load=args.load, pv=args.pv, resid=args.resid,
                         ns=args.ns, seed=args.seed)
    label = f"{cfg.load}-{cfg.pv}-{cfg.resid}-a{args.alpha_rt:g}"
    print(f"配置 {label}，ν^end = {args.nu_end:g}，κ = {KAPPA:g}，"
          f"滚动 {args.days} 天（暖机 {WARMUP_DAYS} 天）")
    days = rollout(year, day1.p, day1.ell, day1.r, cfg,
                   nu_end=args.nu_end, alpha_rt=args.alpha_rt,
                   days=args.days)

    summary = summarize(days, year, day1.p)
    print("== 费用分层（报告期）==")
    for key in ("Z_online", "Z_expost", "Z_pi", "EVPI",
                "日前信息损失", "实时信息损失"):
        print(f"  {key}: {summary[key]:,.2f} 元")
    print("== 自检 ==")
    for key in ("C6 对账残差 max", "C7 能量守恒残差 max", "C8 max(c·d)",
                "C8 max(u·w)", "性质8 u∧c 时段数", "C9 序关系违例天数"):
        print(f"  {key}: {summary[key]}")
    print("== 紧急购电 ==")
    print(f"  天数 {summary['紧急购电天数']}/{summary['天数']}，"
          f"时段数 {summary['紧急购电时段数']}，"
          f"费用 {summary['紧急购电费合计']:,.2f} 元"
          f"（占比 {summary['紧急费占比'] * 100:.2f}%）")

    out_dir = args.output_dir / label
    for path in (Result2Xlsx(args.template).export(days, year, out_dir),
                 DetailsCsv().export(days, year, day1.p, out_dir),
                 SummaryMd(label, summary).export(days, year, day1.p,
                                                  out_dir)):
        print(f"已导出 {path}")


if __name__ == "__main__":
    main()
