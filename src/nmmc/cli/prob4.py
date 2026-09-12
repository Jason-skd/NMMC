"""问题4 命令行入口（双线：4-2 题2 骨架 / 4-3 题3 骨架，共用价格层）。

用法：
    uv run prob4 --line 4-2                       # 读法 R 主交付，V2 ν，全年
    uv run prob4 --line 4-3
    uv run prob4 --line 4-2 --days 45             # 快速冒烟（含 31 天暖机）
    uv run prob4 --line 4-2 --price-mode a1       # C4-2 复现：应逐日复现题2
    uv run prob4 --line 4-3 --price-mode a1       # C4-2 复现：应逐日复现题3
    uv run prob4 --line 4-2 --price-mode d        # 读法 D 诊断（Z^D 上界）
    uv run prob4 --line 4-2 --price PR0           # 电价建模价值消融（Z^PR0）
    uv run prob4 --line 4-2 --nu-rule V1 --gamma 0.6   # ν 标定对照臂
    uv run prob4 --line 4-3 --epochs 0,6,12       # 逐级增量（继承题3 §12.2）
"""
import argparse
from pathlib import Path

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.price import load_price
from nmmc.data.year import DAYS, load_year
from nmmc.model.forecast import ForecastConfig
from nmmc.model.forecast3 import Q3Config
from nmmc.model.forecast4 import PriceConfig
from nmmc.model.q2 import KAPPA
from nmmc.report.q4 import (TEMPLATE_42, TEMPLATE_43, DetailsCsv, Result4Xlsx,
                            SummaryMd, summarize)
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.run.q4 import rollout_q42, rollout_q43


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
    """构建 prob4 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="C 题问题4：波动电价双线（读法 R 诚实回测 + 价格预报层）")
    parser.add_argument("--line", choices=["4-2", "4-3"], default="4-2",
                        help="交付线：4-2 题2 骨架 / 4-3 题3 骨架")
    parser.add_argument("--input", type=Path, default=ATTACH / "附件2.xlsx",
                        help="附件2 数据路径")
    parser.add_argument("--attach1", type=Path, default=ATTACH / "附件1.xlsx",
                        help="附件1 数据路径（电价与典型日先验）")
    parser.add_argument("--attach4", type=Path, default=ATTACH / "附件4.xlsx",
                        help="附件4 电价数据路径")
    parser.add_argument("--forecast", type=Path, default=ATTACH / "附件3.xlsx",
                        help="附件3 光伏预报路径（仅 4-3 线）")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/q4"), help="结果输出根目录")
    parser.add_argument("--price", choices=["PR1", "PR0"], default="PR1",
                        help="电价点预报候选：PR1 近 4 同星期几（默认）/"
                             "PR0 附件1 典型日（零信息基准）")
    parser.add_argument("--wp", type=int, default=4,
                        help="PR1 同星期几窗长 W_p")
    parser.add_argument("--price-mode", choices=["r", "d", "a1"], default="r",
                        help="价格口径：r 读法 R（主交付）/ d 读法 D（诊断）/"
                             "a1 附件1 退化（C4-2 复现，ν 强制 0.472）")
    parser.add_argument("--nu-rule", choices=["V0", "V1", "V2"], default="V2",
                        help="ν^end 标定：V2 次日预报价循环对偶（默认，零"
                             "超参数）/ V1 γ·次日均价 / V0 常数 0.472")
    parser.add_argument("--gamma", type=float, default=0.60,
                        help="V1 的 γ（默认 0.60 取 §3 第 18 条实测均值）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--days", type=int, default=DAYS,
                        help="滚动天数（含暖机期）")
    parser.add_argument("--epochs", type=parse_epochs, default=(0, 1, 2, 3),
                        help="行使修订权的时刻（小时，含 0 子集），仅 4-3 线")
    return parser


def main(argv: list[str] | None = None) -> None:
    """prob4 主流程：加载 → 双线全年滚动 → 打印诊断 → 导出 result4 与摘要。"""
    args = build_parser().parse_args(argv)
    day1 = load_day1(args.attach1)
    year = load_year(args.input)
    price = load_price(day1.p, args.attach4)
    cfgp = PriceConfig(price=args.price, wp=args.wp, nu_rule=args.nu_rule,
                       gamma=args.gamma)
    # a1 退化口径强制 V0（C4-2 复现），标签如实反映生效规则
    nu_label = "V0" if args.price_mode == "a1" else cfgp.nu_rule
    label = f"{args.price}-{nu_label}-{args.price_mode}"
    if args.line == "4-3" and tuple(args.epochs) != (0, 1, 2, 3):
        label += "-E" + "-".join(str(6 * m) for m in args.epochs)
    nu_hint = (f"ν = 0.472（a1 强制）" if args.price_mode == "a1"
               else f"ν 规则 = {cfgp.nu_rule}"
               + (f"（γ = {cfgp.gamma:g}）" if cfgp.nu_rule == "V1" else ""))
    print(f"问题 {args.line}，配置 {label}，价格口径 = {args.price_mode}，"
          f"{nu_hint}，κ = {KAPPA:g}，滚动 {args.days} 天"
          f"（暖机 {WARMUP_DAYS} 天）")

    if args.line == "4-2":
        cfg = ForecastConfig(seed=args.seed)
        days = rollout_q42(year, price, day1.ell, day1.r, day1.p, cfg, cfgp,
                           mode=args.price_mode, days=args.days)
    else:
        from nmmc.data.fcst import load_forecast
        fc = load_forecast(args.forecast)
        cfg = Q3Config(load="L1", seed=args.seed)
        days = rollout_q43(year, fc, price, day1.ell, day1.r, day1.p, cfg,
                           cfgp, mode=args.price_mode, days=args.days,
                           epochs=args.epochs)

    summary = summarize(days, year, price, args.price_mode, day1.p)
    print("== 费用分层（报告期，实现价结算）==")
    for key in ("Z_online", "Z_ce", "执行层增益", "Z_expost", "Z_pi", "EVPI",
                "Z_adj 年末修正"):
        print(f"  {key}: {summary[key]:,.2f} 元")
    print("== 价格诊断 ==")
    print(f"  价格预报 MAE/MAPE: {summary['价格预报 MAE']:.4f} 元/kWh / "
          f"{summary['价格预报 MAPE'] * 100:.2f}%")
    print(f"  ν^end 均值/范围: {summary['ν 均值']:.4f} / "
          f"[{summary['ν 范围'][0]:.4f}, {summary['ν 范围'][1]:.4f}] 元/kWh")
    print(f"  星期台阶（日末 SOC）: 次日周五 {summary['星期台阶 次日周五']:,.0f} / "
          f"次日周六 {summary['星期台阶 次日周六']:,.0f} / "
          f"次日周一 {summary['星期台阶 次日周一']:,.0f} / "
          f"其他 {summary['星期台阶 其他']:,.0f} kWh")
    print("== 自检 ==")
    for key in ("C4-4 结算价残差 max", "C4-14 前缀偏差 max", "C6 对账残差 max",
                "C7 能量守恒残差 max", "C4-14 max(c·d)", "C4-14 max(u·w)",
                "C4-14 u∧c 时段数", "C9 序关系违例天数",
                "C4-8 ν 区间违例天数"):
        print(f"  {key}: {summary[key]}")
    print("== 紧急购电 ==")
    print(f"  天数 {summary['紧急购电天数']}/{summary['天数']}，"
          f"时段数 {summary['紧急购电时段数']}"
          f"（频率 {summary['缺口时段频率'] * 100:.2f}%），"
          f"电量 {summary['紧急购电量合计']:,.1f} kWh，"
          f"费用 {summary['紧急购电费合计']:,.2f} 元"
          f"（占比 {summary['紧急费占比'] * 100:.2f}%）")
    print(f"  电价分段（实现价）：谷 {summary['紧急购电谷段占比'] * 100:.2f}% / "
          f"平 {summary['紧急购电平段占比'] * 100:.2f}% / "
          f"峰 {summary['紧急购电峰段占比'] * 100:.2f}%，"
          f"光伏盈余时段 {summary['紧急购电盈余时段占比'] * 100:.2f}%")
    print(f"  归因：① {summary['归因①功率强制占比'] * 100:.2f}% / "
          f"② {summary['归因②储能耗尽占比'] * 100:.2f}% / "
          f"③ {summary['归因③有电不放占比'] * 100:.2f}% / "
          f"④ {summary['归因④低水位带占比'] * 100:.2f}%")
    print("== C4-19 归因健康度 ==")
    for key in ("C4-19 ③为少数", "C4-19 峰段高于谷段", "C4-19 盈余近零"):
        print(f"  {key}: {'过' if summary[key] else '不过'}")

    adjusted = args.line == "4-3"
    template = TEMPLATE_43 if adjusted else TEMPLATE_42
    filename = f"result{args.line}.xlsx"
    prefix = "q4" + args.line[1:]                  # q4-2 / q4-3
    out_dir = args.output_dir / args.line / label
    exports = (
        Result4Xlsx(template, filename, adjusted).export(
            days, year, price, out_dir, mode=args.price_mode, typ_p=day1.p),
        DetailsCsv().export(days, year, price, out_dir, prefix=prefix,
                            mode=args.price_mode, typ_p=day1.p),
        SummaryMd(label, summary, args.price_mode, args.line).export(
            days, year, price, out_dir, prefix=prefix),
    )
    for path in exports:
        print(f"已导出 {path}")


if __name__ == "__main__":
    main()
