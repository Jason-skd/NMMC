"""论文图命令行：uv run paperfigs

默认写出 results/paper/ 下 PNG + PDF，并打印文字碰撞检查。
`--v2` 走白底栏宽集，默认写出 results/paper-v2/。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="C 题问题 1–4 论文图")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录")
    parser.add_argument("--v2", action="store_true",
                        help="白底 v2 集（results/paper-v2）")
    parser.add_argument("--only", nargs="+", default=None,
                        help="只画指定编号，如 q4 或 fig20 fig21")
    args = parser.parse_args(argv)
    if args.v2:
        from nmmc.report.plot.paper_v2 import render_all
        out = args.output_dir or Path("results/paper-v2")
        render_all(out, only=args.only)
    else:
        from nmmc.report.plot.paper import render_all
        out = args.output_dir or Path("results/paper")
        render_all(out)


if __name__ == "__main__":
    main()
