"""论文图命令行：uv run paperfigs

默认写出 results/paper/ 下 PNG + PDF，并打印文字碰撞检查。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="C 题问题 1–3 论文图")
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"),
                        help="输出目录")
    args = parser.parse_args(argv)
    from nmmc.report.plot.paper import render_all
    render_all(args.output_dir)


if __name__ == "__main__":
    main()
