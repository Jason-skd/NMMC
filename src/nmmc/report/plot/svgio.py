"""把 SVG / Graphviz DOT 落成论文 PNG+PDF。

流程图走 Graphviz（dot），母线 / 对照表 / 时间轴走手写 SVG；栅格与矢量
都经 librsvg，避免 Matplotlib 圆角卡片。需要本机 `dot` 与 `rsvg-convert`
（macOS: brew install graphviz librsvg）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from nmmc.report.plot.style import SURFACE

_HOMEBREW = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))


def find_bin(name: str) -> Path:
    """定位系统二进制：先 Homebrew 路径，再 PATH。"""
    for folder in _HOMEBREW:
        cand = folder / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    which = shutil.which(name)
    if which:
        return Path(which)
    raise FileNotFoundError(
        f"未找到 {name}。示意图需要 Graphviz 与 librsvg："
        "brew install graphviz librsvg"
    )


def render_svg(svg: str, out_dir: Path, stem: str, *,
               width_px: int = 3300) -> tuple[Path, list[str]]:
    """SVG 字符串 → PNG（约 300 dpi 宽边）+ PDF。碰撞检查不适用，返回空问题表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    rsvg = find_bin("rsvg-convert")
    fd, tmp_name = tempfile.mkstemp(suffix=".svg")
    tmp = Path(tmp_name)
    try:
        os.write(fd, svg.encode("utf-8"))
        os.close(fd)
        subprocess.run(
            [str(rsvg), "-f", "png", "-w", str(width_px), "-a",
             "-b", SURFACE, "-o", str(png), str(tmp)],
            check=True, capture_output=True)
        subprocess.run(
            [str(rsvg), "-f", "pdf", "-a", "-b", SURFACE,
             "-o", str(pdf), str(tmp)],
            check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "replace")
        raise RuntimeError(f"rsvg-convert 失败（{stem}）: {err}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return png, []


def render_dot(dot: str, out_dir: Path, stem: str, *,
               width_px: int = 3300) -> tuple[Path, list[str]]:
    """DOT → SVG → PNG/PDF。"""
    dotbin = find_bin("dot")
    fd, tmp_name = tempfile.mkstemp(suffix=".dot")
    tmp_dot = Path(tmp_name)
    tmp_svg = tmp_dot.with_suffix(".svg")
    try:
        os.write(fd, dot.encode("utf-8"))
        os.close(fd)
        subprocess.run(
            [str(dotbin), "-Tsvg", "-o", str(tmp_svg), str(tmp_dot)],
            check=True, capture_output=True)
        svg = tmp_svg.read_text(encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "replace")
        raise RuntimeError(f"dot 失败（{stem}）: {err}") from exc
    finally:
        tmp_dot.unlink(missing_ok=True)
        tmp_svg.unlink(missing_ok=True)
    return render_svg(svg, out_dir, stem, width_px=width_px)
