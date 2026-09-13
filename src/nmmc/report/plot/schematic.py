"""论文框架示意图：IEEE / OR 线框，不是圆角色块。

母线用单线；三问用三线表；架构用直角嵌套框；分块用时间轴。
填色只出现在潮流箭头和图例色块上。
"""
from __future__ import annotations

from pathlib import Path

from nmmc.report.plot.style import (C_CH, C_DIS, C_LOAD, C_PRICE, C_PV, C_SOC,
                                    INK, INK2, MUTED, SURFACE)
from nmmc.report.plot.svgio import render_svg

FONT = "PingFang SC, Hiragino Sans GB, sans-serif"
STROKE = "#1a1a1a"
SW = 0.38          # ≈ 1.1 pt
SW_BUS = 2.4
RULE = 0.45
RULE_MID = 0.22


def _xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _mi(name: str, sub: str | None = None) -> str:
    s = f'<tspan font-style="italic">{_xml(name)}</tspan>'
    if sub:
        s += (f'<tspan font-size="65%" dy="1.25">{_xml(sub)}</tspan>'
              f'<tspan dy="-1.25">&#x200b;</tspan>')
    return s


def _svg(w: float, h: float, body: str, extras: str = "") -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm"
     viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{SURFACE}"/>
  <defs>
    {_markers()}
    {extras}
  </defs>
  {body}
</svg>
'''


def _marker(mid: str, color: str) -> str:
    return f'''<marker id="{mid}" markerWidth="5.6" markerHeight="5.6"
      refX="5.0" refY="2.8" orient="auto">
      <path d="M0.2,0.3 L5.2,2.8 L0.2,5.3 z" fill="{color}"/>
    </marker>'''


def _markers() -> str:
    pairs = {
        "arr-price": C_PRICE, "arr-pv": C_PV, "arr-load": C_LOAD,
        "arr-curt": MUTED, "arr-ch": C_CH, "arr-dis": C_DIS,
        "arr-soc": C_SOC, "arr-ink": STROKE,
    }
    return "\n".join(_marker(k, v) for k, v in pairs.items())


def _rect(x, y, w, h, *, fill=SURFACE, stroke=STROKE, sw=SW, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def _txt(x, y, content, *, fs=3.2, fill=INK, anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{fs}" font-weight="{weight}" fill="{fill}">{content}</text>'
    )


def _ibox(x, y, w, h, title, sub="", *, fs=3.6, fs_sub=2.65):
    """直角白框，字贴着框，不是留白很大的卡片。"""
    parts = [_rect(x, y, w, h)]
    cx = x + w / 2
    if sub:
        parts.append(_txt(cx, y + h * 0.42, title, fs=fs, anchor="middle",
                          weight="500"))
        parts.append(_txt(cx, y + h * 0.72, sub, fs=fs_sub, anchor="middle",
                          fill=INK2))
    else:
        parts.append(_txt(cx, y + h * 0.58, title, fs=fs, anchor="middle",
                          weight="500"))
    return "\n".join(parts)


def _hline(x1, x2, y, *, sw=SW, color=STROKE):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
            f'stroke="{color}" stroke-width="{sw}"/>')


def _vline(x, y1, y2, *, sw=SW, color=STROKE, marker=""):
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}"{m}/>')


def _arrow(x1, y1, x2, y2, *, color=STROKE, marker="arr-ink", sw=0.42):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#{marker})"/>'
    )


# ---------------------------------------------------------------------------
# 图 1  单线图
# ---------------------------------------------------------------------------

def fig01_bus(out_dir: Path) -> tuple[Path, list[str]]:
    """微网母线：单线 + 直角端子，不是色块墙。"""
    w, h = 280, 138
    bw, bh = 46.0, 14.0
    xs = (18.0, 84.0, 150.0, 216.0)
    y_box = 32.0
    cx = [x + bw / 2 for x in xs]
    y_bus = 74.0

    eq1 = (
        f'{_mi("g","t")} + {_mi("r","t")} + {_mi("d","t")}'
        f' = {_mi("ℓ","t")} + {_mi("c","t")} + {_mi("w","t")}'
    )
    eq2 = (
        f'{_mi("S","t")} = {_mi("S","t−1")} + η{_mi("c","t")} − {_mi("d","t")}/η'
        f'<tspan fill="{INK2}">，η = 0.9</tspan>'
    )

    names = ("外网", "光伏", "小区负载", "弃电")
    parts = [
        _txt(w / 2, 12.5, eq1, fs=4.0, fill=INK, anchor="middle"),
        _txt(w / 2, 22.0, eq2, fs=4.0, fill=INK, anchor="middle"),
    ]
    for x, name in zip(xs, names):
        parts.append(_ibox(x, y_box, bw, bh, name, fs=3.5))

    bus_x0, bus_x1 = 12.0, 262.0
    parts.append(_hline(bus_x0, bus_x1, y_bus, sw=SW_BUS, color=STROKE))
    parts.append(_txt(bus_x1 + 4.0, y_bus + 1.2, "母线", fs=3.3, fill=INK))

    feeders = [
        (cx[0], "arr-price", C_PRICE, f'{_mi("g","t")}', True),
        (cx[1], "arr-pv", C_PV, f'{_mi("r","t")}', True),
        (cx[2], "arr-load", C_LOAD, f'{_mi("ℓ","t")}', False),
        (cx[3], "arr-curt", MUTED, f'{_mi("w","t")}', False),
    ]
    y_box_b = y_box + bh
    for x, mid, color, lab, inject in feeders:
        if inject:
            y1, y2 = y_box_b + 1.0, y_bus - 3.2
        else:
            y1, y2 = y_bus - 3.2, y_box_b + 1.2
        parts.append(_arrow(x, y1, x, y2, color=color, marker=mid, sw=0.55))
        parts.append(
            f'<circle cx="{x}" cy="{y_bus}" r="1.45" fill="{SURFACE}" '
            f'stroke="{STROKE}" stroke-width="0.4"/>'
        )
        dx = -7.5 if inject else 7.5
        anchor = "end" if inject else "start"
        parts.append(_txt(x + dx, (y_box_b + y_bus) / 2 + 0.8, lab,
                          fs=3.2, fill=color, anchor=anchor))

    mid = 140.0
    inv_w, inv_h, bat_h = 50.0, 13.0, 13.0
    inv_x = mid - inv_w / 2
    inv_y, bat_y = 88.0, 114.0
    parts.append(
        f'<circle cx="{mid}" cy="{y_bus}" r="1.45" fill="{SURFACE}" '
        f'stroke="{STROKE}" stroke-width="0.4"/>'
    )
    parts.append(_ibox(inv_x, inv_y, inv_w, inv_h, "逆变器  η = 0.9", fs=3.2))
    parts.append(_ibox(inv_x, bat_y, inv_w, bat_h,
                       f'电池  1200–10800 kWh', fs=3.0))

    x_c, x_d = mid - 5.5, mid + 5.5
    parts.append(_arrow(x_c, y_bus + 2.4, x_c, inv_y - 2.2,
                        color=C_CH, marker="arr-ch", sw=0.5))
    parts.append(_arrow(x_d, inv_y - 1.0, x_d, y_bus + 3.0,
                        color=C_DIS, marker="arr-dis", sw=0.5))
    parts.append(_txt(x_c - 7.5, (y_bus + inv_y) / 2 + 1.0,
                      f'{_mi("c","t")}', fs=3.0, fill=C_CH, anchor="end"))
    parts.append(_txt(x_d + 7.5, (y_bus + inv_y) / 2 + 1.0,
                      f'{_mi("d","t")}', fs=3.0, fill=C_DIS, anchor="start"))

    y_inv_b = inv_y + inv_h
    parts.append(_arrow(x_c, y_inv_b + 1.0, x_c, bat_y - 2.2,
                        color=C_SOC, marker="arr-soc", sw=0.5))
    parts.append(_arrow(x_d, bat_y - 1.0, x_d, y_inv_b + 2.4,
                        color=C_SOC, marker="arr-soc", sw=0.5))
    parts.append(_txt(x_c - 7.5, (y_inv_b + bat_y) / 2 + 1.0,
                      f'η{_mi("c","t")}', fs=2.8, fill=C_SOC, anchor="end"))
    parts.append(_txt(x_d + 7.5, (y_inv_b + bat_y) / 2 + 1.0,
                      f'{_mi("d","t")}/η', fs=2.8, fill=C_SOC, anchor="start"))

    return render_svg(_svg(w, h, "\n".join(parts)), out_dir,
                      "fig01_微网母线与电量流转")


# ---------------------------------------------------------------------------
# 图 2  三线表
# ---------------------------------------------------------------------------

def fig02_framework(out_dir: Path) -> tuple[Path, list[str]]:
    """信息 / 决策 / 罚则：三线表。"""
    w, h = 280, 118
    x0, x1 = 8.0, 272.0
    stub_x = 22.0
    col_x = (48.0, 122.0, 198.0)
    heads = ("信息结构", "决策节奏", "结算罚则")
    rows = [
        ("题 1", (
            ("附件 1 典型日", "电价 / 负载 / 光伏已知", "无随机"),
            (f'0:00 一次 LP',
             f'{_mi("S","144")} = {_mi("S","0")} 昼夜循环',
             f'{_mi("S","0")} 自由'),
            (f'min Σ {_mi("p")}&#x2009;{_mi("g")}', "无紧急、无偏差", "弃电免费"),
        )),
        ("题 2", (
            ("读法 C：仅历史 + 日历", "附件 2 按日解锁", "当日真值 0:00 不可见"),
            ("0:00 两阶段 SAA 锁 g", "日内在线随机 MPC", "跨日真实 SOC 链"),
            (f'{_mi("Z")} = Σ {_mi("p")}&#x2009;{_mi("g")} + 5Σ {_mi("p")}&#x2009;{_mi("u")}',
             f'{_mi("u")} 只补计划缺口',
             "末端估值 ν"
             '<tspan font-size="65%" dy="1.25">end</tspan>'
             '<tspan dy="-1.25">&#x200b;</tspan>'),
        )),
        ("题 3", (
            ("+ 附件 3 一日四次", "官方光伏 × 持续法融合", "负载日内比例缩放"),
            ("0:00 仍锁全天 g", "6 / 12 / 18 各锁一块 a", "内环 MPC 无权改单"),
            (f'{_mi("Z")} = Σ[{_mi("p")}&#x2009;{_mi("a")} + ½{_mi("p")}|{_mi("a")}−{_mi("g")}|]',
             f'+ 5Σ {_mi("p")}&#x2009;{_mi("u")}',
             f'{_mi("g")} 降为计费基准'),
        )),
    ]

    y_top = 8.0
    y_head = 16.5
    row_h = 30.0
    y_rows = (22.0, 52.0, 82.0)
    y_bot = 112.0

    parts = [
        _hline(x0, x1, y_top, sw=RULE),
        _hline(x0, x1, y_head, sw=RULE_MID),
        _hline(x0, x1, y_bot, sw=RULE),
    ]
    for x, name in zip(col_x, heads):
        parts.append(_txt(x, y_top + 6.2, name, fs=3.5, fill=INK, weight="500"))
    for i, ((name, cells), y) in enumerate(zip(rows, y_rows)):
        if i:
            parts.append(_hline(x0, x1, y, sw=0.15, color="#bdbdb5"))
        parts.append(_txt(stub_x, y + 16.0, name, fs=3.8, fill=INK,
                          anchor="middle", weight="500"))
        for x, lines in zip(col_x, cells):
            for k, line in enumerate(lines):
                parts.append(_txt(x, y + 8.5 + k * 7.6, line, fs=3.05, fill=INK2))

    return render_svg(_svg(w, h, "\n".join(parts)), out_dir,
                      "fig02_三问递进-信息决策罚则")


# ---------------------------------------------------------------------------
# 图 9  嵌套直角框
# ---------------------------------------------------------------------------

def fig09_q2_arch(out_dir: Path) -> tuple[Path, list[str]]:
    """题 2 四层：直角线框，执行层为嵌套框。"""
    w, h = 280, 48
    y, bh = 9.0, 30.0
    bw = 44.0
    exec_w = 102.0
    gap = 9.5
    x0 = 6.0
    x_fcst = x0
    x_plan = x_fcst + bw + gap
    x_exec = x_plan + bw + gap
    x_set = x_exec + exec_w + gap

    inner_w, inner_h = 40.0, 22.0
    inner_y = y + 6.5
    x_now = x_exec + 8.0
    x_fut = x_exec + exec_w - 8.0 - inner_w

    parts = [
        _ibox(x_fcst, y, bw, bh, "预报层", "L1 × P1 × R2"),
        _ibox(x_plan, y, bw, bh, "规划层", "两阶段 SAA，锁 g"),
        _rect(x_exec, y - 4.0, exec_w, bh + 8.0),
        _txt(x_exec + exec_w / 2, y + 1.8, "执行层 · 在线随机 MPC",
             fs=3.0, fill=INK, anchor="middle", weight="500"),
        _ibox(x_now, inner_y, inner_w, inner_h, "当步 t", "真值 · 水位"),
        _ibox(x_fut, inner_y, inner_w, inner_h, "未来时域", "12 个平行情景"),
        _ibox(x_set, y, bw, bh, "结算层", "Σ p g + 5 Σ p u"),
    ]
    mid_y = y + bh / 2
    parts.append(_arrow(x_fcst + bw + 1.0, mid_y, x_plan - 2.6, mid_y))
    parts.append(_arrow(x_plan + bw + 1.0, mid_y, x_exec - 2.6, mid_y))
    parts.append(_arrow(x_now + inner_w + 1.0, inner_y + inner_h / 2,
                        x_fut - 2.6, inner_y + inner_h / 2))
    parts.append(_arrow(x_exec + exec_w + 1.0, mid_y, x_set - 2.6, mid_y))

    return render_svg(_svg(w, h, "\n".join(parts)), out_dir,
                      "fig09_题2四层架构与随机MPC")


# ---------------------------------------------------------------------------
# 图 14  时间轴 + 价签色块（色块只作图例）
# ---------------------------------------------------------------------------

def fig14_q3_commit(out_dir: Path) -> tuple[Path, list[str]]:
    """分块承诺：一条日带，价签用 3 mm 色块而不是卡片。"""
    w = 280
    x0, x1 = 16.0, 264.0
    span = x1 - x0
    y_bar, h_bar = 20.0, 6.0
    n = 4
    seg_w = span / n
    titles = (
        f'{_mi("B","0")}  0–6 时',
        f'{_mi("B","1")}  6–12 时',
        f'{_mi("B","2")}  12–18 时',
        f'{_mi("B","3")}  18–24 时',
    )
    notes = (
        "0:00 锁死，a ≡ g",
        "6:00 锁定 a · 光伏预报",
        "12:00 锁定 a · 短前置",
        "18:00 锁定 a · 负载观测",
    )
    parts = [_rect(x0, y_bar, span, h_bar)]
    for i in range(n):
        x = x0 + i * seg_w
        parts.append(_txt(x + seg_w / 2, y_bar - 4.0, titles[i],
                          fs=3.5, fill=INK, anchor="middle", weight="500"))
        parts.append(_txt(x + seg_w / 2, y_bar + h_bar + 16.5, notes[i],
                          fs=2.8, fill=INK2, anchor="middle"))
        if i:
            parts.append(_vline(x, y_bar, y_bar + h_bar, sw=0.35))

    y_ax = y_bar + h_bar
    for i, lab in enumerate(("0:00", "6:00", "12:00", "18:00", "24:00")):
        x = x0 + span * i / 4
        parts.append(_vline(x, y_ax, y_ax + 2.4, sw=0.4))
        parts.append(_txt(x, y_ax + 8.0, lab, fs=2.9, fill=INK2, anchor="middle"))

    chips = (
        (C_PRICE, "违约  0.5p", "计划多报、实际下调"),
        (C_SOC, "计划  1.0p", "重合部分原价"),
        (C_CH, "加订  1.5p", "计划少报、实际上调"),
        (C_LOAD, "紧急  5.0p", "末次修订后的残差"),
    )
    y_leg = 58.0
    for i, (color, title, note) in enumerate(chips):
        x = x0 + i * seg_w
        parts.append(
            f'<rect x="{x}" y="{y_leg}" width="3.2" height="3.2" fill="{color}" '
            f'stroke="{STROKE}" stroke-width="0.2"/>'
        )
        parts.append(_txt(x + 5.2, y_leg + 2.7, title, fs=3.3, fill=INK,
                          weight="500"))
        parts.append(_txt(x, y_leg + 10.0, note, fs=2.65, fill=INK2))

    formula = (
        f'{_mi("Z","t")}'
        f'<tspan font-size="60%" dy="-1.55">buy</tspan>'
        f'<tspan dy="1.55">&#x200b;</tspan>'
        f' = {_mi("p","t")}&#x2009;{_mi("a","t")} + ½&#x2009;{_mi("p","t")}'
        f'|{_mi("a","t")} − {_mi("g","t")}|'
    )
    parts.append(_txt(w / 2, 84.0, formula, fs=4.05, fill=INK, anchor="middle"))

    return render_svg(_svg(w, 94, "\n".join(parts)), out_dir,
                      "fig14_题3分块承诺与四档价签")
