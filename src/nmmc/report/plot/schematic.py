"""论文框架示意图：母线单线图、三问对照表、题 2 四层、题 3 分块时间轴。

流程图用 Graphviz；母线 / 对照表 / 甘特用 SVG。不再用 Matplotlib 圆角卡片。
图面不放大标题；守恒式、层名、价签是图的内容。
"""
from __future__ import annotations

from pathlib import Path

from nmmc.report.plot.style import (C_CH, C_DIS, C_LOAD, C_PRICE, C_PV, C_SOC,
                                    INK, INK2, MUTED, SURFACE)
from nmmc.report.plot.svgio import render_dot, render_svg

FONT = "PingFang SC, Hiragino Sans GB, sans-serif"
HAIR = "#d0cfc6"
BUS = "#5a8f5c"

_FILL = {
    "grid": "#e7eef7",
    "pv": "#eeeaf8",
    "bus": "#d5ead8",
    "load": "#f8e8e7",
    "curt": "#f3f1ec",
    "inv": "#fff4de",
    "bat": "#dff0de",
    "plan": "#d9f0e4",
    "exec": "#fde6d4",
    "settle": "#eadff3",
    "info": "#e7eef7",
    "q1": "#e7f2ea",
    "q2": "#e4eef8",
    "q3": "#f8efe2",
}


def _xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _mi(name: str, sub: str | None = None) -> str:
    """斜体变量 + 下标（dy 复位，避免后续字符掉下去）。"""
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
    return f'''<marker id="{mid}" markerWidth="6.2" markerHeight="6.2"
      refX="5.4" refY="3.1" orient="auto">
      <path d="M0.4,0.4 L5.8,3.1 L0.4,5.8 z" fill="{color}"/>
    </marker>'''


def _markers() -> str:
    pairs = {
        "arr-price": C_PRICE, "arr-pv": C_PV, "arr-load": C_LOAD,
        "arr-curt": MUTED, "arr-ch": C_CH, "arr-dis": C_DIS, "arr-soc": C_SOC,
        "arr-ink": INK2,
    }
    return "\n".join(_marker(k, v) for k, v in pairs.items())


def _box(x, y, w, h, fill, accent, title, body, *, title_fs=4.0, body_fs=2.85):
    """浅底卡片 + 左侧色条（不是 PPT 大圆角）。"""
    rail = 2.0
    body_svg = ""
    if body:
        body_svg = (
            f'<text x="{x + 6.2}" y="{y + h - 6.4}" font-family="{FONT}" '
            f'font-size="{body_fs}" fill="{INK2}">{body}</text>'
        )
    return f'''
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2.0"
          fill="{fill}" stroke="{HAIR}" stroke-width="0.42"/>
    <rect x="{x}" y="{y + 1.6}" width="{rail}" height="{h - 3.2}" rx="0.7"
          fill="{accent}"/>
    <text x="{x + 6.2}" y="{y + 9.2}" font-family="{FONT}" font-size="{title_fs}"
          font-weight="500" fill="{INK}">{title}</text>
    {body_svg}
    '''


def _txt(x, y, content, *, fs=3.2, fill=INK, anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{fs}" font-weight="{weight}" fill="{fill}">{content}</text>'
    )


# ---------------------------------------------------------------------------
# 图 1  电气单线：四回出线落到母线，储能下挂
# ---------------------------------------------------------------------------

def fig01_bus(out_dir: Path) -> tuple[Path, list[str]]:
    """微网母线与电量流转（单线图，不是卡片墙）。"""
    w, h = 280, 152
    bw, bh = 58, 24
    xs = (8.0, 76.5, 145.0, 213.5)
    y_box = 34.0
    cx = [x + bw / 2 for x in xs]
    y_bus = 80.0
    bus_x0, bus_x1 = 16.0, 258.0

    eq1 = (
        f'{_mi("g","t")} + {_mi("r","t")} + {_mi("d","t")}'
        f' = {_mi("ℓ","t")} + {_mi("c","t")} + {_mi("w","t")}'
    )
    eq2 = (
        f'{_mi("S","t")} = {_mi("S","t−1")} + η{_mi("c","t")} − {_mi("d","t")}/η'
    )

    devices = [
        (_FILL["grid"], C_PRICE, "外网", f'购电 {_mi("g","t")}，无功率上限'),
        (_FILL["pv"], C_PV, "光伏", f'注入 {_mi("r","t")}，不可售回'),
        (_FILL["load"], C_LOAD, "小区负载", f'需电 {_mi("ℓ","t")}，硬约束'),
        (_FILL["curt"], MUTED, "弃电", f'{_mi("w","t")} 免费松弛'),
    ]
    boxes = []
    for x, (fc, ac, title, body) in zip(xs, devices):
        boxes.append(_box(x, y_box, bw, bh, fc, ac, title, body))

    # 母线导体 + 四回 T 接（名称放在右端，不压导体）
    bus = f'''
    <rect x="{bus_x0}" y="{y_bus}" width="{bus_x1 - bus_x0}" height="7.0" rx="1.1"
          fill="{_FILL["bus"]}" stroke="#8aaa8a" stroke-width="0.45"/>
    <line x1="{bus_x0 + 3}" y1="{y_bus + 3.5}" x2="{bus_x1 - 3}" y2="{y_bus + 3.5}"
          stroke="{BUS}" stroke-width="1.7" stroke-linecap="round"/>
    {_txt(bus_x1 + 4.5, y_bus + 5.4, "母线", fs=3.4, fill=INK2, weight="500")}
    '''
    taps = []
    for x in cx:
        taps.append(
            f'<circle cx="{x}" cy="{y_bus + 3.5}" r="1.55" fill="{BUS}"/>'
        )

    # 四回出线：源向下注入，荷向上取走
    feeders = [
        (cx[0], "arr-price", C_PRICE, f'{_mi("g","t")}', True, -9.0),
        (cx[1], "arr-pv", C_PV, f'{_mi("r","t")}', True, -9.0),
        (cx[2], "arr-load", C_LOAD, f'{_mi("ℓ","t")}', False, 9.0),
        (cx[3], "arr-curt", MUTED, f'{_mi("w","t")}', False, 9.0),
    ]
    lines = []
    y_box_b = y_box + bh
    y_tap = y_bus - 0.4
    for x, mid, color, lab, inject, dx in feeders:
        if inject:
            y1, y2 = y_box_b + 1.2, y_tap - 2.4
        else:
            y1, y2 = y_tap - 2.4, y_box_b + 1.2
        lines.append(
            f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" '
            f'stroke-width="0.75" marker-end="url(#{mid})"/>'
        )
        anchor = "end" if dx < 0 else "start"
        lines.append(_txt(x + dx, (y_box_b + y_tap) / 2 + 1.1, lab,
                          fs=3.4, fill=color, anchor=anchor))

    # 储能下挂在母线中点
    mid = 140.0
    inv_w, inv_h, bat_h = 58.0, 18.5, 18.5
    inv_x = mid - inv_w / 2
    inv_y, bat_y = 98.0, 125.0
    storage = []
    storage.append(_box(inv_x, inv_y, inv_w, inv_h, _FILL["inv"], C_CH,
                        "逆变器  η = 0.9", "效率损失只发生在这里",
                        title_fs=3.5, body_fs=2.6))
    storage.append(_box(inv_x, bat_y, inv_w, bat_h, _FILL["bat"], C_SOC,
                        f'电池  {_mi("S","t")}',
                        "1200 ≤ S ≤ 10800 kWh",
                        title_fs=3.5, body_fs=2.6))
    storage.append(
        f'<circle cx="{mid}" cy="{y_bus + 3.5}" r="1.55" fill="{BUS}"/>'
    )
    # 母线 ↔ 逆变器：c 下、d 上
    x_c, x_d = mid - 6.5, mid + 6.5
    y_bus_b = y_bus + 7.0
    storage.append(
        f'<line x1="{x_c}" y1="{y_bus_b + 1.0}" x2="{x_c}" y2="{inv_y - 2.3}" '
        f'stroke="{C_CH}" stroke-width="0.75" marker-end="url(#arr-ch)"/>'
    )
    storage.append(
        f'<line x1="{x_d}" y1="{inv_y - 1.0}" x2="{x_d}" y2="{y_bus_b + 2.5}" '
        f'stroke="{C_DIS}" stroke-width="0.75" marker-end="url(#arr-dis)"/>'
    )
    storage.append(_txt(x_c - 8.5, (y_bus_b + inv_y) / 2 + 1.0,
                        f'{_mi("c","t")}', fs=3.1, fill=C_CH, anchor="end"))
    storage.append(_txt(x_d + 8.5, (y_bus_b + inv_y) / 2 + 1.0,
                        f'{_mi("d","t")}', fs=3.1, fill=C_DIS, anchor="start"))
    # 逆变器 ↔ 电池
    y_inv_b = inv_y + inv_h
    storage.append(
        f'<line x1="{x_c}" y1="{y_inv_b + 1.0}" x2="{x_c}" y2="{bat_y - 2.3}" '
        f'stroke="{C_SOC}" stroke-width="0.75" marker-end="url(#arr-soc)"/>'
    )
    storage.append(
        f'<line x1="{x_d}" y1="{bat_y - 1.0}" x2="{x_d}" y2="{y_inv_b + 2.5}" '
        f'stroke="{C_SOC}" stroke-width="0.75" marker-end="url(#arr-soc)"/>'
    )
    storage.append(_txt(x_c - 8.5, (y_inv_b + bat_y) / 2 + 1.0,
                        f'η{_mi("c","t")}', fs=2.9, fill=C_SOC, anchor="end"))
    storage.append(_txt(x_d + 8.5, (y_inv_b + bat_y) / 2 + 1.0,
                        f'{_mi("d","t")}/η', fs=2.9, fill=C_SOC, anchor="start"))

    body = "\n".join([
        _txt(w / 2, 13.0, eq1, fs=4.05, fill=INK, anchor="middle"),
        _txt(w - 10, 13.0, "母线瞬时守恒", fs=3.2, fill=MUTED, anchor="end"),
        _txt(w / 2, 23.0, eq2, fs=4.05, fill=INK, anchor="middle"),
        _txt(w - 10, 23.0, "电池记账，η = 0.9", fs=3.2, fill=MUTED, anchor="end"),
        *boxes, bus, *taps, *lines, *storage,
    ])
    return render_svg(_svg(w, h, body), out_dir, "fig01_微网母线与电量流转")


# ---------------------------------------------------------------------------
# 图 2  三问 × 三维：表，不是假流程图
# ---------------------------------------------------------------------------

def fig02_framework(out_dir: Path) -> tuple[Path, list[str]]:
    """信息 / 决策 / 罚则对照表。"""
    w, h = 280, 152
    heads = ("信息结构", "决策节奏", "结算罚则")
    stub_w = 28.0
    col_x = (40.0, 120.0, 200.0)
    col_w = 76.0
    header_y = 8.0
    rows = [
        ("题 1", _FILL["q1"], (
            ("附件 1 典型日", "电价 / 负载 / 光伏已知", "无随机"),
            (f'0:00 一次 LP',
             f'{_mi("S","144")} = {_mi("S","0")} 昼夜循环',
             f'{_mi("S","0")} 自由'),
            (f'min Σ {_mi("p")}&#x2009;{_mi("g")}', "无紧急、无偏差", "弃电免费"),
        )),
        ("题 2", _FILL["q2"], (
            ("读法 C：仅历史 + 日历", "附件 2 按日解锁", "当日真值 0:00 不可见"),
            ("0:00 两阶段 SAA 锁 g", "日内在线随机 MPC", "跨日真实 SOC 链"),
            (f'{_mi("Z")} = Σ {_mi("p")}&#x2009;{_mi("g")} + 5Σ {_mi("p")}&#x2009;{_mi("u")}',
             f'{_mi("u")} 只补计划缺口',
             f'末端估值 {_mi("ν","end")}'),
        )),
        ("题 3", _FILL["q3"], (
            ("+ 附件 3 一日四次", "官方光伏 × 持续法融合", "负载日内比例缩放"),
            ("0:00 仍锁全天 g", "6 / 12 / 18 各锁一块 a", "内环 MPC 无权改单"),
            (f'{_mi("Z")} = Σ[{_mi("p")}&#x2009;{_mi("a")} + ½{_mi("p")}|{_mi("a")}−{_mi("g")}|]',
             f'+ 5Σ {_mi("p")}&#x2009;{_mi("u")}',
             f'{_mi("g")} 降为计费基准'),
        )),
    ]
    row_h = 40.0
    row_y = (22.0, 66.0, 110.0)

    parts = []
    for i, name in enumerate(heads):
        parts.append(_txt(col_x[i] + col_w / 2, header_y + 8.5, name,
                          fs=3.8, fill=INK, anchor="middle", weight="500"))

    for (name, fc, cells), y in zip(rows, row_y):
        parts.append(
            f'<rect x="6" y="{y}" width="{w - 12}" height="{row_h}" rx="2.2" '
            f'fill="{fc}" stroke="{HAIR}" stroke-width="0.4"/>'
        )
        parts.append(_txt(6 + stub_w / 2, y + row_h / 2 + 1.4, name,
                          fs=5.2, fill=INK, anchor="middle", weight="500"))
        for x, lines in zip(col_x, cells):
            leading = 9.0
            y0 = y + 12.2
            for k, line in enumerate(lines):
                parts.append(_txt(x + 3.5, y0 + k * leading, line,
                                  fs=3.15, fill=INK2))
        # 列间细分隔，读成表而不是三张卡
        for x in col_x:
            parts.append(
                f'<line x1="{x}" y1="{y + 6}" x2="{x}" y2="{y + row_h - 6}" '
                f'stroke="{HAIR}" stroke-width="0.35"/>'
            )

    return render_svg(_svg(w, h, "\n".join(parts)), out_dir,
                      "fig02_三问递进-信息决策罚则")


# ---------------------------------------------------------------------------
# 图 9  Graphviz：四层流水线，执行层内嵌随机 MPC
# ---------------------------------------------------------------------------

def fig09_q2_arch(out_dir: Path) -> tuple[Path, list[str]]:
    """题 2 四层架构：预报 → 规划 → 执行（MPC）→ 结算。"""
    dot = r'''
digraph Q2 {
  graph [
    rankdir=LR
    bgcolor="#fcfcfb"
    fontname="PingFang SC"
    pad="0.28"
    nodesep="0.48"
    ranksep="0.72"
    splines=ortho
    compound=true
  ];
  node [
    shape=box
    style="rounded,filled"
    fontname="PingFang SC"
    fontsize=11
    penwidth=0.85
    color="#c8c6bd"
    margin="0.18,0.14"
  ];
  edge [
    color="#6a6964"
    arrowsize=0.48
    penwidth=1.05
    arrowhead=vee
  ];

  fcst [fillcolor="#e7eef7", label=<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">
      <TR><TD><FONT POINT-SIZE="13"><B>预报层</B></FONT></TD></TR>
      <TR><TD><FONT POINT-SIZE="9" COLOR="#52514e">L1 × P1 × R2</FONT></TD></TR>
    </TABLE>>];
  plan [fillcolor="#d9f0e4", label=<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">
      <TR><TD><FONT POINT-SIZE="13"><B>规划层</B></FONT></TD></TR>
      <TR><TD><FONT POINT-SIZE="9" COLOR="#52514e">两阶段 SAA，锁全天 g</FONT></TD></TR>
    </TABLE>>];
  settle [fillcolor="#eadff3", label=<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">
      <TR><TD><FONT POINT-SIZE="13"><B>结算层</B></FONT></TD></TR>
      <TR><TD><FONT POINT-SIZE="9" COLOR="#52514e">Σ p g + 5 Σ p u</FONT></TD></TR>
    </TABLE>>];

  subgraph cluster_exec {
    label=<<FONT POINT-SIZE="12"><B>执行层 · 在线随机 MPC</B></FONT>>
    fontname="PingFang SC"
    style="rounded,filled"
    color="#e0c4a8"
    fillcolor="#fff7f0"
    margin=20
    now [fillcolor="#fde6d4", label=<
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">
        <TR><TD><FONT POINT-SIZE="12"><B>当步 t</B></FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9" COLOR="#52514e">真值 n · 水位 S</FONT></TD></TR>
      </TABLE>>];
    fut [fillcolor="#e4eef8", label=<
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1">
        <TR><TD><FONT POINT-SIZE="12"><B>未来时域</B></FONT></TD></TR>
        <TR><TD><FONT POINT-SIZE="9" COLOR="#52514e">12 个平行情景</FONT></TD></TR>
      </TABLE>>];
    now -> fut;
  }

  fcst -> plan;
  plan -> now [lhead=cluster_exec];
  fut -> settle [ltail=cluster_exec];
}
'''
    return render_dot(dot, out_dir, "fig09_题2四层架构与随机MPC", width_px=3300)


# ---------------------------------------------------------------------------
# 图 14  一日甘特 + 四档价签
# ---------------------------------------------------------------------------

def fig14_q3_commit(out_dir: Path) -> tuple[Path, list[str]]:
    """分块承诺时间轴与四档价签。"""
    w = 280
    x0, x1 = 16.0, 264.0
    span = x1 - x0
    y_bar, h_bar = 18.0, 16.5
    segs = [
        (_FILL["curt"], MUTED, f'{_mi("B","0")}  0–6 时',
         "0:00 锁死，此后 a ≡ g"),
        (_FILL["plan"], C_SOC, f'{_mi("B","1")}  6–12 时',
         "6:00 锁定 a · 光伏预报"),
        (_FILL["q2"], C_PRICE, f'{_mi("B","2")}  12–18 时',
         "12:00 锁定 a · 短前置"),
        (_FILL["exec"], C_CH, f'{_mi("B","3")}  18–24 时',
         "18:00 锁定 a · 负载观测"),
    ]
    parts = []
    n = len(segs)
    seg_w = span / n
    for i, (fc, accent, title, note) in enumerate(segs):
        x = x0 + i * seg_w
        # 首尾随外框圆角，中间直角，读成一条日带
        if i == 0:
            clip = f"url(#barL)"
        elif i == n - 1:
            clip = f"url(#barR)"
        else:
            clip = ""
        clip_attr = f' clip-path="{clip}"' if clip else ""
        parts.append(
            f'<rect x="{x}" y="{y_bar}" width="{seg_w}" height="{h_bar}" '
            f'fill="{fc}" stroke="none"{clip_attr}/>'
        )
        parts.append(
            f'<rect x="{x}" y="{y_bar}" width="{seg_w}" height="2.0" '
            f'fill="{accent}"{clip_attr}/>'
        )
        parts.append(_txt(x + seg_w / 2, y_bar + 11.2, title,
                          fs=3.6, fill=INK, anchor="middle", weight="500"))
        parts.append(_txt(x + seg_w / 2, y_bar + h_bar + 8.8, note,
                          fs=2.85, fill=INK2, anchor="middle"))
        if i:
            parts.append(
                f'<line x1="{x}" y1="{y_bar}" x2="{x}" y2="{y_bar + h_bar}" '
                f'stroke="{HAIR}" stroke-width="0.4"/>'
            )
    parts.append(
        f'<rect x="{x0}" y="{y_bar}" width="{span}" height="{h_bar}" rx="2.0" '
        f'fill="none" stroke="{HAIR}" stroke-width="0.5"/>'
    )

    # 时间轴
    y_ax = y_bar + h_bar + 18.0
    parts.append(
        f'<line x1="{x0}" y1="{y_ax}" x2="{x1}" y2="{y_ax}" '
        f'stroke="{HAIR}" stroke-width="0.7"/>'
    )
    for i, lab in enumerate(("0:00", "6:00", "12:00", "18:00", "24:00")):
        x = x0 + span * i / 4
        parts.append(
            f'<line x1="{x}" y1="{y_ax - 2.2}" x2="{x}" y2="{y_ax + 2.2}" '
            f'stroke="{INK2}" stroke-width="0.6"/>'
        )
        parts.append(_txt(x, y_ax + 8.0, lab, fs=3.0, fill=INK2, anchor="middle"))

    chips = [
        ("#d7e6f6", C_PRICE, "违约  0.5p", "计划多报、实际下调"),
        ("#cfe6cd", C_SOC, "计划  1.0p", "重合部分原价"),
        ("#f3d3a1", C_CH, "加订  1.5p", "计划少报、实际上调"),
        ("#e8b4b2", C_LOAD, "紧急  5.0p", "末次修订后的残差"),
    ]
    y_chip = 78.0
    h_chip = 20.0
    for i, (fc, accent, title, note) in enumerate(chips):
        gap = 1.6
        x = x0 + i * seg_w + gap / 2
        cw = seg_w - gap
        parts.append(
            f'<rect x="{x}" y="{y_chip}" width="{cw}" height="{h_chip}" '
            f'rx="1.6" fill="{fc}" stroke="{HAIR}" stroke-width="0.4"/>'
        )
        parts.append(
            f'<rect x="{x}" y="{y_chip + 1.4}" width="2.0" height="{h_chip - 2.8}" '
            f'rx="0.7" fill="{accent}"/>'
        )
        parts.append(_txt(x + 6.5, y_chip + 9.0, title,
                          fs=3.5, fill=INK, weight="500"))
        parts.append(_txt(x + 6.5, y_chip + 16.8, note, fs=2.7, fill=INK2))

    formula = (
        f'{_mi("Z","t")}'
        f'<tspan font-size="60%" dy="-1.55">buy</tspan>'
        f'<tspan dy="1.55">&#x200b;</tspan>'
        f' = {_mi("p","t")}&#x2009;{_mi("a","t")} + ½&#x2009;{_mi("p","t")}|{_mi("a","t")} − {_mi("g","t")}|'
    )
    parts.append(_txt(w / 2, 116.0, formula, fs=4.15, fill=INK, anchor="middle"))

    extras = f'''
    <clipPath id="barL">
      <rect x="{x0}" y="{y_bar}" width="{seg_w + 2}" height="{h_bar}" rx="2.0"/>
    </clipPath>
    <clipPath id="barR">
      <rect x="{x0 + 3 * seg_w - 2}" y="{y_bar}" width="{seg_w + 2}" height="{h_bar}" rx="2.0"/>
    </clipPath>
    '''
    return render_svg(_svg(w, 128, "\n".join(parts), extras=extras), out_dir,
                      "fig14_题3分块承诺与四档价签")
