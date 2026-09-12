"""题3 预报层：反方差融合（F1/PB0）、左锚线性降尺度（I1 定稿）、
负载日内比例缩放（LU0/LU1）、逐阶段×逐提前期分池残差场景（S-b/S-a/S-c）。

文档 §3–§4：附件3 误差随提前期急剧劣化、持续法误差与提前期无关（数据
核验第 4 条），故点预报按提前期反方差加权融合；残差池必须按提前期分桶，
否则混池会严重高估短提前期的不确定性。负载日前基线沿用题2 候选
（model.forecast，只用 [:j] 历史）；日内缩放只用当日已实现段（τ_m 时合法）。

题4 扩展：build_scenarios 接受可选电价输入时返回 (nets, prices) 二元组
——电价段不分提前期（题4 档 §4.2），S-b 下与该桶 L/R 段取自同一历史日
（桶内联合），已实现前缀 [0, 6τ_m) 填实现价。prices 为 None 即常数价
口径（题3，逐位不变）。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.data import DELTA
from nmmc.data.fcst import YearForecast
from nmmc.data.price import YearPrice
from nmmc.data.year import YearData
from nmmc.model.forecast import R_EPS, price_residual
from nmmc.model.lp import T
from nmmc.model.q3 import VINTAGE_HOURS

HOUR_SLOTS = slice(5, T, 6)   # 各整点对应时段（0 基 t = 6H−1，行标签为该整点）


@dataclass(frozen=True)
class Q3Config:
    """题3 预报层配置（文档 §4.5 网格三轴 + 各候选参数默认值）。

    Attributes:
        load: 负载日前基线候选（题2 L1/L2/L3，随题2 消融胜者）。
        fusion: 光伏融合候选，F1 反方差融合 / PB0 官方直用。
        load_update: 负载日内更新，LU1 比例缩放 / LU0 静态。
        pool: 残差池配置，S-b 提前期桶分池 / S-a 按阶段整条 / S-c 混池。
        pv_window: 融合与冷启动持续法的近窗（日）。
        rho_clip: LU1 缩放系数 ρ 的截断区间。
        ns: S-c 抽样情景数（S-b/S-a 为池内全取，池容量上限）。
        resid_window: 残差池回看窗（日）。
        seed: 随机种子。
    """

    load: str = "L1"
    fusion: str = "F1"
    load_update: str = "LU1"
    pool: str = "S-b"
    pv_window: int = 7
    rho_clip: tuple[float, float] = (0.7, 1.3)
    ns: int = 12
    resid_window: int = 28
    seed: int = 0


def hour_powers(r: np.ndarray) -> np.ndarray:
    """全天光伏电量 (n, 144) → 各整点功率 (n, 24)，kWh / Δ = kW。

    整点 H 对应时段 0 基 6H−1（行标签为时段末点，数据核验第 2 条）。
    """
    return r[:, HOUR_SLOTS] / DELTA


def fusion_sigmas(r_hour: np.ndarray, phi: np.ndarray, j: int,
                  window: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """两预报器逐提前期误差标准差（kW，文档 §4.1 的诊断量）。

    官方误差 = 实际整点功率 − 附件3；持续法误差 = 实际 − 该日自己的
    近 window 日均值。目标整点按 vintage τ_m + h 对齐，同一提前期 h 的
    样本跨 4 个 vintage 池化。

    Args:
        r_hour: 全年整点功率，shape (365, 24)。
        phi: 官方预报，shape (365, 4, 24)。
        j: 0 基日索引（只用 i < j）。
        window: 持续法近窗（去超参实验逐日选窗时传入，缺省 7 = 交付）。

    Returns:
        (σ_官方, σ_持续)，各 shape (24,)，kW；自由度不足处 NaN
        （自然规则：2 个样本才有方差可言，非可调阈值）。
    """
    # 持续法预报：近 window 日同时段均值（day 0 占位，误差层屏蔽）
    pers = np.empty((j, 24))
    if j:
        pers[0] = r_hour[0]                  # 占位，day 0 无持续法，误差层屏蔽
    for i in range(1, j):
        pers[i] = r_hour[max(0, i - window):i].mean(axis=0)

    # 目标整点对齐：vintage τ_m + h → 附件 2 整点列
    tau = np.array(VINTAGE_HOURS)
    tgt = tau[:, None] + np.arange(1, 25)[None, :]     # (4, 24) 目标整点
    valid = tgt <= 24
    hidx = np.where(valid, tgt - 1, 0)
    r_tgt = r_hour[:j][:, hidx]                         # (j, 4, 24)

    # 两预报器逐提前期误差：官方 err_fc = 实际 − 附件3；持续 err_pe = 实际 − 持续法
    err_fc = r_tgt - phi[:j]
    err_pe = r_tgt - pers[:, hidx]
    has_day = np.arange(j)[:, None, None] >= 1
    mask = valid[None] & has_day

    sig_fc = np.full(24, np.nan)
    sig_pe = np.full(24, np.nan)
    for lead in range(24):
        m_ = mask[:, :, lead]
        if m_.sum() < 2:
            continue                # 自由度 ≥ 2：2 个样本才有方差可言
        sig_fc[lead] = np.sqrt(err_fc[:, :, lead][m_].var())
        sig_pe[lead] = np.sqrt(err_pe[:, :, lead][m_].var())
    return sig_fc, sig_pe


def fusion_weights(r_hour: np.ndarray, phi: np.ndarray, j: int,
                   window: int = 7) -> np.ndarray:
    """滚动估计反方差融合权重 φ(h)，h = 1..24（文档 §4.1）。

    φ = σ_pe²/(σ_pe² + σ_fc²) 为官方预报的权重（σ 取自
    fusion_sigmas；自由度不足的提前期官方直用 φ=1）。无冷启动
    阈值：j ∈ {0, 1} 无可用样本（day 0 无持续法），全提前期自动
    φ = 1 退化为官方直用；自第 2 日起自由度 ≥ 2 的提前期先行
    融合，权重噪声随样本累积自平均（方差估计无偏，少样本是
    噪声不是偏差）。

    Args:
        r_hour: 全年整点功率，shape (365, 24)。
        phi: 官方预报，shape (365, 4, 24)。
        j: 0 基日索引（只用 i < j）。
        window: 持续法近窗（去超参实验逐日选窗时传入，缺省 7 = 交付）。

    Returns:
        φ(h)，shape (24,)。
    """
    # 反方差权重：φ = σ_pe²/(σ_pe² + σ_fc²)（NaN 提前期官方直用 φ=1）
    sig_fc, sig_pe = fusion_sigmas(r_hour, phi, j, window)
    var_fc, var_pe = sig_fc**2, sig_pe**2
    den = var_pe + var_fc
    out = np.ones(24)
    ok = ~np.isnan(den) & (den > 1e-9)
    out[ok] = var_pe[ok] / den[ok]
    return out


def pv_point(cfg: Q3Config, year: YearData, fc: YearForecast, j: int, m: int,
             typ_r: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """阶段 m 的光伏点预报（文档 §4.1）：官方预报 I1 左锚线性降尺度后
    与近 7 日持续法按 φ(h) 融合。已实现前缀填实际值（预报语义只在
    t ≥ 6τ_m）。

    Args:
        cfg: 题3 预报层配置。
        year: 全年数据（持续法与左锚只用 [:j] 与当日已实现段）。
        fc: 附件3 官方预报。
        j: 0 基日索引。
        m: 修订阶段 0..3。
        typ_r: 附件1 典型日光伏曲线（无历史时持续法先验；j = 0 融合
            权重恒为 1，先验实际不生效，仅为防御性兜底）。
        weights: fusion_weights 的输出 φ(h)（PB0 时忽略）。

    Returns:
        点预报 r̂，shape (144,)，已实现前缀为实际值。
    """
    h0 = 36 * m
    tau = VINTAGE_HOURS[m]

    # 左锚：τ_m 已实现功率（m=0 无已实现段，锚 0）
    anchor = 0.0 if m == 0 else year.r[j, h0 - 1] / DELTA

    # I1 线性降尺度：左锚 + 官方整点预报为节点，线性插值到 10 分钟网格 × Δ
    node_t = np.array([tau] + [tau + h for h in range(1, 25 - tau)], float)
    node_v = np.concatenate([[anchor], fc.phi[j, m, :24 - tau]])
    theta = (np.arange(h0, T) + 1) * DELTA             # 各时段末点，小时
    off = np.interp(theta, node_t, node_v) * DELTA      # kWh

    # PB0：官方降尺度直用
    if cfg.fusion == "PB0":
        tail = off
    else:
        # F1 融合：r̂ = φ(h)·降尺度 + (1−φ(h))·持续法，提前期 h = ceil(θ−τ)
        hist = year.r[:j]
        pers = (hist[-cfg.pv_window:].mean(axis=0) if len(hist) else typ_r)[h0:]
        lead = np.ceil(theta - tau).astype(int) - 1      # 提前期 h = ceil(θ−τ)
        tail = weights[lead] * off + (1.0 - weights[lead]) * pers

    # 已实现前缀填实际值（预报语义只在 t ≥ 6τ_m），负值截 0
    out = year.r[j].copy()
    out[h0:] = np.maximum(tail, 0.0)
    return out


def load_point(cfg: Q3Config, ell_hat0: np.ndarray, year: YearData,
               j: int, m: int) -> np.ndarray:
    """阶段 m 的负载点预报（文档 §4.2）：LU1 用当日已实现段与基线同段
    之比缩放；LU0 静态。m = 0 无已实现段，恒为基线。"""
    h0 = 36 * m

    # LU0 / m=0 无已实现段：静态沿用基线
    if cfg.load_update == "LU0" or m == 0:
        return ell_hat0

    # LU1：ρ = Σℓ_已实现/Σℓ̂_基线（截断 [ρ_lo, ρ_hi]），ℓ̂ ← ℓ̂·ρ
    done = slice(0, h0)
    den = float(ell_hat0[done].sum())
    rho = (year.ell[j][done].sum() / den if den > 1e-9 else 1.0)
    return ell_hat0 * np.clip(rho, *cfg.rho_clip)


def build_scenarios(cfg: Q3Config, year: YearData,
                    hist: list[list[tuple[np.ndarray, np.ndarray]]], j: int,
                    m: int, ell_hat: np.ndarray, r_hat: np.ndarray,
                    rng: np.random.Generator, *, price: YearPrice | None = None,
                    p_fcasts: list[np.ndarray] | None = None,
                    p_hat: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray | None]:
    """逐阶段分池残差场景（文档 §4.3）：伪样本外残差整条贴加。

    残差定义（只对预报段 t ≥ 6τ_m）：负载加性 ε^L = ℓ − ℓ̂；光伏相对
    ε^R = (r − r̂)/max(r̂, r_ε)。池 = 近 resid_window 日（含当日阶段预报
    历史）中最新 ns 条阶段 m 残差曲线。提前期桶 {1–6, 7–12, 13–24} h
    与修订块对齐（文档 §4.3）：S-b 各桶独立洗牌拼接、S-a 整条全取、
    S-c 跨阶段混池（m_ ≤ m）有放回抽样。

    电价段（题4 档 §4.2，给定价格输入时）：ε^P = (p − p̂)/p̂ 不分提前期，
    S-b 下与该桶 (ε^L, ε^R) 取自同一历史日（桶内三元联合），已实现前缀
    [0, 6τ_m) 填实现价。

    Args:
        cfg: 题3 预报层配置。
        year: 全年数据。
        hist: 历史各日四阶段点预报 [(ℓ̂, r̂) × 4]，长度应为 j。
        j: 0 基日索引。
        m: 修订阶段。
        ell_hat / r_hat: 当日阶段 m 点预报。
        rng: 固定种子随机源（S-b 洗牌与 S-c 抽样）。
        price: 附件4 全年实现电价（价格残差真值源；题4 口径）。
        p_fcasts: 历史各日当时的电价点预报 p̂_i（与 hist 同步滚动，
            ε^P 不分阶段——电价不做日内更新）。
        p_hat: 当日 0:00 电价点预报。

    Returns:
        (nets, prices)：净负荷场景 n^(s)，shape (N_s, 144)，已实现前缀
        为真值；电价场景 p^(s)（题4 口径），常数价口径为 None。
    """
    h0 = 36 * m
    base = ell_hat - r_hat
    seg = slice(h0, T)

    # 无阶段预报历史：退化 N_s = 1（点预报本身）
    if not hist:
        prices = (None if p_hat is None
                  else np.concatenate([price.p[j, :h0], p_hat[h0:]])
                  [None, :])
        return base[None, :], prices
    pool = np.arange(max(0, j - cfg.resid_window), j)[-cfg.ns:]

    # 残差（只对预报段 t ≥ 6τ_m）：ε^L = ℓ − ℓ̂（加性）；ε^R = (r − r̂)/max(r̂, r_ε)
    def residuals(i: int) -> tuple[np.ndarray, np.ndarray]:
        ell_i, r_i = hist[i][m]
        eps_l = year.ell[i][seg] - ell_i[seg]
        eps_r = (year.r[i][seg] - r_i[seg]) / np.maximum(r_i[seg], R_EPS)
        return eps_l, eps_r

    # S-c：混池只取覆盖当前预报段的阶段（m_ ≤ m）有放回抽样——更晚阶段
    # 残差在近段恒为 0（已实现前缀），混入会注入伪完美预见情景
    if cfg.pool == "S-c":
        flat = [(i, m_) for i in pool for m_ in range(m + 1)]
        draws = [flat[k] for k in rng.integers(0, len(flat), size=cfg.ns)]
        pairs = [(year.ell[i][seg] - hist[i][m_][0][seg],
                  (year.r[i][seg] - hist[i][m_][1][seg])
                  / np.maximum(hist[i][m_][1][seg], R_EPS))
                 for i, m_ in draws]
        days = [np.full(T - h0, int(i)) for i, _ in draws]
    # S-b：提前期桶 {1–6, 7–12, 13–24} h（与修订块对齐）各桶独立洗牌拼接
    elif cfg.pool == "S-b":
        buckets = [b for b in
                   ((h0, h0 + 36), (h0 + 36, h0 + 72), (h0 + 72, T))
                   if b[0] < min(b[1], T)]
        orders = [rng.permutation(len(pool)) for _ in buckets]
        pairs = []
        days = []
        # 情景 s：逐桶取洗牌后第 s 条残差曲线的对应段拼接
        for s in range(len(pool)):
            eps_l = np.zeros(T - h0)
            eps_r = np.zeros(T - h0)
            day_s = np.zeros(T - h0, dtype=int)
            for (lo, hi), order in zip(buckets, orders):
                i = int(pool[order[s]])
                e_l, e_r = residuals(i)
                eps_l[lo - h0:hi - h0] = e_l[lo - h0:hi - h0]
                eps_r[lo - h0:hi - h0] = e_r[lo - h0:hi - h0]
                day_s[lo - h0:hi - h0] = i
            pairs.append((eps_l, eps_r))
            days.append(day_s)
    else:                                                # S-a：整条全取
        pairs = [residuals(int(i)) for i in pool]
        days = [np.full(T - h0, int(i)) for i in pool]

    # 贴加（同题2）：ℓ^(s) = [ℓ̂+ε^L]_+，r^(s) = [r̂(1+ε^R)]_+；已实现前缀为真值
    nets = np.empty((len(pairs), T))
    prices = np.empty((len(pairs), T)) if p_hat is not None else None
    for s, (eps_l, eps_r) in enumerate(pairs):
        nets[s, :h0] = base[:h0]
        nets[s, seg] = (np.maximum(ell_hat[seg] + eps_l, 0.0)
                        - np.maximum(r_hat[seg] * (1.0 + eps_r), 0.0))
        if prices is not None:
            # 电价段：预报段同日贴加 ε^P（S-b 按桶同日），前缀填实现价
            prices[s, :h0] = price.p[j, :h0]
            eps_p = np.zeros(T - h0)
            for d in np.unique(days[s]):
                mask = days[s] == d
                eps_p[mask] = price_residual(price, p_fcasts, int(d))[seg][mask]
            prices[s, seg] = np.maximum(p_hat[seg] * (1.0 + eps_p), 0.0)
    return nets, prices
