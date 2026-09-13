"""题4 执行层：波动价双线全年滚动（Q4-2 题2 骨架 / Q4-3 题3 骨架）。

读法 R（文档 §2.3 / A18）：0:00 只掌握截至昨日的全部实现值（负载、光伏、
电价）与日历，当日三源真值封闭。规划层用 PR1 预报价 + 三元联合情景价
（一阶段项按性质 2 折叠为情景均值）；执行层当步可见实现价（题面"交易
时刻电价"，用于当期紧急购电定价）、未来分支用情景价；结算一律用实现
价（§8.1）。三种价格口径共用本实现（§6.3，靠情景矩阵的三种填法切换）：

    r（读法 R，主交付）  情景价 = 预报价 × 同日联合残差；折叠价 = 情景均值
    d（读法 D，诊断）    情景退化：全时域用当日实现价（0:00 已知价格的
                         上界口径，Z^D 基准）
    a1（附件1，C4-2）    全时域用附件1 常数价 + ν^end ≡ 0.472 ⇒ 与
                         run.q2 / run.q3 同函数、同参数流，逐日精确复现
                         题2 v6 / 题3 v7 的解与费用

相对 hjc 原管线的基础对齐（只改 zzj 工作区）：
    ① 紧急购电上界 u ≤ (n−g)⁺ / (n−a)⁺，规划层迭代到不动点，四条臂同口径；
    ② Q4-3 的 0:00 走题2 两阶段 SAA（题3 v7 / CF-A），不再把 a^(s) 写进 g。
"""
from dataclasses import dataclass, replace

import numpy as np

from nmmc.data.fcst import YearForecast
from nmmc.data.price import YearPrice
from nmmc.data.year import YearData
from nmmc.model.forecast import (ForecastConfig, build_scenarios, forecast_load,
                                 forecast_pv, reference_path)
from nmmc.model.forecast3 import Q3Config, build_scenarios as build_scenarios3
from nmmc.model.forecast3 import fusion_weights, hour_powers, load_point, pv_point
from nmmc.model.forecast4 import PriceConfig, forecast_price
from nmmc.model.lp import T
from nmmc.model.q2 import KAPPA, emergency_cap, solve_recourse
from nmmc.model.q3 import settle_cost
from nmmc.model.q4 import folded_price, nu_end_daily
from nmmc.run.q2 import (ALPHA_RT, S0_INIT, cash_cost, run_mpc_day, run_smpc_day,
                         solve_plan)
from nmmc.run.q3 import run_mpc_segment, run_smpc_segment, solve_stage_iter

MODES = ("r", "d", "a1")  # 价格口径：读法 R / 读法 D / 附件1 退化（C4-2）


@dataclass
class DayResult:
    """单日回测结果（双线共用；Q4-2 的 a 恒为 None，等价 a ≡ g）。

    Attributes:
        j: 0 基日索引。
        g: 计划购电量（0:00 锁定，计费基准），shape (144,)。
        a: 全天最终注入（仅 Q4-3；B_0 段 = g），shape (144,) 或 None。
        ell_hat / r_hat: 0:00 点预报（诊断用）。
        p_hat: 当日 0:00 电价点预报（诊断用；d 口径 = 实现价）。
        nu_end: 当日末端估值 ν^end^(j)（诊断用）。
        ns: 0:00 情景数 N_s。
        c / d / u / w: 随机 MPC 真实路径补救量（母线侧），shape (144,)。
        S: 真实路径储电量，shape (145,)。
        z: 当日现金费用（实现价结算），元——Q4-2 为 Σp·g + κΣp·u，
            Q4-3 为 Σ[p·a + ½p|a−g|] + κΣp·u。
        z_ce: 确定性等价 MPC 配对对照的现金费用（同 g、同日初 SOC）。
        z_expost: 事后最优补救的现金费用（同 g/a，同真路径），元。
        z_pi: 完美信息（0:00 已知当日 ℓ, r, p 真值）的现金费用，元。
        s_ce / s_expost / s_pi: 各臂日末 SOC（kWh）。C9 用
            Z̃ = Z^cash − ν^end·S_144^arm，避免 PI 把现金换成库存造成
            假阳性（波动价下现金口径会失效，题2 常数价碰巧不触发）。
        cap_resid / cap_iters: 规划层 u 上界序列线性化末轮残差与轮数。
    """

    j: int
    g: np.ndarray
    a: np.ndarray | None
    ell_hat: np.ndarray
    r_hat: np.ndarray
    p_hat: np.ndarray
    nu_end: float
    ns: int
    c: np.ndarray
    d: np.ndarray
    u: np.ndarray
    w: np.ndarray
    S: np.ndarray
    z: float
    z_ce: float
    z_expost: float
    z_pi: float
    s_ce: float = 0.0
    s_expost: float = 0.0
    s_pi: float = 0.0
    cap_resid: float = 0.0
    cap_iters: int = 0


def _price_view(mode: str, cfgp: PriceConfig, price: YearPrice, typ_p: np.ndarray,
                j: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """当日价格三视图：(结算/当步价 p_day, 0:00 点预报 p̂, 是否装配情景价)。

    a1：全时域附件1 常数价（C4-2 退化口径）；d：结算与预报同为实现价
    （0:00 已知价格）；r：结算用实现价、预报用 PR1、情景价装配开。
    """
    if mode == "a1":
        return typ_p, typ_p, False
    p_hat = forecast_price(cfgp, price, j, typ_p)
    return price.p[j], (p_hat if mode == "r" else price.p[j]), mode == "r"


def _nu_series(mode: str, cfgp: PriceConfig, price: YearPrice,
               typ_ell: np.ndarray, typ_r: np.ndarray, typ_p: np.ndarray,
               days: int) -> np.ndarray:
    """逐日 ν^end 序列；a1 口径强制 V0 常数 0.472（C4-2 复现要求）。"""
    rule = "V0" if mode == "a1" else cfgp.nu_rule
    return nu_end_daily(replace(cfgp, nu_rule=rule), price, typ_ell, typ_r,
                        typ_p, days)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"未知价格口径: {mode}（须为 {MODES} 之一）")


def rollout_q42(year: YearData, price: YearPrice, typ_ell: np.ndarray,
                typ_r: np.ndarray, typ_p: np.ndarray, cfg: ForecastConfig,
                cfgp: PriceConfig, mode: str = "r", kappa: float = KAPPA,
                alpha_rt: float = ALPHA_RT,
                days: int | None = None,
                emerg_cap: bool = True) -> list[DayResult]:
    """Q4-2 全年诚实回测：题2 v6 骨架（L1-P1-R2 + u 上界）+ 波动价。

    预报（负载/光伏/电价）→ 三元联合情景 → 逐日 ν^end → 日前场景 LP
    （g 折叠计价，序列线性化到不动点）→ 在线随机 MPC（当步实现价、
    未来情景价）→ 实现价结算 → SOC 链推进。四条臂同口径。

    Args:
        year: 附件2 全年数据（按日解锁使用）。
        price: 附件4 全年实现电价（按日解锁使用）。
        typ_ell / typ_r: 附件1 典型日曲线（暖机先验）。
        typ_p: 附件1 典型日电价（PR0 先验与 a1 退化口径）。
        cfg: 题2 预报层配置。
        cfgp: 题4 价格配置。
        mode: 价格口径 r / d / a1（模块 docstring）。
        kappa: 紧急购电倍率。
        alpha_rt: 确定性等价对照的参考路径分位（题2 v5 冻结 0.65）。
        days: 只滚动前 N 天（含暖机）；缺省全年 365。
        emerg_cap: True（默认）= u ≤ (n−g)⁺，四条臂同口径；False = v5 无上界。

    Returns:
        逐日结果列表（长度 = min(days, 365)），暖机期完整包含。
    """
    _check_mode(mode)
    n_days = days or len(year.dates)
    nu = _nu_series(mode, cfgp, price, typ_ell, typ_r, typ_p, n_days)
    rng = np.random.default_rng(cfg.seed)
    fcasts: list[tuple[np.ndarray, np.ndarray]] = []
    p_fcasts: list[np.ndarray] = []
    soc = S0_INIT
    out: list[DayResult] = []
    for j in range(n_days):
        ell_hat = forecast_load(cfg, year, j, typ_ell)
        r_hat = forecast_pv(cfg, year, j, typ_r)
        p_day, p_hat, use_scn = _price_view(mode, cfgp, price, typ_p, j)
        kwargs = (dict(price=price, p_fcasts=p_fcasts, p_hat=p_hat,
                       couple=cfgp.couple)
                  if use_scn else {})
        scn, prices = build_scenarios(cfg, year, fcasts, j, ell_hat, r_hat,
                                      rng, **kwargs)
        probs = np.full(len(scn), 1.0 / len(scn))
        folded = folded_price(prices, probs) if use_scn else p_day
        p_scn = prices if use_scn else None
        g, u_cap, resid, iters = solve_plan(
            folded, scn, probs, soc, nu[j], kappa, emerg_cap, p_scn=p_scn)
        ref = reference_path(scn, alpha_rt)
        net = year.ell[j] - year.r[j]
        # 执行层：当步计价用实现价 p_day，未来分支用情景价（题4 档 §7）
        c, d, u, w, s = run_smpc_day(p_day, net, g, scn, probs, soc,
                                     nu[j], kappa, u_cap, p_scn=p_scn)
        _, _, ce_u, _, ce_s = run_mpc_day(
            p_day, net, g, soc, ref, nu[j], kappa,
            emerg_cap=emerg_cap, p_fut=folded)
        cap_path = emergency_cap(net, g) if emerg_cap else None
        expost = solve_recourse(p_day, net, g, soc, nu[j], kappa, cap_path)
        pi_g, _, _, _ = solve_plan(
            p_day, net[None, :], np.ones(1), soc, nu[j], kappa, emerg_cap)
        pi_cap = emergency_cap(net, pi_g) if emerg_cap else None
        pi = solve_recourse(p_day, net, pi_g, soc, nu[j], kappa, pi_cap)
        out.append(DayResult(
            j=j, g=g, a=None, ell_hat=ell_hat, r_hat=r_hat, p_hat=p_hat,
            nu_end=float(nu[j]), ns=len(scn), c=c, d=d, u=u, w=w, S=s,
            z=cash_cost(p_day, g, u, kappa),
            z_ce=cash_cost(p_day, g, ce_u, kappa),
            z_expost=cash_cost(p_day, g, expost.u, kappa),
            z_pi=cash_cost(p_day, pi_g, pi.u, kappa),
            s_ce=float(ce_s[T]), s_expost=float(expost.S[-1]),
            s_pi=float(pi.S[-1]),
            cap_resid=resid, cap_iters=iters))
        soc = s[T]
        fcasts.append((ell_hat, r_hat))
        p_fcasts.append(p_hat)
        if (j + 1) % 31 == 0:
            print(f"已回测 {j + 1}/{n_days} 天，年末 SOC = {soc:.1f} kWh",
                  flush=True)
    return out


def rollout_q43(year: YearData, fc: YearForecast, price: YearPrice,
                typ_ell: np.ndarray, typ_r: np.ndarray, typ_p: np.ndarray,
                cfg: Q3Config, cfgp: PriceConfig, mode: str = "r",
                kappa: float = KAPPA, days: int | None = None,
                epochs: tuple[int, ...] = (0, 1, 2, 3),
                emerg_cap: bool = True) -> list[DayResult]:
    """Q4-3 全年诚实回测：题3 v7 骨架（日前 SAA + 嵌套双环）+ 波动价。

    0:00 用题2 两阶段 SAA 定 g（CF-A）；6/12/18 锁块。内环随机 MPC
    （当步实现价、未来情景价），CE 配对对照。两条交付线独立链 SOC（A26）。

    Args:
        year: 附件2 全年数据（按日解锁使用）。
        fc: 附件3 官方光伏预报。
        price: 附件4 全年实现电价（按日解锁使用）。
        typ_ell / typ_r: 附件1 典型日曲线（冷启动先验）。
        typ_p: 附件1 典型日电价（PR0 先验与 a1 退化口径）。
        cfg: 题3 预报层配置。
        cfgp: 题4 价格配置。
        mode: 价格口径 r / d / a1（模块 docstring）。
        kappa: 紧急购电倍率。
        days: 只滚动前 N 天（含暖机）；缺省全年 365。
        epochs: 行使修订权的阶段子集（语义同 run.q3.rollout）。
        emerg_cap: True（默认）= u ≤ (n−a)⁺，四条臂同口径。

    Returns:
        逐日结果列表，暖机期完整包含（报告期切片在 report 层）。

    Raises:
        ValueError: epochs 不含 0 或越界。
    """
    _check_mode(mode)
    exercised = tuple(sorted(set(epochs)))
    if not exercised or exercised[0] != 0 or exercised[-1] > 3:
        raise ValueError(f"epochs 须为含 0 的阶段子集，收到 {epochs!r}")
    n_days = days or len(year.dates)
    nu = _nu_series(mode, cfgp, price, typ_ell, typ_r, typ_p, n_days)
    rng = np.random.default_rng(cfg.seed)
    cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
    r_hour = hour_powers(year.r)
    hist: list[list[tuple[np.ndarray, np.ndarray]]] = []
    p_fcasts: list[np.ndarray] = []
    soc = S0_INIT
    out: list[DayResult] = []
    for j in range(n_days):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, typ_ell)
        net = year.ell[j] - year.r[j]
        p_day, p_hat, use_scn = _price_view(mode, cfgp, price, typ_p, j)
        g = None
        committed = None
        stage_fcasts: list[tuple[np.ndarray, np.ndarray]] = []
        z_pi = np.inf
        s_pi = 0.0
        ns0 = 1
        cap_resid, cap_iters = 0.0, 0
        c, d, u, w = (np.zeros(T) for _ in range(4))
        s = np.zeros(T + 1)
        s[0] = soc
        # CE 对照臂的平行数组：同 g、同 committed、同日初 SOC，SOC 链
        # 不入跨日反馈（仅供 z_ce，机制归因干净）
        cec, ced, ceu, cew = (np.zeros(T) for _ in range(4))
        ce_s = np.zeros(T + 1)
        ce_s[0] = soc
        for m in range(4):
            h0, h1 = 36 * m, 36 * (m + 1)
            r_hat = pv_point(cfg, year, fc, j, m, typ_r, weights)
            ell_hat = ell0 if m == 0 else load_point(cfg, ell0, year, j, m)
            stage_fcasts.append((ell_hat, r_hat))
            kwargs = (dict(price=price, p_fcasts=p_fcasts, p_hat=p_hat,
                           couple=cfgp.couple)
                      if use_scn else {})
            scn, prices = build_scenarios3(cfg, year, hist, j, m, ell_hat,
                                           r_hat, rng, **kwargs)
            probs = np.full(len(scn), 1.0 / len(scn))
            folded = folded_price(prices, probs) if use_scn else p_day
            p_scn = prices if use_scn else None
            if m in exercised:
                if m == 0:
                    g, _, resid0, iters0 = solve_plan(
                        folded, scn, probs, s[0], nu[j], kappa, emerg_cap,
                        p_scn=p_scn)
                    committed = g.copy()
                    ns0 = len(scn)
                    cap_resid, cap_iters = resid0, iters0
                    pi_g, _, _, _ = solve_plan(
                        p_day, net[None, :], np.ones(1), s[0], nu[j],
                        kappa, emerg_cap)
                    pi_cap = (emergency_cap(net, pi_g) if emerg_cap
                              else None)
                    pi = solve_recourse(p_day, net, pi_g, s[0], nu[j],
                                        kappa, pi_cap)
                    z_pi = settle_cost(p_day, pi_g, pi_g, pi.u, kappa)
                    s_pi = float(pi.S[-1])
                else:
                    res, resid_m, iters_m = solve_stage_iter(
                        folded, g, committed, scn, probs, m, s[h0], nu[j],
                        kappa, emerg_cap, p_scn=p_scn)
                    committed = res.commit
                    if resid_m > cap_resid:
                        cap_resid = resid_m
                    cap_iters = max(cap_iters, iters_m)
            # CE 臂参考路径：当前阶段情景的 α^rt 分位（v4 基线，冻结不调）
            ref = reference_path(scn, ALPHA_RT)
            nxt = next((e for e in exercised if e > m), None)
            free_from = T if nxt is None else 36 * nxt
            run_smpc_segment(p_day, net, committed, g, scn, probs, c, d, u,
                             w, s, h0, h1, nu[j], kappa, free_from, p_scn,
                             emerg_cap=emerg_cap)
            run_mpc_segment(p_day, net, committed, g, ref, cec, ced, ceu,
                            cew, ce_s, h0, h1, nu[j], kappa, free_from,
                            p_fut=folded, emerg_cap=emerg_cap)
        expost_cap = (emergency_cap(net, committed) if emerg_cap else None)
        expost = solve_recourse(p_day, net, committed, soc, nu[j], kappa,
                                expost_cap)
        out.append(DayResult(
            j=j, g=g, a=committed, ell_hat=ell0, r_hat=stage_fcasts[0][1],
            p_hat=p_hat, nu_end=float(nu[j]), ns=ns0, c=c, d=d, u=u, w=w,
            S=s, z=settle_cost(p_day, g, committed, u, kappa),
            z_ce=settle_cost(p_day, g, committed, ceu, kappa),
            z_expost=settle_cost(p_day, g, committed, expost.u, kappa),
            z_pi=z_pi, s_ce=float(ce_s[T]),
            s_expost=float(expost.S[-1]), s_pi=s_pi,
            cap_resid=cap_resid, cap_iters=cap_iters))
        soc = s[T]
        hist.append(stage_fcasts)
        p_fcasts.append(p_hat)
        if (j + 1) % 31 == 0:
            print(f"已回测 {j + 1}/{n_days} 天，年末 SOC = {soc:.1f} kWh",
                  flush=True)
    return out
