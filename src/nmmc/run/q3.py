"""题3 执行层：嵌套双环（外环锁块、内环随机 MPC 交付 + CE 配对对照）
+ 全年滚动 + PI/事后诊断。

文档 §8 与 §10：外环每日 4 次在 τ_m 解阶段 LP 并锁定本块承诺 a(B_m)
（偏差费即时入账）；内环每 10 分钟以真实 SOC 重锚、当步真值 + 剩余时域
按当前阶段情景集分支解两阶段 LP（run_smpc_segment，交付）、只执行当步；
同链配跑确定性等价 MPC（run_mpc_segment，未来压成 α^rt=0.65 分位参考
路径——v4 基线机制冻结，文档 §12.3 执行层增益的对照组）。跨日唯一反馈
是真实路径 S_144 与当日实际值；诚实回测纪律同题2 档 §9。
"""
from dataclasses import dataclass

import numpy as np

from nmmc.data.fcst import YearForecast
from nmmc.data.year import YearData
from nmmc.model.forecast import (ForecastConfig, forecast_load,
                                 reference_path)
from nmmc.model.forecast3 import (Q3Config, build_scenarios, fusion_weights,
                                  hour_powers, load_point, pv_point)
from nmmc.model.lp import ETA, T
from nmmc.model.q2 import KAPPA, NU_END, solve_planning, solve_recourse
from nmmc.model.q3 import settle_cost, solve_stage, solve_step, solve_tail
from nmmc.run.q2 import ALPHA_RT, S0_INIT


@dataclass
class DayResult:
    """单日回测结果。

    Attributes:
        j: 0 基日索引。
        g: 0:00 计划购电量（计费基准），shape (144,)。
        a: 全天最终注入（B_0 段 = g，B_1..3 段为各阶段锁定值），shape (144,)。
        ell_hat / r_hat: 0:00 点预报（诊断用）。
        ns: 阶段 0 情景数。
        c / d / u / w: 随机 MPC 真实路径补救量（母线侧），shape (144,)。
        S: 真实路径储电量，shape (145,)。
        z: 当日现金费用 Σ[p·a + ½p|a−g|] + κΣp·u，元。
        z_ce: 确定性等价 MPC 配对对照的现金费用（同 g、a、同日初 SOC，
            未来压成 α^rt=0.65 分位路径），元——执行层增益 = z_ce − z。
        z_expost: 事后最优补救的现金费用（同 g、a，同真路径），元。
        z_pi: 完美信息（0:00 已知真值）的现金费用，元。
    """

    j: int
    g: np.ndarray
    a: np.ndarray
    ell_hat: np.ndarray
    r_hat: np.ndarray
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


def run_mpc_segment(p: np.ndarray, net: np.ndarray, committed: np.ndarray,
                    g: np.ndarray, ref: np.ndarray, c: np.ndarray,
                    d: np.ndarray, u: np.ndarray, w: np.ndarray,
                    s: np.ndarray, h0: int, h1: int, nu_end: float,
                    kappa: float, free_from: int | None = None) -> None:
    """确定性等价 MPC（CE 对照臂）：就地执行本块时段 [h0, h1)。

    与交付臂 run_smpc_segment 只差未来表示：当前时段用真值 net，未来
    压成 α^rt 分位参考路径 ref（v4 基线机制冻结，文档 §12.3 执行层
    增益的对照组）；注入在 [t, free_from) 段锁定为 committed（默认
    free_from = h1，即本块；逐级增量实验中更晚块未行使修订权时冻结到
    下一行使块首或日末，文档 §12.2）。

    Args:
        p: 逐时段电价，shape (144,)。
        net: 当日真实净负荷，shape (144,)。
        committed: 截至本块的已承诺注入（[free_from, 144) 段取值未定、
            被掩码）。
        g: 0:00 计划（未锁定段的偏差基准）。
        ref: α^rt 分位参考路径，shape (144,)。
        c / d / u / w / s: 全日数组（就地写入 [h0, h1) 与 s[h0+1..h1]，
            s[h0] 为本块起点真实储电量）。
        h0 / h1: 本块起止时段（0 基，左闭右开）。
        nu_end / kappa: 末端估值 / 紧急倍率。
        free_from: 注入解锁（退回内部变量）的首时段；缺省 h1。
    """
    free_from = h1 if free_from is None else free_from
    for t in range(h0, h1):
        slots = np.arange(t, T)
        free = slots >= free_from
        comm = np.where(free, 0.0, committed[slots])
        tail_net = np.concatenate([net[t:t + 1], ref[t + 1:]])
        c_t, d_t, u_t, w_t, _ = solve_tail(p[t:], tail_net, comm, free,
                                           g[t:], s[t], nu_end, kappa)
        if c_t[0] > 0.0 and d_t[0] > 0.0:
            # 性质 3 同款：HiGHS 可能返回 c·d>0 的等价最优顶点，按非等量
            # 缩减归一化，SOC 轨迹与费用严格不变（继承 q2 执行层处理）。
            x = min(c_t[0], d_t[0] / ETA**2)
            c_t[0] -= x
            d_t[0] -= ETA**2 * x
            w_t[0] += x * (1 - ETA**2)
        c[t], d[t], u[t], w[t] = c_t[0], d_t[0], u_t[0], w_t[0]
        s[t + 1] = s[t] + ETA * c[t] - d[t] / ETA


def run_smpc_segment(p: np.ndarray, net: np.ndarray, committed: np.ndarray,
                     g: np.ndarray, scn: np.ndarray, probs: np.ndarray,
                     c: np.ndarray, d: np.ndarray, u: np.ndarray,
                     w: np.ndarray, s: np.ndarray, h0: int, h1: int,
                     nu_end: float, kappa: float,
                     free_from: int | None = None) -> None:
    """内环在线随机 MPC（交付）：就地执行本块时段 [h0, h1)。

    逐时段解 solve_step——当步 here-and-now 共享、剩余时域按当前阶段
    情景集分支（锁定段注入冻结、自由段按偏差费 wait-and-see），只执行
    当步、SOC 按真实递推推进。与 CE 对照臂 run_mpc_segment 只差未来
    表示：情景分布 vs 单条 α^rt 分位路径——影子价格按缺口发生概率
    折算，而非按必然发生。

    Args:
        p: 逐时段电价，shape (144,)。
        net: 当日真实净负荷，shape (144,)。
        committed: 截至本块的已承诺注入（[free_from, 144) 段取值未定、
            被掩码）。
        g: 0:00 计划（未锁定段的偏差基准）。
        scn: 当前阶段净负荷场景 n^(s)，shape (N_s, 144)。
        probs: 情景概率 π_s，shape (N_s,)。
        c / d / u / w / s: 全日数组（就地写入 [h0, h1) 与 s[h0+1..h1]，
            s[h0] 为本块起点真实储电量）。
        h0 / h1: 本块起止时段（0 基，左闭右开）。
        nu_end / kappa: 末端估值 / 紧急倍率。
        free_from: 注入解锁（退回内部变量）的首时段；缺省 h1。
    """
    free_from = h1 if free_from is None else free_from
    free_mask = np.arange(T) >= free_from
    for t in range(h0, h1):
        step = solve_step(p, net, committed, free_mask, g, scn, probs, t,
                          s[t], nu_end, kappa)
        c_t, d_t = step.c, step.d
        u[t], w[t] = step.u, step.w
        if c_t > 0.0 and d_t > 0.0:
            # 同 run_mpc_segment：HiGHS 可能返回 c·d>0 的等价最优顶点，
            # 按非等量缩减归一化，SOC 轨迹与费用严格不变。
            x = min(c_t, d_t / ETA**2)
            c_t -= x
            d_t -= ETA**2 * x
            w[t] += x * (1 - ETA**2)
        c[t], d[t] = c_t, d_t
        s[t + 1] = s[t] + ETA * c_t - d_t / ETA


def rollout(year: YearData, fc: YearForecast, p: np.ndarray,
            typ_ell: np.ndarray, typ_r: np.ndarray, cfg: Q3Config,
            nu_end: float = NU_END, kappa: float = KAPPA,
            days: int | None = None,
            epochs: tuple[int, ...] = (0, 1, 2, 3)
            ) -> list[DayResult]:
    """全年诚实回测（文档 §10）：外环四阶段 → 内环随机 MPC（交付）+
    确定性等价 MPC（CE 配对对照，α^rt=0.65 冻结）→ SOC 链推进。

    两臂同日同 g、同 committed、同日初 SOC，只差内环未来表示（情景
    分布 vs 单条分位路径）；SOC 链按交付臂推进，CE 臂仅供 z_ce 诊断
    （文档 §12.3 执行层增益 = Z^CE − Z^online 的逐日素材）。

    Args:
        year: 全年数据（按日解锁使用）。
        fc: 附件3 官方光伏预报。
        p: 逐时段电价（附件1，全年相同）。
        typ_ell / typ_r: 附件1 典型日曲线（冷启动先验）。
        cfg: 题3 预报层配置。
        nu_end / kappa: 末端估值 / 紧急倍率。
        days: 只滚动前 N 天（含暖机）；缺省全年 365。
        epochs: 行使修订权的阶段子集（0 基阶段号，须含 0，文档 §12.2
            逐级增量）。未行使阶段不重解承诺，注入冻结为既有承诺（后续
            块即计划 g）直至下一行使块首；四阶段点预报与场景照常构建
            （信息可得性与 rng 流不随机制改变，跨档可比）。

    Returns:
        逐日结果列表，暖机期完整包含（报告期切片在 report 层）。

    Raises:
        ValueError: epochs 不含 0 或越界。
    """
    exercised = tuple(sorted(set(epochs)))
    if not exercised or exercised[0] != 0 or exercised[-1] > 3:
        raise ValueError(f"epochs 须为含 0 的阶段子集，收到 {epochs!r}")
    rng = np.random.default_rng(cfg.seed)
    cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
    r_hour = hour_powers(year.r)
    hist: list[list[tuple[np.ndarray, np.ndarray]]] = []
    soc = S0_INIT
    out: list[DayResult] = []
    n_days = days or len(year.dates)
    for j in range(n_days):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, typ_ell)
        net = year.ell[j] - year.r[j]
        g = None
        committed = None
        stage_fcasts: list[tuple[np.ndarray, np.ndarray]] = []
        z_pi = np.inf
        ns0 = 1
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
            scn = build_scenarios(cfg, year, hist, j, m, ell_hat, r_hat, rng)
            probs = np.full(len(scn), 1.0 / len(scn))
            if m in exercised:
                res = solve_stage(p, g, committed, scn, probs, m, s[h0],
                                  nu_end, kappa)
                if m == 0:
                    g = res.commit
                    ns0 = len(scn)
                    # PI 诊断：0:00 已知真值 ⇒ 确定性全天 LP（题2 模板，无偏差
                    # 机制——完美信息下 a≡g，一次定准）
                    pi = solve_planning(p, net[None, :], np.ones(1), s[0],
                                        nu_end, kappa)
                    u_pi = solve_recourse(p, net, pi.g, s[0], nu_end,
                                          kappa).u
                    z_pi = settle_cost(p, pi.g, pi.g, u_pi, kappa)
                committed = res.commit
            # CE 臂参考路径：当前阶段情景的 α^rt 分位（v4 基线，冻结不调）
            ref = reference_path(scn, ALPHA_RT)
            nxt = next((e for e in exercised if e > m), None)
            free_from = T if nxt is None else 36 * nxt
            run_smpc_segment(p, net, committed, g, scn, probs, c, d, u, w,
                             s, h0, h1, nu_end, kappa, free_from=free_from)
            run_mpc_segment(p, net, committed, g, ref, cec, ced, ceu, cew,
                            ce_s, h0, h1, nu_end, kappa, free_from=free_from)
        u_exp = solve_recourse(p, net, committed, soc, nu_end, kappa).u
        out.append(DayResult(
            j=j, g=g, a=committed, ell_hat=ell0, r_hat=stage_fcasts[0][1],
            ns=ns0, c=c, d=d, u=u, w=w, S=s,
            z=settle_cost(p, g, committed, u, kappa),
            z_ce=settle_cost(p, g, committed, ceu, kappa),
            z_expost=settle_cost(p, g, committed, u_exp, kappa),
            z_pi=z_pi))
        soc = s[T]
        hist.append(stage_fcasts)
        if (j + 1) % 31 == 0:
            print(f"已回测 {j + 1}/{n_days} 天，年末 SOC = {soc:.1f} kWh",
                  flush=True)
    return out
