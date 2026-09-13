"""题2 执行层：在线随机 MPC（交付）、确定性等价 MPC（消融对照）、
事后最优与完美信息（诊断下界）、全年滚动。

主从关系（文档 §7）：交付值 Z^online 来自满足非预期性的在线随机
MPC——逐时段以真实 SOC 重锚、当步用真值净负荷、剩余时域保留 0:00
情景集解两阶段 LP、只执行当步（§7.2）；确定性等价 MPC 把未来压成
单条 α^rt 分位参考路径，为执行层消融对照 Z^CE（§7.3）——两者只差
"未来是一条路径还是一个分布"，同 g 链上的费用差是干净的机制归因；
事后最优与完美信息为两个诊断下界（§7.4/§7.5）。全年滚动按 §9
诚实回测协议：跨日唯一反馈是真实路径 S_144 与当日实际值。
"""
from dataclasses import dataclass, replace

import numpy as np

from nmmc.data.year import YearData
from nmmc.model.forecast import (ForecastConfig, build_scenarios, forecast_load,
                                 forecast_pv, reference_path)
from nmmc.model.lp import ETA, T
from nmmc.model.q2 import (KAPPA, NU_END, emergency_cap, solve_planning,
                           solve_recourse, solve_stochastic_step)

S0_INIT = 6000.0  # 2025.1.1 0:00 储电量，全年 SOC 链的锚（文档 §2.2）
ALPHA_RT = 0.65   # 确定性等价对照的参考路径分位（v4 交付参数，文档 §7.3）
CAP_TOL = 1e-6    # 规划层 u 上界不动点判据 ‖Δg‖₁/Σg（文档 §6.1）
CAP_MAX = 12      # 序列线性化轮数上限（防振荡不收敛；命中即如实报告）


@dataclass
class DayResult:
    """单日回测结果。

    Attributes:
        j: 0 基日索引。
        g: 计划购电量（0:00 锁定），shape (144,)。
        ell_hat / r_hat: 当日点预报，shape (144,)。
        ns: 当日情景数 N_s。
        c / d / u / w: 交付执行层真实路径补救量（母线侧），shape (144,)。
        S: 真实路径储电量，shape (145,)，S[0]=S_0，S[144]=24:00。
        z: 当日现金费用 Σp·g + κΣp·u，元。
        z_ce: 确定性等价 MPC 的现金费用（同 g、同日初 SOC，消融对照）。
        z_expost: 事后最优补救的现金费用（同 g、同真路径），元。
        z_pi: 完美信息的现金费用，元。
        cap_resid: 规划层 u 上界序列线性化的末轮残差 ‖Δg‖₁/Σg（自检
            C16）；旧口径（无上界）恒为 0。
        cap_iters: 序列线性化实际轮数；命中 CAP_MAX 即该日未收敛到不动点。
    """

    j: int
    g: np.ndarray
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
    cap_resid: float = 0.0
    cap_iters: int = 0


def cash_cost(p: np.ndarray, g: np.ndarray, u: np.ndarray,
              kappa: float = KAPPA) -> float:
    """现金费用 Z = Σp·g + κΣp·u（唯一上报口径，文档 §8.1）。"""
    return float(p @ g + kappa * (p @ u))


def solve_plan(p: np.ndarray, scn: np.ndarray, probs: np.ndarray, s0: float,
               nu_end: float, kappa: float, emerg_cap: bool,
               tol: float = CAP_TOL,
               max_passes: int = CAP_MAX,
               p_scn: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray | None, float, int]:
    """日前规划：新口径下带 u 上界，迭代至不动点（文档 §6.1）。

    上界 (n^s − g)⁺ 依赖第一阶段决策 g，作为 g 的上界约束非凸，故规划层
    不是单个 LP。按序列线性化处理：第 0 轮解无上界 LP 得 g⁰（= v5 口径），
    第 k 轮用 cap = (n^s − g^{k−1})⁺ 重解，每轮仍是纯 LP、亚秒级。

    **收敛判据**：迭代到 ‖g^k − g^{k−1}‖₁/Σg ≤ tol 即认定不动点——此时
    g 对自己诱导的上界最优，模型自洽。⚠️ 不能用固定轮数：实测 2025-09-10
    这类日子 2 轮时残差仍有 1.4×10⁻²，且续算到不动点使当日交付费用下降
    约 1300 元（2.9%）。序列非单调，不得在中间迭代里"挑最好的 Z̃"——不同
    轮次的可行域不同，只有不动点这一个自洽点。max_passes 命中即视为未收敛，
    如实报告（自检 C16）。

    Args:
        p: 逐时段电价，shape (144,)。
        scn: 净负荷情景，shape (N_s, 144)。
        probs: 情景概率。
        s0: 当日 0:00 储电量。
        nu_end / kappa: 末端估值 / 紧急倍率。
        emerg_cap: True 用新口径（带上界），False 退回 v5 无上界。
        tol: 不动点判据。
        max_passes: 轮数上限。
        p_scn: 情景电价 p^(s)（题4：一阶段折叠价 p 之外逐情景计价）；
            None 即常数价口径（题2/题3，全时域用 p）。

    Returns:
        (g, u_cap, resid, iters)：计划购电量；情景上界（旧口径为 None）；
        末轮残差；实际轮数（旧口径为 0.0 / 0）。
    """
    plan = solve_planning(p, scn, probs, s0, nu_end, kappa, p_scn=p_scn)
    if not emerg_cap:
        return plan.g, None, 0.0, 0
    resid, k = 0.0, 0
    for k in range(1, max_passes + 1):
        prev = plan.g
        plan = solve_planning(p, scn, probs, s0, nu_end, kappa,
                              emergency_cap(scn, prev), p_scn=p_scn)
        resid = float(np.abs(plan.g - prev).sum()
                      / max(float(plan.g.sum()), 1e-9))
        if resid <= tol:
            break
    return plan.g, emergency_cap(scn, plan.g), resid, k


def run_mpc_day(p: np.ndarray, net: np.ndarray, g: np.ndarray, s0: float,
                ref: np.ndarray, nu_end: float, kappa: float = KAPPA,
                emerg_cap: bool = True,
                p_fut: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                           np.ndarray]:
    """确定性等价 MPC 执行全天：未来压成单条参考路径（消融对照，§7.3）。

    当前时段用真值 n_t，未来时段用参考路径 ref。

    Args:
        p: 当步计价电价，shape (144,)（题4 读法 R 下为当日实现价）。
        net: 当日真实净负荷，shape (144,)。
        g: 已锁定的计划购电量，shape (144,)。
        s0: 当日 0:00 储电量。
        ref: α^rt 分位参考路径，shape (144,)。
        nu_end / kappa: 末端估值 / 紧急倍率。
        emerg_cap: True 给 u 加"只补缺口"上界（与交付臂同口径，保证执行
            层消融只差未来表示），False 退回 v5 无上界。
        p_fut: CE 对照臂的未来价格视图（题4：τ > t 段用情景均值折叠价，
            "把分布换成其均值"的确定性等价口径）；None 即全时域用 p。

    Returns:
        (c, d, u, w, S)：MPC 真实路径，S shape (145,)。
    """
    c = np.zeros(T)
    d = np.zeros(T)
    u = np.zeros(T)
    w = np.zeros(T)
    s = np.empty(T + 1)
    s[0] = s0
    for t in range(T):
        tail = np.concatenate([net[t:t + 1], ref[t + 1:]])
        cap = emergency_cap(tail, g[t:]) if emerg_cap else None
        p_t = (p[t:] if p_fut is None
               else np.concatenate([p[t:t + 1], p_fut[t + 1:]]))
        sol = solve_recourse(p_t, tail, g[t:], s[t], nu_end, kappa, cap)
        c_t, d_t = sol.c[0], sol.d[0]
        u[t], w[t] = sol.u[0], sol.w[0]
        if c_t > 0.0 and d_t > 0.0:
            # 性质 2 非等量缩减归一化：c,d 无直接成本，HiGHS 可能返回 c·d>0
            # 的等价最优顶点；按构造消除互斥，SOC 轨迹与费用严格不变。
            x = min(c_t, d_t / ETA**2)
            c_t -= x
            d_t -= ETA**2 * x
            w[t] += x * (1 - ETA**2)
        c[t], d[t] = c_t, d_t
        s[t + 1] = s[t] + ETA * c_t - d_t / ETA
    return c, d, u, w, s


def run_smpc_day(p: np.ndarray, net: np.ndarray, g: np.ndarray,
                 scn: np.ndarray, probs: np.ndarray, s0: float,
                 nu_end: float, kappa: float = KAPPA,
                 u_cap: np.ndarray | None = None,
                 p_scn: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray]:
    """在线随机 MPC 执行全天（交付，文档 §7.2）。

    逐时段解"当步共享 + 剩余时域按 0:00 情景分支"的两阶段 LP，
    只执行当步，SOC 按真实递推推进。与确定性等价 MPC（run_mpc_day）
    只差未来表示：情景分布 vs 单条分位路径——影子价格按缺口发生概率
    折算，而非按必然发生。

    Args:
        p: 当步计价电价，shape (144,)（题4 读法 R 下为当日实现价——
            交易时刻可见，用于当期紧急购电定价，题4 档 §7）。
        net: 当日真实净负荷，shape (144,)。
        g: 已锁定的计划购电量，shape (144,)。
        scn: 0:00 净负荷情景，shape (N_s, 144)。
        probs: 情景概率 π_s，shape (N_s,)。
        s0: 当日 0:00 储电量。
        nu_end / kappa: 末端估值 / 紧急倍率。
        u_cap: 情景紧急购电上界 (N_s, 144)，None 为 v5 无上界口径。
        p_scn: 0:00 情景电价 p^(s)（题4 未来分支用，τ > t）；None 即
            常数价口径（题2，全时域用 p）。

    Returns:
        (c, d, u, w, S)：随机 MPC 真实路径，S shape (145,)。
    """
    c = np.zeros(T)
    d = np.zeros(T)
    u = np.zeros(T)
    w = np.zeros(T)
    s = np.empty(T + 1)
    s[0] = s0
    for t in range(T):
        step = solve_stochastic_step(p, net, g, scn, probs, t, s[t],
                                     nu_end, kappa, u_cap, p_scn)
        c_t, d_t = step.c, step.d
        u[t], w[t] = step.u, step.w
        if c_t > 0.0 and d_t > 0.0:
            # 同 run_mpc_day：HiGHS 可能返回 c·d>0 的等价最优顶点，
            # 按构造归一化，SOC 轨迹与费用严格不变。
            x = min(c_t, d_t / ETA**2)
            c_t -= x
            d_t -= ETA**2 * x
            w[t] += x * (1 - ETA**2)
        c[t], d[t] = c_t, d_t
        s[t + 1] = s[t] + ETA * c_t - d_t / ETA
    return c, d, u, w, s


def rollout(year: YearData, p: np.ndarray, typ_ell: np.ndarray,
            typ_r: np.ndarray, cfg: ForecastConfig, nu_end: float = NU_END,
            kappa: float = KAPPA, alpha_rt: float = ALPHA_RT,
            policy: str = "smpc",
            win_schedule: tuple[np.ndarray, np.ndarray] | None = None,
            days: int | None = None,
            emerg_cap: bool = True) -> list[DayResult]:
    """全年诚实回测（文档 §9）：预报 → SAA 规划 → 执行层 → SOC 链推进。

    Args:
        year: 全年数据（按日解锁使用）。
        p: 逐时段电价（附件1，全年相同）。
        typ_ell / typ_r: 附件1 典型日曲线（暖机先验）。
        cfg: 预报层配置。
        nu_end: 末端估值——常数（缺省锚）或逐日数组（按日索引 j 取值，
            ν 锚实验 V2b 换装用）；kappa: 紧急倍率。
        alpha_rt: 确定性等价对照的参考路径分位（文档 §7.3）。
        policy: 交付执行层，"smpc" 在线随机 MPC（§7.2）/"mpc" 确定性
            等价（§7.3，消融口径）。无论交付为何，另一档执行层始终在
            同一 g、同日初 SOC 上加跑并记入 z / z_ce，保证机制归因干净。
        win_schedule: 逐日选窗 (w_l, w_r) 数组对（去超参实验，issue #6），
            第 j 日用 replace(cfg, load_lags=w_l[j], pv_window=w_r[j])；
            None = 固定 cfg 常数窗（交付口径）。
        days: 只滚动前 N 天（含暖机）；缺省全年 365。
        emerg_cap: 紧急购电口径。True（v6 默认，交付）= 加"只补缺口"
            上界 u ≤ (n−g)⁺（文档 §5.1，假设 A6′），四条臂（交付 / CE /
            事后 / PI）同口径；False = v5 旧口径（u 无上界），保留作
            "口径消融"对照，用于量化合规成本。

    Returns:
        逐日结果列表（长度 = min(days, 365)），暖机期也完整包含。

    Raises:
        ValueError: policy 不在 {"smpc", "mpc"}。
    """
    if policy not in {"smpc", "mpc"}:
        raise ValueError(f"未知执行策略: {policy}")
    rng = np.random.default_rng(cfg.seed)
    fcasts: list[tuple[np.ndarray, np.ndarray]] = []
    soc = S0_INIT
    out: list[DayResult] = []
    for j in range(days or len(year.dates)):
        # 逐日末端估值（V2b 换装）：数组按日取值，常数原样透传
        nu_j = (float(nu_end[j])
                if isinstance(nu_end, (np.ndarray, list, tuple)) else nu_end)
        # 逐日选窗（去超参实验）：按日替换 W_L/W_R，其余配置不动
        if win_schedule is not None:
            cfg = replace(cfg, load_lags=int(win_schedule[0][j]),
                          pv_window=int(win_schedule[1][j]))
        ell_hat = forecast_load(cfg, year, j, typ_ell)
        r_hat = forecast_pv(cfg, year, j, typ_r)
        scn, _ = build_scenarios(cfg, year, fcasts, j, ell_hat, r_hat, rng)
        probs = np.full(len(scn), 1.0 / len(scn))
        g, u_cap, resid, iters = solve_plan(p, scn, probs, soc, nu_j,
                                            kappa, emerg_cap)
        ref = reference_path(scn, alpha_rt)
        net = year.ell[j] - year.r[j]
        if policy == "smpc":
            c, d, u, w, s = run_smpc_day(p, net, g, scn, probs, soc,
                                         nu_j, kappa, u_cap)
            ce_u = run_mpc_day(p, net, g, soc, ref, nu_j, kappa,
                               emerg_cap)[2]
        else:
            c, d, u, w, s = run_mpc_day(p, net, g, soc, ref, nu_j,
                                        kappa, emerg_cap)
            ce_u = u
        cap_path = emergency_cap(net, g) if emerg_cap else None
        expost = solve_recourse(p, net, g, soc, nu_j, kappa, cap_path)
        pi_g, _, _, _ = solve_plan(p, net[None, :], np.ones(1), soc, nu_j,
                                   kappa, emerg_cap)
        pi_cap = emergency_cap(net, pi_g) if emerg_cap else None
        pi_u = solve_recourse(p, net, pi_g, soc, nu_j, kappa, pi_cap).u
        out.append(DayResult(
            j=j, g=g, ell_hat=ell_hat, r_hat=r_hat, ns=len(scn),
            c=c, d=d, u=u, w=w, S=s, z=cash_cost(p, g, u, kappa),
            z_ce=cash_cost(p, g, ce_u, kappa),
            z_expost=cash_cost(p, g, expost.u, kappa),
            z_pi=cash_cost(p, pi_g, pi_u, kappa), cap_resid=resid,
            cap_iters=iters))
        soc = s[T]
        fcasts.append((ell_hat, r_hat))
        if (j + 1) % 31 == 0:
            print(f"已回测 {j + 1}/{days or len(year.dates)} 天，"
                  f"年末 SOC = {soc:.1f} kWh", flush=True)
    return out
