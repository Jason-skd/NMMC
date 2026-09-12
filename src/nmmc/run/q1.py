"""题1 流程执行器：主 LP 求解、Z^6000/Z^rule 对照与 S_0 敏感性扫描。"""
from dataclasses import dataclass, field

import numpy as np

from nmmc.data.day import DayData
from nmmc.model.lp import S_MAX, S_MIN
from nmmc.model.q1 import Solution, solve


@dataclass
class Q1Outcome:
    """题1 全流程结果。

    Attributes:
        sol: 主 LP 最优解。
        baselines: 对照基准（Z^6000、S_0 决策化收益、Z^rule、规则节省
            比例）；s0_fixed 模式为空 dict。
        s0_probes: S_0 平坦性探测 (钉值, 费用)，最优值 ±200 kWh 两点；
            s0_fixed 模式为 None。
        s0_grid: S_0 敏感性扫描网格（最优点不在 200 网格上时注入），
            kWh；s0_fixed 模式为 None。
        s0_costs: 扫描网格对应的费用，元；与 s0_grid 同缺省。
    """

    sol: Solution
    baselines: dict[str, float] = field(default_factory=dict)
    s0_probes: list[tuple[float, float]] | None = None
    s0_grid: np.ndarray | None = None
    s0_costs: list[float] | None = None


def run_q1(day: DayData, s0_fixed: float | None = None) -> Q1Outcome:
    """运行题1 全流程：主 LP → 对照基准 → S_0 探测与敏感性扫描。

    Args:
        day: 附件1 典型日数据。
        s0_fixed: 钉住 S_0 的取值；None 表示 S_0 自由决策（此时追加
            对照与扫描）。

    Returns:
        Q1Outcome: 最优解与对照/扫描结果。
    """
    p, ell, r = day.p, day.ell, day.r
    sol = solve(p, ell, r, s0_fixed=s0_fixed)
    if s0_fixed is not None:
        return Q1Outcome(sol=sol)

    sol6000 = solve(p, ell, r, s0_fixed=6000.0)
    low, high = (float(q) for q in np.quantile(p, (0.35, 0.65)))
    rule = solve(p, ell, r, rule_thresholds=(low, high))
    baselines = {
        "Z^6000": sol6000.cost,
        "S_0 决策化收益": sol6000.cost - sol.cost,
        "Z^rule": rule.cost,
        "规则节省比例": (rule.cost - sol.cost) / rule.cost * 100.0,
    }
    s0_probes = [(float(sol.S[0] + delta),
                  solve(p, ell, r, s0_fixed=float(sol.S[0] + delta)).cost)
                 for delta in (-200.0, 200.0)]
    s0_grid = np.arange(S_MIN, S_MAX + 1.0, 200.0)
    if abs((sol.S[0] - S_MIN) % 200.0) > 1e-6:
        s0_grid = np.sort(np.append(s0_grid, sol.S[0]))
    s0_costs = [solve(p, ell, r, s0_fixed=float(v)).cost for v in s0_grid]
    return Q1Outcome(sol=sol, baselines=baselines, s0_probes=s0_probes,
                     s0_grid=s0_grid, s0_costs=s0_costs)
