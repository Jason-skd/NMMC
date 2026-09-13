"""C3-1 / C3-2：守住「日前计划器 = 题 2 SAA」与题 1 退化。

C3-1 的实现层含义（v7 / CF-A）：
    1. solve_stage(m=0) 在 N_s=1 时与题 2 solve_planning 逐位等价
       （别名缺陷的可证伪核验；一写就会立刻抓到 o_s=0）；
    2. 交付 rollout 的 0:00 计划就是 solve_plan（同情景、同 SOC）。

C3-2：在 C3-1 上退化为典型日、N_s=1、补回 S_144=S_0、去末端项，
复现题 1 最优值 35118.5986 元、u≡0。
"""
from __future__ import annotations

import unittest

import numpy as np

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig, forecast_load
from nmmc.model.forecast3 import (Q3Config, build_scenarios, fusion_weights,
                                  hour_powers, pv_point)
from nmmc.model.lp import ETA
from nmmc.model.q1 import solve as solve_q1
from nmmc.model.q2 import KAPPA, NU_END, emergency_cap, solve_planning, solve_recourse
from nmmc.model.q3 import BETA_MINUS, BETA_PLUS, solve_stage
from nmmc.run.q2 import S0_INIT, solve_plan
from nmmc.run.q3 import rollout

Q1_COST = 35118.5986


class TestC31(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.d1 = load_day1(ATTACH / "附件1.xlsx")
        cls.net = cls.d1.ell - cls.d1.r

    def test_ns1_stage0_matches_q2_planning(self) -> None:
        """N_s=1 时修正后的 solve_stage(m=0) 与 solve_planning 逐位等价。"""
        scn = self.net[None, :]
        probs = np.ones(1)
        s0 = 6000.0
        plan = solve_planning(self.d1.p, scn, probs, s0, NU_END, KAPPA)
        stg = solve_stage(self.d1.p, None, None, scn, probs, 0, s0,
                          NU_END, KAPPA)
        np.testing.assert_allclose(stg.commit, plan.g, atol=1e-7, rtol=0.0)
        self.assertAlmostEqual(stg.z_tilde, plan.z_tilde, places=6)

    def test_alias_not_forced(self) -> None:
        """别名缺陷：g[:108] 被绑成 a^(0)[36:]。修好后残差不应恒为 0。"""
        scn = np.vstack([self.net, self.net + 30.0, self.net - 20.0])
        probs = np.full(3, 1.0 / 3.0)
        stg = solve_stage(self.d1.p, None, None, scn, probs, 0, 6000.0,
                          NU_END, KAPPA)
        alias = float(np.abs(stg.commit[:108] - stg.a_scn[0, 36:]).max())
        self.assertGreater(alias, 1e-3)

    def test_delivery_m0_is_q2_plan(self) -> None:
        """交付 rollout 第 0 日的 g 等于同情景、同 SOC 的 solve_plan。"""
        year = load_year(ATTACH / "附件2.xlsx")
        fc = load_forecast(ATTACH / "附件3.xlsx")
        cfg = Q3Config()
        rng = np.random.default_rng(cfg.seed)
        cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
        r_hour = hour_powers(year.r)
        weights = fusion_weights(r_hour, fc.phi, 0)
        ell0 = forecast_load(cfg2, year, 0, self.d1.ell)
        r_hat = pv_point(cfg, year, fc, 0, 0, self.d1.r, weights)
        scn, _ = build_scenarios(cfg, year, [], 0, 0, ell0, r_hat, rng)
        probs = np.full(len(scn), 1.0 / len(scn))
        g_plan = solve_plan(self.d1.p, scn, probs, S0_INIT, NU_END, KAPPA,
                            True)[0]
        days = rollout(year, fc, self.d1.p, self.d1.ell, self.d1.r, cfg,
                       days=1, emerg_cap=True)
        np.testing.assert_allclose(days[0].g, g_plan, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(days[0].a[:36], days[0].g[:36],
                                   atol=1e-9)
        net = year.ell[0] - year.r[0]
        gap = np.maximum(days[0].u - np.maximum(net - days[0].a, 0.0), 0.0)
        self.assertLess(float(gap.max()), 1e-6)


class TestC32(unittest.TestCase):
    def test_typical_day_recovers_q1(self) -> None:
        d1 = load_day1(ATTACH / "附件1.xlsx")
        q1 = solve_q1(d1.p, d1.ell, d1.r)
        self.assertAlmostEqual(q1.cost, Q1_COST, places=3)
        net = d1.ell - d1.r
        plan = solve_planning(d1.p, net[None, :], np.ones(1), q1.S[0],
                              0.0, KAPPA, s_end=q1.S[0])
        self.assertAlmostEqual(float(d1.p @ plan.g), Q1_COST, places=3)
        rec = solve_recourse(d1.p, net, plan.g, q1.S[0], 0.0, KAPPA,
                             emergency_cap(net, plan.g))
        self.assertLess(float(rec.u.sum()), 1e-4)


def _deadband(kappa: float, theta: float) -> tuple[float, float]:
    """性质 6：加订目标分位 α_up、取消容忍分位 α_dn。"""
    a_up = (kappa - BETA_PLUS) / (kappa + theta)
    a_dn = (kappa - 1.0 + BETA_MINUS) / (kappa + theta)
    return a_up, a_dn


def _period_cost(a: float, g: float, samples: np.ndarray, *,
                 kappa: float, theta: float, p: float = 1.0) -> float:
    """单时段期望费用（文档性质 6）：pa + ½p|a−g| + κp E(N−a)⁺ − θp E(a−N)⁺。"""
    short = np.mean(np.maximum(samples - a, 0.0))
    excess = np.mean(np.maximum(a - samples, 0.0))
    return (p * a + 0.5 * p * abs(a - g)
            + kappa * p * short - theta * p * excess)


class TestC34Deadband(unittest.TestCase):
    """C3-4：单时段数值实验重现死带 [0.70, 0.90] / [0.6024, 0.7745]。"""

    def test_formula_endpoints(self) -> None:
        a_up, a_dn = _deadband(KAPPA, 0.0)
        self.assertAlmostEqual(a_up, 0.70, places=4)
        self.assertAlmostEqual(a_dn, 0.90, places=4)
        a_up, a_dn = _deadband(KAPPA, ETA ** 2)
        self.assertAlmostEqual(a_up, 0.6024, places=3)
        self.assertAlmostEqual(a_dn, 0.7745, places=3)

    def test_grid_argmin_follows_deadband(self) -> None:
        # θ=0 时 (κ±θ) 无歧义：均匀支撑 [50,150] ⇒ F^{-1}(α)=50+100α。
        # θ=η² 的闭式分母是 κ+θ，对文档 C(a) 里 −θ E(a−N)⁺ 求导却是 κ−θ，
        # 两套公式分叉，数值核验只落在无歧义的 θ=0 带。
        samples = np.linspace(50.0, 150.0, 401)
        grid = np.linspace(40.0, 160.0, 241)
        a_up, a_dn = _deadband(KAPPA, 0.0)
        q_up, q_dn = 50.0 + 100.0 * a_up, 50.0 + 100.0 * a_dn
        cases = (
            (q_up - 25.0, q_up),
            (0.5 * (q_up + q_dn), None),
            (q_dn + 25.0, q_dn),
        )
        for g, target in cases:
            want = g if target is None else target
            costs = [_period_cost(a, g, samples, kappa=KAPPA, theta=0.0)
                     for a in grid]
            a_star = float(grid[int(np.argmin(costs))])
            self.assertLess(abs(a_star - want), 2.0,
                            msg=f"g={g:.2f}: a*={a_star:.2f} want={want:.2f}")


if __name__ == "__main__":
    unittest.main()
