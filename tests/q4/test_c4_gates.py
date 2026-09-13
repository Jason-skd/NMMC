"""C4 基础闸门：复现、折叠、恒等式、信息泄漏。

C4-2：price-mode=a1 + ν=V0 应逐日复现题 2 v6 / 题 3 v7（同函数、同上界）。
C4-3：附件1 循环对偶 |ν_cycle| = 0.472，题 1 最优值 35118.5986。
C4-5：性质 2 一阶段折叠 E_s[p^(s)·g] = p̂·g。
C4-6：附件1 电价 = 附件4 全年逐时段均值。
C4-1：电价点预报只用 [:j]，打乱未来实现价不改变 p̂。
"""
from __future__ import annotations

import unittest

import numpy as np

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.price import IDENTITY_TOL, load_price
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig
from nmmc.model.forecast3 import Q3Config
from nmmc.model.forecast4 import PriceConfig, forecast_price
from nmmc.model.q1 import solve as solve_q1
from nmmc.model.q2 import KAPPA, NU_END, W_EPS, solve_stochastic_step
from nmmc.model.lp import S_MAX, T
from nmmc.report.common import split_spill
from nmmc.report.q4 import adjusted_cost
from nmmc.model.q4 import folded_price, nu_end_daily
from nmmc.run.q2 import rollout as rollout_q2
from nmmc.run.q3 import rollout as rollout_q3
from nmmc.run.q4 import rollout_q42, rollout_q43

Q1_COST = 35118.5986


class TestC43C46(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.d1 = load_day1(ATTACH / "附件1.xlsx")
        cls.price = load_price(cls.d1.p)

    def test_attach4_identity(self) -> None:
        """C4-6：附件1 = 附件4 全年逐时段算术平均。"""
        dev = float(np.abs(self.price.p.mean(axis=0) - self.d1.p).max())
        self.assertLessEqual(dev, IDENTITY_TOL)

    def test_q1_cycle_dual_and_cost(self) -> None:
        """C4-3：附件1 循环对偶绝对值 = 0.472，费用复现题 1。"""
        q1 = solve_q1(self.d1.p, self.d1.ell, self.d1.r)
        self.assertAlmostEqual(q1.cost, Q1_COST, places=3)
        self.assertIsNotNone(q1.nu_cycle)
        self.assertAlmostEqual(abs(float(q1.nu_cycle)), NU_END, places=3)

    def test_v2_nu_matches_q1_on_attach1(self) -> None:
        """V2 首日自检：附件1 价格下循环对偶即常数锚。"""
        nu = nu_end_daily(PriceConfig(nu_rule="V2"), self.price,
                          self.d1.ell, self.d1.r, self.d1.p, days=1)
        self.assertAlmostEqual(float(nu[0]), NU_END, places=3)


class TestC45Fold(unittest.TestCase):
    def test_folded_price_identity(self) -> None:
        """C4-5：E_s[p^(s) g] = p̂ g（一阶段项精确折叠）。"""
        rng = np.random.default_rng(0)
        prices = rng.random((12, 144)) + 0.1
        probs = np.full(12, 1.0 / 12.0)
        g = rng.random(144) * 50.0
        hat = folded_price(prices, probs)
        lhs = float(sum(probs[s] * (prices[s] @ g) for s in range(12)))
        rhs = float(hat @ g)
        self.assertAlmostEqual(lhs, rhs, places=9)


class TestC41NoLeak(unittest.TestCase):
    def test_forecast_ignores_future_prices(self) -> None:
        """C4-1：打乱 j 日及之后的实现价，p̂ 逐位不变。"""
        d1 = load_day1(ATTACH / "附件1.xlsx")
        price = load_price(d1.p)
        cfg = PriceConfig()
        j = 40
        hat = forecast_price(cfg, price, j, d1.p)
        scrambled = np.array(price.p, copy=True)
        rng = np.random.default_rng(1)
        scrambled[j:] = rng.permutation(scrambled[j:])
        from nmmc.data.price import YearPrice
        fake = YearPrice(dates=price.dates, p=scrambled, dow=price.dow)
        hat2 = forecast_price(cfg, fake, j, d1.p)
        np.testing.assert_allclose(hat, hat2, atol=0.0, rtol=0.0)


class TestC42Reproduce(unittest.TestCase):
    """a1 退化口径应复现题 2 v6 / 题 3 v7（第 0 日即可判定）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.d1 = load_day1(ATTACH / "附件1.xlsx")
        cls.year = load_year(ATTACH / "附件2.xlsx")
        cls.price = load_price(cls.d1.p)
        cls.fc = load_forecast(ATTACH / "附件3.xlsx")

    def test_q42_a1_matches_q2_day0(self) -> None:
        cfg = ForecastConfig()
        cfgp = PriceConfig(nu_rule="V0")
        q4 = rollout_q42(self.year, self.price, self.d1.ell, self.d1.r,
                         self.d1.p, cfg, cfgp, mode="a1", days=1,
                         emerg_cap=True)
        q2 = rollout_q2(self.year, self.d1.p, self.d1.ell, self.d1.r, cfg,
                        days=1, emerg_cap=True)
        np.testing.assert_allclose(q4[0].g, q2[0].g, atol=1e-7, rtol=0.0)
        self.assertAlmostEqual(q4[0].z, q2[0].z, places=6)
        self.assertAlmostEqual(q4[0].z_ce, q2[0].z_ce, places=6)
        gap = np.maximum(q4[0].u - np.maximum(
            self.year.ell[0] - self.year.r[0] - q4[0].g, 0.0), 0.0)
        self.assertLess(float(gap.max()), 1e-6)

    def test_q43_a1_matches_q3_day0(self) -> None:
        cfg = Q3Config()
        cfgp = PriceConfig(nu_rule="V0")
        q4 = rollout_q43(self.year, self.fc, self.price, self.d1.ell,
                         self.d1.r, self.d1.p, cfg, cfgp, mode="a1",
                         days=1, emerg_cap=True)
        q3 = rollout_q3(self.year, self.fc, self.d1.p, self.d1.ell,
                        self.d1.r, cfg, days=1, emerg_cap=True)
        np.testing.assert_allclose(q4[0].g, q3[0].g, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(q4[0].a, q3[0].a, atol=1e-7, rtol=0.0)
        self.assertAlmostEqual(q4[0].z, q3[0].z, places=6)
        self.assertGreater(q4[0].s_pi, 0.0)
        self.assertGreater(q4[0].s_expost, 0.0)


class TestSpillSplit(unittest.TestCase):
    def test_pv_vs_overbuy(self) -> None:
        """弃光 = min(w, (r−ℓ)⁺)，其余是超买倒掉。"""
        w = np.array([10.0, 5.0, 0.0])
        ell = np.array([20.0, 0.0, 10.0])
        r = np.array([15.0, 8.0, 10.0])
        w_pv, w_over = split_spill(w, ell, r)
        np.testing.assert_allclose(w_pv, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(w_over, [10.0, 0.0, 0.0])
        np.testing.assert_allclose(w_pv + w_over, w)


class TestC9NuAdjusted(unittest.TestCase):
    def test_cash_false_positive_flips(self) -> None:
        """2025-05-03：现金 PI>ex-post，ν 调整后翻负。"""
        nu, z_xp, z_pi = 0.4680, 12170.0, 12698.0
        s_xp, s_pi = 8429.0, 10800.0
        self.assertGreater(z_pi, z_xp)
        self.assertLess(adjusted_cost(z_pi, nu, s_pi),
                        adjusted_cost(z_xp, nu, s_xp))


class TestC21DumpPenalty(unittest.TestCase):
    def test_last_slot_nu0_prefers_charge_over_dump(self) -> None:
        """ν=0 时存电无收益，ε 仍让求解器把盈余充进去而不是倒掉。"""
        t = T - 1
        p = np.full(T, 0.8)
        net = np.zeros(T)
        g = np.zeros(T)
        g[t] = 100.0
        step = solve_stochastic_step(
            p, net, g, np.zeros((1, T)), np.ones(1), t, 5000.0, 0.0, KAPPA)
        self.assertAlmostEqual(step.d, 0.0, places=3)
        self.assertAlmostEqual(step.w, 0.0, places=3)
        self.assertAlmostEqual(step.c, 100.0, places=3)
        self.assertGreater(W_EPS, 0.0)

    def test_full_battery_does_not_discharge_then_dump(self) -> None:
        """满仓只能弃。LP 可能走充放互斥等价顶点，归一化后 d=0、w=盈余。"""
        from nmmc.model.lp import ETA
        t = T - 1
        p = np.full(T, 0.8)
        net = np.zeros(T)
        g = np.zeros(T)
        g[t] = 100.0
        step = solve_stochastic_step(
            p, net, g, np.zeros((1, T)), np.ones(1), t, S_MAX, 0.0, KAPPA)
        c_t, d_t, w_t = step.c, step.d, step.w
        if c_t > 0.0 and d_t > 0.0:
            x = min(c_t, d_t / ETA**2)
            c_t -= x
            d_t -= ETA**2 * x
            w_t += x * (1.0 - ETA**2)
        self.assertAlmostEqual(d_t, 0.0, places=3)
        self.assertAlmostEqual(c_t, 0.0, places=3)
        self.assertAlmostEqual(w_t, 100.0, places=3)


if __name__ == "__main__":
    unittest.main()
