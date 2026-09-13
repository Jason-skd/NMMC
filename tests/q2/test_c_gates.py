"""C1 / C2：题 2 规划层退化为题 1。

C1（判定性）：典型日、N_s=1、情景=该曲线、补回 S_144=S_0、去末端项
⇒ 精确复现题 1 最优值 35118.5986 元，u≡0。

C2：C1 设定下去掉循环约束、取 ν^end=ν*=0.472，强对偶 ⇒ 最优解与
题 1 一致或极接近（现金费用与 g 同量级）。
"""
from __future__ import annotations

import unittest

import numpy as np

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.model.q1 import solve as solve_q1
from nmmc.model.q2 import KAPPA, NU_END, emergency_cap, solve_planning, solve_recourse

Q1_COST = 35118.5986


class TestC1Q1Reproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.d1 = load_day1(ATTACH / "附件1.xlsx")
        cls.q1 = solve_q1(cls.d1.p, cls.d1.ell, cls.d1.r)
        cls.net = cls.d1.ell - cls.d1.r

    def test_q1_cost_anchor(self) -> None:
        self.assertAlmostEqual(self.q1.cost, Q1_COST, places=3)

    def test_ns1_cycle_recovers_q1(self) -> None:
        """C1：N_s=1 + 循环 + 无末端项 ⇒ 现金费用复现题 1，u≡0。"""
        s0 = float(self.q1.S[0])
        plan = solve_planning(self.d1.p, self.net[None, :], np.ones(1),
                              s0, 0.0, KAPPA, s_end=s0)
        self.assertAlmostEqual(float(self.d1.p @ plan.g), Q1_COST, places=3)
        rec = solve_recourse(self.d1.p, self.net, plan.g, s0, 0.0, KAPPA,
                             emergency_cap(self.net, plan.g))
        self.assertLess(float(rec.u.sum()), 1e-4)


class TestC2TerminalDual(unittest.TestCase):
    def test_nu_star_relaxes_cycle(self) -> None:
        """C2：循环换成 ν^end=0.472，规划目标 Ẑ 与题 1 松弛值同位，u≡0。"""
        d1 = load_day1(ATTACH / "附件1.xlsx")
        q1 = solve_q1(d1.p, d1.ell, d1.r)
        net = d1.ell - d1.r
        s0 = float(q1.S[0])
        plan = solve_planning(d1.p, net[None, :], np.ones(1), s0,
                              NU_END, KAPPA)
        rec = solve_recourse(d1.p, net, plan.g, s0, NU_END, KAPPA,
                             emergency_cap(net, plan.g))
        self.assertLess(float(rec.u.sum()), 1e-4)
        # 强对偶钉的是 Ẑ = Σp g − ν S_144，不是现金：S_0 钉死后求解器
        # 可沿 Ẑ 等值线抽干日末，现金低于题 1，但 Ẑ 与题 1 松弛值同位。
        self.assertAlmostEqual(plan.z_tilde, Q1_COST - NU_END * s0, places=3)


if __name__ == "__main__":
    unittest.main()
