"""诊断②：把紧急购电拆成「囤电型 / 真缺口」并核验套利门槛。

链条位置：病因定位的第二步，也是 §7.6 归因五分类（③a/③b）的来源。
读 diag_slots.py 写出的 npz，按机制分类、检验 p_t/max p_剩余 < η²，
并打印样本内最差一天的完整时段轨迹。

用法：
    .venv/bin/python experiments/q2/diag_mechanism.py
    SLOTS=out/q2_slots_nocap.npz .venv/bin/python experiments/q2/diag_mechanism.py
"""
import os
import numpy as np

from nmmc.data.day import load_day1
from nmmc.model.lp import ETA, E_MAX_PER

z = np.load(os.environ.get("SLOTS", "out/q2_slots.npz"))
U, W, C, D, G, S, NET = (z[k] for k in ("U", "W", "C", "D", "G", "S", "NET"))
p = load_day1().p
n, T = U.shape
P = np.tile(p, (n, 1))
soc_prev = S[:, :-1]
mask = U > 1e-6
tot = U.sum()

print("=== %d 天，紧急电量 %.0f kWh，缺口时段 %d（%.2f%%）"
      % (n, tot, mask.sum(), 100 * mask.mean()))

# 机制分类：同时充电 / 同时弃电 / 净增储（SOC 上升）/ 真放电 / 电池空
chg = mask & (C > 1e-6)
dis = mask & (D > 1e-6)
up = mask & (S[:, 1:] > soc_prev + 1e-6)
empty = mask & (soc_prev < 2000)
full = mask & (soc_prev > 8000)
print("\n--- 机制分类（按紧急电量占比）")
for name, m in [("同时充电 c>0", chg), ("SOC 当步上升", up),
                ("同时放电 d>0", dis), ("电池近空 SOC<2000", empty),
                ("电池近满 SOC>8000", full)]:
    print("  %-22s %5.1f%%   时段 %4d   均价 %.4f"
          % (name, 100 * U[m].sum() / tot, m.sum(),
             (P[m] * U[m]).sum() / max(U[m].sum(), 1e-9)))

# 囤电判据：当步买入的电没有当步用掉（SOC 不降）且电池不空
hoard = mask & (S[:, 1:] >= soc_prev - 1e-6) & (soc_prev > 3000)
real = mask & ~hoard
print("\n--- 二分：囤电型（SOC 不降且非空）vs 真缺口")
for name, m in [("囤电型", hoard), ("真缺口", real)]:
    print("  %-8s 电量 %8.0f kWh（%5.1f%%）时段 %4d 均价 %.4f"
          % (name, U[m].sum(), 100 * U[m].sum() / tot, m.sum(),
             (P[m] * U[m]).sum() / max(U[m].sum(), 1e-9)))

# 囤电型是否满足套利条件：当前价 vs 当日剩余最高价
rem_max = np.array([[p[t:].max() for t in range(T)]] * 1)[0]
ratio = P / np.tile(rem_max, (n, 1))
m = hoard
print("  囤电型 p_t / max(p_剩余) 的电量加权均值 %.3f（<η²=0.81 即套利成立）"
      % ((ratio[m] * U[m]).sum() / U[m].sum()))
print("  其中比值 < 0.81 的电量占囤电型 %.1f%%"
      % (100 * U[m & (ratio < 0.81)].sum() / U[m].sum()))

# 真缺口的物理成因
m = real
print("\n--- 真缺口的成因（占真缺口电量）")
print("  电池空(SOC<1300)        %.1f%%" % (100 * U[m & (soc_prev < 1300)].sum()
                                            / U[m].sum()))
print("  放电顶功率上限          %.1f%%"
      % (100 * U[m & (D > E_MAX_PER - 1e-3)].sum() / U[m].sum()))
print("  缺口 n-g > D_max        %.1f%%"
      % (100 * U[m & ((NET - G) > E_MAX_PER)].sum() / U[m].sum()))

# 同一天里既弃电又紧急购电
day_u = U.sum(1)
day_w = W.sum(1)
both = (day_u > 1e-6) & (day_w > 1e-6)
print("\n--- 日级共存")
print("  既弃电又紧急购电的天数 %d / %d" % (both.sum(), n))
print("  这些天 平均弃电 %.0f kWh，平均紧急 %.0f kWh"
      % (day_w[both].mean(), day_u[both].mean()))
print("  全样本：Σw %.0f kWh 对应可回收成本≈%.0f 元；Σu %.0f kWh 花费 %.0f 元"
      % (W.sum(), (W * P).sum(), U.sum(), 5 * (U * P).sum()))

# SOC 上界/充电功率是否卡住中午
full_slot = S[:, 1:] > 10800 - 1e-3
cmax = C > E_MAX_PER - 1e-3
hour = np.tile(np.arange(T) // 6, (n, 1))
print("\n--- 中午（9–14 时）约束紧度")
mm = (hour >= 9) & (hour <= 14)
print("  SOC 贴上界占该段时段 %.1f%%；充电顶功率上限占 %.1f%%；弃电时段占 %.1f%%"
      % (100 * full_slot[mm].mean(), 100 * cmax[mm].mean(),
         100 * (W[mm] > 1e-6).mean()))
print("  该段弃电时 SOC 均值 %.0f kWh" % soc_prev[mm & (W > 1e-6)].mean())

# 典型一天
worst = int(np.argmax(day_u))
print("\n--- 紧急最多的一天（样本内第 %d 天，u=%.0f kWh）逐时段摘要"
      % (worst, day_u[worst]))
print("  t  hour   p      g      net     c      d      u      w     SOC")
for t in range(T):
    if U[worst, t] > 1e-6 or W[worst, t] > 500:
        print("  %3d %4d %.4f %7.1f %7.1f %6.1f %6.1f %6.1f %6.1f %7.0f"
              % (t, t // 6, p[t], G[worst, t], NET[worst, t], C[worst, t],
                 D[worst, t], U[worst, t], W[worst, t], S[worst, t]))
