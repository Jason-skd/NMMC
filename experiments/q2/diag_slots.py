"""诊断①：题 2 逐时段紧急购电成因 + SOC 水平对比（逐时段重放）。

链条位置：病因定位的第一步。把 rollout 的真实路径按时段摊开，回答"紧急
购电发生在什么水位、当步在充还是在放、是否顶到功率上限、集中在哪些钟点"。
输出 out/q2_slots.npz 供 diag_mechanism.py 做机制分类。

用法：
    .venv/bin/python experiments/q2/diag_slots.py            # 新口径（v6 默认）
    DAYS=100 .venv/bin/python experiments/q2/diag_slots.py   # 改天数
    EMERG_CAP=0 .venv/bin/python experiments/q2/diag_slots.py  # v5 无上界口径
约 100 天 / 3.5 分钟（新口径更慢，规划层要迭代到不动点）。
"""
import os
import numpy as np

from nmmc.data.day import load_day1
from nmmc.data.year import load_year
from nmmc.model.forecast import ForecastConfig
from nmmc.model.lp import ETA, S_MAX, S_MIN, E_MAX_PER
from nmmc.run.q2 import rollout

DAYS = int(os.environ.get("DAYS", 100))
EMERG_CAP = os.environ.get("EMERG_CAP", "1") != "0"
OUT = "out/q2_slots.npz" if EMERG_CAP else "out/q2_slots_nocap.npz"
year = load_year()
d1 = load_day1()
cfg = ForecastConfig(load="L1", pv="P1", resid="R2", seed=0)
res = rollout(year, d1.p, d1.ell, d1.r, cfg, days=DAYS,
              emerg_cap=EMERG_CAP)

p = d1.p
rep = [r for r in res if r.j >= 31]          # 跳过暖机
U = np.array([r.u for r in rep])
W = np.array([r.w for r in rep])
C = np.array([r.c for r in rep])
D = np.array([r.d for r in rep])
G = np.array([r.g for r in rep])
S = np.array([r.S for r in rep])              # (n,145)
NET = np.array([year.ell[r.j] - year.r[r.j] for r in rep])
n = len(rep)
print(f"\n=== 样本 {n} 天（day 32..{rep[-1].j + 1}）")

mask = U > 1e-6
print("缺口时段频率 %.2f%%   紧急电量 %.0f kWh" % (100 * mask.mean(), U.sum()))
print("SOC 日内均值 %.1f   S0 均值 %.1f   S144 均值 %.1f"
      % (S[:, 1:].mean(), S[:, 0].mean(), S[:, 144].mean()))
print("弃电 %.0f kWh（率 %.2f%%）"
      % (W.sum(), 100 * W.sum() / np.array([year.r[r.j] for r in rep]).sum()))

soc_prev = S[:, :-1]
print("\n--- 紧急购电时的 SOC 分布（按电量加权）")
for lo, hi in [(1200, 2000), (2000, 3000), (3000, 5000), (5000, 8000),
               (8000, 10801)]:
    m = mask & (soc_prev >= lo) & (soc_prev < hi)
    print("  SOC[%5d,%5d) 电量占比 %5.1f%%  时段数 %4d"
          % (lo, hi, 100 * U[m].sum() / U.sum(), m.sum()))

print("\n--- 紧急购电时放电是否已顶到功率上限")
m = mask
d_at = D[m]
print("  放电达上限(>%.1f) 的时段占 %.1f%%，放电=0 的占 %.1f%%"
      % (E_MAX_PER - 1e-6, 100 * (d_at > E_MAX_PER - 1e-3).mean(),
         100 * (d_at < 1e-6).mean()))
print("  按电量：放电顶上限 %.1f%%，放电为 0 %.1f%%"
      % (100 * U[m][d_at > E_MAX_PER - 1e-3].sum() / U.sum(),
         100 * U[m][d_at < 1e-6].sum() / U.sum()))
print("  同时段 g 为 0 的电量占比 %.1f%%" % (100 * U[m][G[m] < 1e-6].sum()
                                            / U.sum()))
print("  缺口幅度 n-g > D_max 的电量占比 %.1f%%"
      % (100 * U[m][(NET[m] - G[m]) > E_MAX_PER].sum() / U.sum()))

print("\n--- 紧急购电按小时（电量占比）与该小时均价")
hour = np.tile(np.arange(144) // 6, (n, 1))
for h in range(24):
    mh = mask & (hour == h)
    q = 100 * U[mh].sum() / U.sum()
    if q > 0.5:
        print("  %2d:00  %5.1f%%   均价 %.4f   时段数 %4d"
              % (h, q, p[np.arange(144) // 6 == h].mean(), mh.sum()))

print("\n--- 计划 g 与真值/情景的对比")
print("  Σg %.0f  Σnet %.0f  Σu %.0f  Σw %.0f"
      % (G.sum(), NET.sum(), U.sum(), W.sum()))
print("  日均 g - net = %.0f kWh（正=计划买多了）"
      % ((G.sum() - NET.sum()) / n))
loss = ((1 - ETA) * C + (1 / ETA - 1) * D).sum()
print("  往返损耗合计 %.0f kWh；SOC 净变化 %.0f"
      % (loss, S[-1, 144] - S[0, 0]))

print("\n--- SOC 触边统计（逐时段占比）")
print("  贴上界 %.2f%%   贴下界 %.2f%%   低于 3000 的时段 %.2f%%"
      % (100 * (S[:, 1:] > S_MAX - 1e-3).mean(),
         100 * (S[:, 1:] < S_MIN + 1e-3).mean(),
         100 * (S[:, 1:] < 3000).mean()))

print("\n--- 弃电按小时（占比）")
for h in range(24):
    mh = (W > 1e-6) & (hour == h)
    q = 100 * W[mh].sum() / W.sum()
    if q > 1:
        print("  %2d:00  %5.1f%%" % (h, q))

print("\n--- u>0 且 c>0 的时段")
mc = mask & (C > 1e-6)
print("  时段数 %d，电量 %.0f kWh，占紧急 %.1f%%，均价 %.4f"
      % (mc.sum(), U[mc].sum(), 100 * U[mc].sum() / U.sum(),
         np.tile(p, (n, 1))[mc].mean() if mc.sum() else 0))

os.makedirs("out", exist_ok=True)
np.savez(OUT, U=U, W=W, C=C, D=D, G=G, S=S, NET=NET,
         days=np.array([r.j for r in rep]))
print(f"\nsaved {OUT}")
