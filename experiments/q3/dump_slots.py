"""题3 诊断：全年逐时段仪表化回测，落盘 npz 供离线分析。

复刻 nmmc.run.q3.rollout 的调用序（含 rng 消耗顺序，保证与交付臂逐位一致），
额外留存：逐阶段承诺 commit_m、阶段0 情景试探注入 a_scn（分位检验用）、
阶段0 情景净负荷分位、以及全部逐时段物理量。

用法：.venv/bin/python experiments/q3/dump_slots.py [--days 365]
"""
import argparse
from pathlib import Path

import numpy as np

from nmmc.data import ATTACH
from nmmc.data.day import load_day1
from nmmc.data.fcst import load_forecast
from nmmc.data.year import DAYS, load_year
from nmmc.model.forecast import ForecastConfig, forecast_load, reference_path
from nmmc.model.forecast3 import (Q3Config, build_scenarios, fusion_weights,
                                  hour_powers, load_point, pv_point)
from nmmc.model.lp import T
from nmmc.model.q2 import KAPPA, NU_END
from nmmc.model.q3 import settle_cost, solve_stage
from nmmc.run.q2 import ALPHA_RT, S0_INIT
from nmmc.run.q3 import run_smpc_segment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q3-diag/slots.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    d1 = load_day1(ATTACH / "附件1.xlsx")
    year = load_year(ATTACH / "附件2.xlsx")
    fc = load_forecast(ATTACH / "附件3.xlsx")
    cfg = Q3Config()
    p = d1.p
    rng = np.random.default_rng(cfg.seed)
    cfg2 = ForecastConfig(load=cfg.load, seed=cfg.seed)
    r_hour = hour_powers(year.r)
    hist: list[list[tuple[np.ndarray, np.ndarray]]] = []
    soc = S0_INIT

    n = args.days
    rec = {k: np.zeros((n, T)) for k in
           ("g", "a", "net", "c", "d", "u", "w", "ref0",
            "scn0_med", "scn0_lo", "scn0_hi", "ascn_med", "ascn_lo",
            "ascn_hi", "ell_hat0", "r_hat0")}
    rec["S"] = np.zeros((n, T + 1))
    rec["commit"] = np.zeros((n, 4, T))
    rec["z"] = np.zeros(n)
    rec["ns"] = np.zeros((n, 4), dtype=int)

    for j in range(n):
        weights = fusion_weights(r_hour, fc.phi, j)
        ell0 = forecast_load(cfg2, year, j, typ_ell := d1.ell)
        net = year.ell[j] - year.r[j]
        g = None
        committed = None
        c, d, u, w = (np.zeros(T) for _ in range(4))
        s = np.zeros(T + 1)
        s[0] = soc
        stage_fcasts: list[tuple[np.ndarray, np.ndarray]] = []
        for m in range(4):
            h0, h1 = 36 * m, 36 * (m + 1)
            r_hat = pv_point(cfg, year, fc, j, m, d1.r, weights)
            ell_hat = ell0 if m == 0 else load_point(cfg, ell0, year, j, m)
            stage_fcasts.append((ell_hat, r_hat))
            scn, _ = build_scenarios(cfg, year, hist, j, m, ell_hat, r_hat, rng)
            probs = np.full(len(scn), 1.0 / len(scn))
            rec["ns"][j, m] = len(scn)
            res = solve_stage(p, g, committed, scn, probs, m, s[h0],
                              NU_END, KAPPA)
            if m == 0:
                g = res.commit
                rec["scn0_med"][j] = np.median(scn, axis=0)
                rec["scn0_lo"][j] = np.percentile(scn, 10, axis=0)
                rec["scn0_hi"][j] = np.percentile(scn, 90, axis=0)
                rec["ascn_med"][j] = np.median(res.a_scn, axis=0)
                rec["ascn_lo"][j] = np.percentile(res.a_scn, 10, axis=0)
                rec["ascn_hi"][j] = np.percentile(res.a_scn, 90, axis=0)
                rec["ref0"][j] = reference_path(scn, ALPHA_RT)
                rec["ell_hat0"][j] = ell_hat
                rec["r_hat0"][j] = r_hat
            committed = res.commit
            rec["commit"][j, m] = committed
            run_smpc_segment(p, net, committed, g, scn, probs, c, d, u, w,
                             s, h0, h1, NU_END, KAPPA, free_from=h1)
        rec["g"][j], rec["a"][j], rec["net"][j] = g, committed, net
        rec["c"][j], rec["d"][j], rec["u"][j], rec["w"][j] = c, d, u, w
        rec["S"][j] = s
        rec["z"][j] = settle_cost(p, g, committed, u, KAPPA)
        soc = s[T]
        hist.append(stage_fcasts)
        if (j + 1) % 31 == 0:
            print(f"  {j + 1}/{n} 天，SOC={soc:.0f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, p=p, dates=year.dates[:n].astype(str),
                        **rec)
    print(f"已落盘 {args.out}；Z(报告期 31:) = "
          f"{rec['z'][31:n].sum():,.2f} 元")


if __name__ == "__main__":
    main()
