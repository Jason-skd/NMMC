"""题2 笨办法对照：Z^nobat / Z^Q1 / Z^rule，同一预报情景与执行层口径。

文档 §11.1：
    Z^nobat  去掉电池——无耦合时报童分位 α 的逐时段报童（θ=0 ⇒ α=0.80）
    Z^Q1     题 1 典型日计划逐日复读 + 在线随机 MPC
    Z^rule   谷充峰放阈值规则（电价 35%/65% 分位）钉在预报日曲线上求解
             题 1 型 LP，再走同一套随机 MPC

用法：
    .venv/bin/python experiments/q2/baselines.py --kind nobat
    .venv/bin/python experiments/q2/baselines.py --kind nobat-scan
    .venv/bin/python experiments/q2/baselines.py --kind q1
    .venv/bin/python experiments/q2/baselines.py --kind rule
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nmmc.data.day import load_day1
from nmmc.data.year import load_year
from nmmc.model.forecast import (ForecastConfig, build_scenarios,
                                 forecast_load, forecast_pv)
from nmmc.model.lp import S_MAX, S_MIN, T
from nmmc.model.q1 import solve as solve_q1
from nmmc.model.q2 import KAPPA, NU_END, emergency_cap
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.run.q2 import S0_INIT, cash_cost, run_smpc_day

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/gap-checks"


def build_parser() -> argparse.ArgumentParser:
    """构建对照实验参数。"""
    ap = argparse.ArgumentParser(description="题2 笨办法全年对照")
    ap.add_argument("--kind",
                    choices=["nobat", "nobat-scan", "q1", "rule"],
                    default="nobat")
    ap.add_argument("--alpha", type=float, default=0.80,
                    help="无电池报童分位（θ=0 时 α*=(κ-1)/κ=0.80）")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return ap


def _cfg() -> ForecastConfig:
    return ForecastConfig(load="L1", pv="P1", resid="R2", seed=0)


def _report(rows: list[dict]) -> dict[str, float]:
    """报告期汇总。"""
    win = [r for r in rows if r["j"] >= WARMUP_DAYS]
    z = float(sum(r["z"] for r in win))
    u = float(sum(r["u"] for r in win))
    w = float(sum(r["w"] for r in win))
    g = float(sum(r["g"] for r in win))
    fee = float(sum(r["fee"] for r in win))
    return {
        "days": len(win), "z_online": z, "g_sum": g, "u_sum": u,
        "emerg_fee": fee, "w_sum": w,
        "emerg_share": fee / z if z else float("nan"),
    }


def run_nobat(year, p, typ_ell, typ_r, days: int, alpha: float) -> list[dict]:
    """无电池：情景分位定 g，缺口全走紧急购电。"""
    cfg = _cfg()
    rng = np.random.default_rng(cfg.seed)
    fcasts: list[tuple[np.ndarray, np.ndarray]] = []
    rows = []
    for j in range(days):
        ell_hat = forecast_load(cfg, year, j, typ_ell)
        r_hat = forecast_pv(cfg, year, j, typ_r)
        scn, _ = build_scenarios(cfg, year, fcasts, j, ell_hat, r_hat, rng)
        g = np.quantile(scn, alpha, axis=0)
        net = year.ell[j] - year.r[j]
        u = np.maximum(net - g, 0.0)
        w = np.maximum(g - net, 0.0)
        z = cash_cost(p, g, u, KAPPA)
        rows.append({"j": j, "z": z, "g": float(g.sum()), "u": float(u.sum()),
                     "w": float(w.sum()), "fee": KAPPA * float(p @ u)})
        fcasts.append((ell_hat, r_hat))
    return rows


def run_smpc_fixed_g(year, p, typ_ell, typ_r, days: int, make_g) -> list[dict]:
    """给定每日 g 的构造器，走交付口径随机 MPC（含 u 上界）。"""
    cfg = _cfg()
    rng = np.random.default_rng(cfg.seed)
    fcasts: list[tuple[np.ndarray, np.ndarray]] = []
    soc = S0_INIT
    rows = []
    for j in range(days):
        ell_hat = forecast_load(cfg, year, j, typ_ell)
        r_hat = forecast_pv(cfg, year, j, typ_r)
        scn, _ = build_scenarios(cfg, year, fcasts, j, ell_hat, r_hat, rng)
        probs = np.full(len(scn), 1.0 / len(scn))
        g = make_g(ell_hat, r_hat, soc)
        net = year.ell[j] - year.r[j]
        u_cap = emergency_cap(scn, g)
        _c, _d, u, w, s = run_smpc_day(p, net, g, scn, probs, soc,
                                       NU_END, KAPPA, u_cap)
        z = cash_cost(p, g, u, KAPPA)
        rows.append({"j": j, "z": z, "g": float(g.sum()), "u": float(u.sum()),
                     "w": float(w.sum()), "fee": KAPPA * float(p @ u),
                     "soc": float(s[1:].mean())})
        soc = s[T]
        fcasts.append((ell_hat, r_hat))
        if (j + 1) % 31 == 0:
            print(f"  {j + 1}/{days} 天  SOC={soc:.1f}", flush=True)
    return rows


def _write(kind: str, summary: dict, rows: list[dict], out_dir: Path,
           extra: dict | None = None) -> None:
    """落盘摘要行 + 逐日明细。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "alpha": float("nan"), **(extra or {}), **summary}
    path = out_dir / "q2_baselines.csv"
    df = pd.DataFrame([rec])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, float_format="%.6f")
    pd.DataFrame(rows).to_csv(out_dir / f"q2_{kind}_daily.csv", index=False,
                              float_format="%.6f")
    print(f"[{kind}] 报告期 {summary['days']} 天  "
          f"Z={summary['z_online']:,.2f} 元  "
          f"紧急费 {summary['emerg_fee']:,.2f} "
          f"({summary['emerg_share'] * 100:.2f}%)  → {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """跑一种对照并写入 results/gap-checks。"""
    args = build_parser().parse_args(argv)
    year = load_year()
    d1 = load_day1()
    p, typ_ell, typ_r = d1.p, d1.ell, d1.r
    out_dir = args.out_dir

    if args.kind == "nobat":
        rows = run_nobat(year, p, typ_ell, typ_r, args.days, args.alpha)
        _write(f"nobat-a{args.alpha:g}", _report(rows), rows, out_dir,
               extra={"alpha": args.alpha})
        return

    if args.kind == "nobat-scan":
        for alpha in (0.70, 0.80, 0.85, 0.90, 0.955, 1.00):
            print(f"无电池 α={alpha:g}", flush=True)
            rows = run_nobat(year, p, typ_ell, typ_r, args.days, alpha)
            _write(f"nobat-a{alpha:g}", _report(rows), rows, out_dir,
                   extra={"alpha": alpha})
        return

    if args.kind == "q1":
        q1 = solve_q1(p, typ_ell, typ_r)
        g_fixed = q1.g.copy()
        print("Z^Q1：典型日计划逐日复读 + 随机 MPC", flush=True)
        rows = run_smpc_fixed_g(year, p, typ_ell, typ_r, args.days,
                                lambda *_: g_fixed)
        _write("q1-replay", _report(rows), rows, out_dir)
        return

    low, high = (float(q) for q in np.quantile(p, (0.35, 0.65)))

    def make_rule(ell_hat, r_hat, soc):
        sol = solve_q1(p, ell_hat, r_hat,
                       s0_fixed=float(np.clip(soc, S_MIN, S_MAX)),
                       rule_thresholds=(low, high))
        return sol.g

    print(f"Z^rule：阈值 [{low:.4f}, {high:.4f}] + 随机 MPC", flush=True)
    rows = run_smpc_fixed_g(year, p, typ_ell, typ_r, args.days, make_rule)
    _write("rule", _report(rows), rows, out_dir)


if __name__ == "__main__":
    main()
