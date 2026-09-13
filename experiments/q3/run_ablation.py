"""题 3 v7 计划器下 12 组消融网格（可断点续跑）。

F1/PB0 × LU0/LU1 × S-a/b/c；已有 q3_summary.md 的臂跳过。
产物 results/q3_v7-ablation/。不要用 uv run。
正式复现入口仍是：
  python -m nmmc.cli.prob3 --grid --jobs 4 --output-dir results/q3_v7-ablation
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd

from nmmc.cli.prob3 import GRID_AXES, _run_one
from nmmc.data import ATTACH
from nmmc.data.year import DAYS, load_year
from nmmc.model.q2 import NU_END
from nmmc.report.q2 import WARMUP_DAYS
from nmmc.report.q3 import TEMPLATE, AggregateCsv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "q3_v7-ablation"
JOBS = 4
EPOCHS = (0, 1, 2, 3)


def _label(fusion: str, lu: str, pool: str) -> str:
    return f"{fusion}-L1-{lu}-{pool}"


def _payload(fusion: str, lu: str, pool: str) -> dict:
    label = _label(fusion, lu, pool)
    return {
        "input": str(ATTACH / "附件2.xlsx"),
        "forecast": str(ATTACH / "附件3.xlsx"),
        "attach1": str(ATTACH / "附件1.xlsx"),
        "template": str(TEMPLATE),
        "load": "L1", "fusion": fusion, "load_update": lu, "pool": pool,
        "seed": 0, "nu_end": NU_END, "days": DAYS,
        "epochs": list(EPOCHS), "label": label, "emerg_cap": True,
        "out_dir": str(OUT / label),
    }


def _row_from_details(label: str, r_rep: float) -> dict:
    df = pd.read_csv(OUT / label / "q3_details.csv")
    z = float(df["z_online"].sum())
    dev = float(df["dev_fee"].sum())
    emerg = float(df["emerg_fee"].sum())
    w = float(df["w_sum"].sum())
    return {
        "label": label,
        "epochs": "0-6-12-18",
        "days": int(len(df)),
        "z_online": z,
        "z_ce": float(df["z_ce"].sum()),
        "smpc_gain": float(df["z_ce"].sum() - z),
        "z_expost": float(df["z_expost"].sum()),
        "z_pi": float(df["z_pi"].sum()),
        "evpi": float(z - df["z_pi"].sum()),
        "base_fee": float(df["base_fee"].sum()),
        "dev_fee": dev,
        "dev_share": round(dev / z, 6),
        "emerg_fee": emerg,
        "emerg_share": round(emerg / z, 6),
        "dp_sum": float(df["dp_sum"].sum()),
        "dm_sum": float(df["dm_sum"].sum()),
        "curtail_kwh": w,
        "curtail_rate": round(w / r_rep, 6),
        "soc_overnight": float(df["S0"].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = list(product(*GRID_AXES))
    todo = []
    for fusion, lu, pool in cells:
        label = _label(fusion, lu, pool)
        if (OUT / label / "q3_summary.md").exists():
            print(f"SKIP {label}", flush=True)
        else:
            todo.append(_payload(fusion, lu, pool))
    print(f"消融网格 todo {len(todo)}/{len(cells)}，jobs={JOBS}", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=JOBS) as pool:
            list(pool.map(_run_one, todo))
    year = load_year(ATTACH / "附件2.xlsx")
    r_rep = float(year.r[WARMUP_DAYS:].sum())
    rows = [_row_from_details(_label(*c), r_rep) for c in cells]
    rows.sort(key=lambda r: r["z_online"])
    ranked = [{"rank": i, **r} for i, r in enumerate(rows, 1)]
    path = AggregateCsv("ablation.csv", ranked).export(OUT)
    print(f"消融表: {path}", flush=True)
    for r in ranked:
        print(f"  {r['rank']:2d}  {r['label']:18s}  {r['z_online']:,.2f}",
              flush=True)


if __name__ == "__main__":
    main()
