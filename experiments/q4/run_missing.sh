#!/usr/bin/env bash
# 题 4 缺数批：4-3 修订阶梯三档 + 联合 vs 独立残差两线。
# 全开交付 PR1-V2-r 已在 results/q4/ 里，这里不重跑。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/results/q4/logs"
mkdir -p "$LOG"

stamp() { date "+%F %T"; }

run_one() {
  local name="$1"; shift
  echo "[$(stamp)] START $name" | tee -a "$LOG/batch-missing.txt"
  if PYTHONUNBUFFERED=1 "$PY" -u -m nmmc.cli.prob4 "$@" >"$LOG/${name}.log" 2>&1; then
    echo "[$(stamp)] DONE  $name" | tee -a "$LOG/batch-missing.txt"
  else
    echo "[$(stamp)] FAIL  $name (exit $?)" | tee -a "$LOG/batch-missing.txt"
    return 1
  fi
}

echo "[$(stamp)] queue start" | tee -a "$LOG/batch-missing.txt"

# 两路并行：阶梯一档 + 另一档，避免把 Air 打满。
run_one 4-3-PR1-V2-r-E0 \
  --line 4-3 --epochs 0 --output-dir results/q4 &
p1=$!
run_one 4-3-PR1-V2-r-E0-6 \
  --line 4-3 --epochs 0,6 --output-dir results/q4 &
p2=$!
wait "$p1" "$p2"

run_one 4-3-PR1-V2-r-E0-6-12 \
  --line 4-3 --epochs 0,6,12 --output-dir results/q4 &
p3=$!
run_one 4-2-PR1-V2-r-indep \
  --line 4-2 --price-resid indep --output-dir results/q4 &
p4=$!
wait "$p3" "$p4"

run_one 4-3-PR1-V2-r-indep \
  --line 4-3 --price-resid indep --output-dir results/q4

echo "[$(stamp)] queue done" | tee -a "$LOG/batch-missing.txt"
