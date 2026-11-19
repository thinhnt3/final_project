#!/bin/bash
# Training queue: can AdaMF be made to read relation edges? Baseline adamf-d256-l2.
# hn4 = hard-negative groups of 4 images sharing an object pair; so0.3 = 30% structure-only rows.
cd "$(dirname "$0")/../.."
PY="/Users/lap17650/workspace/codebase/final-project /.venv/bin/python"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
COMMON="--model adamf_mat --no-mat --normalize-modalities --wandb-mode offline --resume"
train_and_score() {  # $1 = run name, rest = extra train args
  local name=$1; shift
  "$PY" -m scripts.train_graphsage $COMMON "$@" && "$PY" -m scripts.run_benchmark --method adamf_mat --run "$name"
  echo "DONE $name exit=$?"
}
queue_a() {
  train_and_score adamf-d256-l2-hn4 --hard-negative-group 4
  train_and_score adamf-d256-l2-so0.3-hn4 --hard-negative-group 4 --structure-only-rate 0.3
}
queue_b() {
  train_and_score adamf-d256-l2-so0.3 --structure-only-rate 0.3
}
queue_a > runs/adamf_mat/relation_queue_a.log 2>&1 &
queue_b > runs/adamf_mat/relation_queue_b.log 2>&1 &
wait
echo ALL_DONE
