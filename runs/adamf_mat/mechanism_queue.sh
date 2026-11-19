#!/bin/bash
# Training queue for the AdaMF-MAT mechanism experiments (E1, E4, E7 and the 2x2 fusion/MAT table).
# Two queues in parallel, five CPU threads each; each config trains 3 seeds then is benchmarked.
cd "$(dirname "$0")/../.."
PY="/Users/lap17650/workspace/codebase/final-project /.venv/bin/python"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
train_and_score() {  # $1 = method, $2 = run name, rest = train args
  local method=$1 name=$2; shift 2
  "$PY" -m scripts.train_graphsage --model "$method" --wandb-mode offline --resume "$@" \
    && "$PY" -m scripts.run_benchmark --method "$method" --run "$name"
  echo "DONE $method/$name exit=$?"
}
queue_a() {
  train_and_score adamf_mat mean-d256 --no-mat --no-adaptive-fusion
  train_and_score adamf_mat adamf-d256-no-structure --no-mat --drop-modality structure
  train_and_score adamf_mat adamf-d256-no-text --no-mat --drop-modality text
  train_and_score adamf_mat adamf-d256-no-image --no-mat --drop-modality image
}
queue_b() {
  train_and_score graphsage_label_text directed-clip_image-l3-h512-labeltext-d256 --image-node clip_image
  train_and_score adamf_mat adamf-d256-l2 --no-mat --normalize-modalities
  train_and_score adamf_mat adamf-d256-no-structure-no-text --no-mat --drop-modality structure --drop-modality text
}
queue_a > runs/adamf_mat/mechanism_queue_a.log 2>&1 &
queue_b > runs/adamf_mat/mechanism_queue_b.log 2>&1 &
wait
echo "ALL DONE"
