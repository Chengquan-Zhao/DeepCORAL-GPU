#!/usr/bin/env bash
# 复现 Table 1（Linux 服务器用）：6 个迁移方向 × 5 次运行，Deep CORAL 与无适配基线（λ=0）各跑一遍
# 用法：LAMBDA=1 ITERS=2000 SEEDS="0 1 2 3 4" GPU=0 bash run_all.sh
set -e
cd "$(dirname "$0")"
LAMBDA=${LAMBDA:-1}
ITERS=${ITERS:-2000}
SEEDS=${SEEDS:-"0 1 2 3 4"}
GPU=${GPU:-0}
for seed in $SEEDS; do
  for st in AD AW DA DW WA WD; do
    for lam in "$LAMBDA" 0; do
      python train.py --source "${st:0:1}" --target "${st:1:1}" --lambda_coral "$lam" \
        --iters "$ITERS" --seed "$seed" --gpu "$GPU"
    done
  done
done
python plot.py --table
python plot.py --shift A2W --lambda_coral "$LAMBDA" --seed 0
