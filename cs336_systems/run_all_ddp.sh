#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p profiles

echo "===== GPU topology ====="
nvidia-smi topo -m || true
echo

echo "===== Experiment 1: All-Reduce benchmark ====="
uv run --no-sync torchrun --standalone --nproc_per_node=2 \
  cs336_systems/ddp_allreduce_bench.py \
  --sizes-mib 1 10 100 1000 \
  --warmup 5 \
  --reps 20 \
  --output profiles/allreduce_bench.csv

echo
echo "===== Experiment 2: Naive vs Flat All-Reduce ====="
uv run --no-sync torchrun --standalone --nproc_per_node=2 \
  cs336_systems/ddp_sync_bench.py \
  --dim 1024 \
  --layers 12 \
  --batch 8 \
  --seq 256 \
  --warmup 3 \
  --steps 10 \
  --output profiles/ddp_sync_bench.csv

echo
echo "===== Experiment 3: Naive vs Overlap DDP ====="
uv run --no-sync torchrun --standalone --nproc_per_node=2 \
  cs336_systems/ddp_overlap_bench.py \
  --dim 1024 \
  --layers 12 \
  --batch 8 \
  --seq 256 \
  --warmup 3 \
  --steps 10 \
  --output profiles/ddp_overlap_bench.csv

echo
echo "===== Done ====="
ls -lh profiles/allreduce_bench.csv profiles/ddp_sync_bench.csv profiles/ddp_overlap_bench.csv
