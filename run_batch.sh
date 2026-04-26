#!/usr/bin/env bash
set -e  # stop on first failure

CONFIGS=(
  "configs/project_runs_1_conv2d_iqgrid_bptt.yaml"
  "configs/project_runs_2_conv2d_iqgrid_bptt.yaml"
  "configs/project_runs_3_conv2d_iqgrid_bptt.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=============================="
  echo "Running config: $cfg"
  echo "=============================="
  python run_experiment.py --config "$cfg"
done