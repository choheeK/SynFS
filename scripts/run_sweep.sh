#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running Optuna sweep"
echo "--------------------"


python train.py -m hydra/sweeper=optuna \
  'hydra.sweeper.direction=maximize' \
  'hydra.sweeper.n_trials=20' \
  'hydra.sweeper.sampler._target_=optuna.samplers.TPESampler' \
  'hydra.sweeper.sampler.seed=0' \
  'model.learning_rate=tag(log, interval(1e-4, 1e-2))' \
  'model.s_lam=interval(0.01, 1.0)' \
  'model.ns_alpha=interval(0.0, 0.5)'

