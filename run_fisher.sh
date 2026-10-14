#!/usr/bin/env bash
#
# run_fisher.sh — Fisher-similarity analysis of the EWC checkpoints
# (src/evaluations/fisher_evolution_pipeline.py), one job per seed.
#
# Usage:
#   ./run_fisher.sh [experiment]
#
# Experiments:
#   fisher_sim_last_all_datasets_local   (default) last checkpoint; each dataset's
#                                        subgroups vs. that dataset's own Fisher
#   fisher_sim_last_all_datasets_global  last checkpoint; subgroups vs. the Fisher
#                                        of the last dataset (StableDiffusion)
#   single_dataset_over_time             one dataset (DATASET_NAME) across
#                                        checkpoints 1..11
#
# Multi-seed (runs sequentially; the per-seed CSVs are merged once all five
# seeds are present in OUT_DIR):
#   SEEDS="42 123 777 2024 3407" ./run_fisher.sh
#
# Optional env:
#   BATCH_SIZE=360 OUT_DIR=results/consistency CHECKPOINT_DIR=checkpoints/ewc
#   DATASETS="VQGAN Palette DCFACe"   (all-datasets experiments: only these)
#   MERGE_INTO=<merged csv>           (replace DATASETS in an existing merged CSV)
#   DATASET_NAME=ff++                 (single_dataset_over_time only)
#   KEEP_MISSING=true                 (also group images without annotation)
#
# Requires EWC checkpoints from ./run_training.sh ewc
# (CHECKPOINT_DIR/seed_run_<seed>/ewc_<task>.pt) and the demographic
# annotations in metadata/train.csv (built from the per-task CSVs if missing).

set -euo pipefail

ENV_PATH="./.env"

# Preserve command-line / exported env overrides before loading .env.
OVERRIDE_VARS=(
  SEEDS DATASET_DIR CUDA_VISIBLE_DEVICES GIT_COMMIT
  BATCH_SIZE OUT_DIR CHECKPOINT_DIR
  DATASETS MERGE_INTO DATASET_NAME KEEP_MISSING
)
# (Plain _PRESET_<VAR> variables instead of an associative array, so this
# also runs on bash 3.2 as shipped with macOS.)
_PRESET_KEYS=()
for _k in "${OVERRIDE_VARS[@]}"; do
  if [[ -n "${!_k+x}" ]]; then
    printf -v "_PRESET_${_k}" '%s' "${!_k}"
    _PRESET_KEYS+=("$_k")
  fi
done

if [[ -f "$ENV_PATH" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_PATH"
  set +a
fi

# Re-apply preserved overrides so command-line env always wins over .env.
for _k in ${_PRESET_KEYS[@]+"${_PRESET_KEYS[@]}"}; do
  _v="_PRESET_${_k}"
  export "$_k=${!_v}"
done

EXPERIMENT="${1:-fisher_sim_last_all_datasets_local}"
case "$EXPERIMENT" in
  fisher_sim_last_all_datasets_local|fisher_sim_last_all_datasets_global|single_dataset_over_time) ;;
  *)
    echo "[ERROR] Unknown experiment: $EXPERIMENT" >&2
    echo "Usage: $0 [fisher_sim_last_all_datasets_local|fisher_sim_last_all_datasets_global|single_dataset_over_time]" >&2
    exit 1
    ;;
esac

export DATASET_DIR="${DATASET_DIR:-datasets/ai_face}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEEDS="${SEEDS:-42 123 777 2024 3407}"
BATCH_SIZE="${BATCH_SIZE:-360}"
OUT_DIR="${OUT_DIR:-results/consistency}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/ewc}"
DATASETS="${DATASETS:-}"
MERGE_INTO="${MERGE_INTO:-}"
DATASET_NAME="${DATASET_NAME:-}"
KEEP_MISSING="${KEEP_MISSING:-false}"

LOG_ROOT="logs/fisher"
mkdir -p "$LOG_ROOT"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

EXTRA_ARGS=()
if [[ -n "$DATASETS" ]]; then
  read -r -a _DATASETS_ARR <<< "$DATASETS"
  EXTRA_ARGS+=(--datasets "${_DATASETS_ARR[@]}")
fi
[[ -n "$MERGE_INTO" ]] && EXTRA_ARGS+=(--merge-into "$MERGE_INTO")
[[ -n "$DATASET_NAME" ]] && EXTRA_ARGS+=(--dataset-name "$DATASET_NAME")
[[ "$KEEP_MISSING" == "true" ]] && EXTRA_ARGS+=(--keep-missing)

# ----------------------------------------------------------------------
# Demographic annotations: build metadata/<split>.csv from the per-task
# CSVs (metadata/<split>/<task>.csv) if missing.
# ----------------------------------------------------------------------
MISSING_SPLITS=()
for _split in train; do
  [[ -f "metadata/${_split}.csv" ]] || MISSING_SPLITS+=("$_split")
done
if [[ ${#MISSING_SPLITS[@]} -gt 0 ]]; then
  echo "Building metadata/<split>.csv from the per-task CSVs for: ${MISSING_SPLITS[*]}"
  uv run ./src/utils/join_metadata.py \
    --metadata-root metadata --output-dir metadata --splits "${MISSING_SPLITS[@]}"
fi

SEEDS_NORMALIZED="$(printf '%s' "$SEEDS" | tr ',' ' ')" # "42 123" or "42,123"
for SEED in ${SEEDS_NORMALIZED}; do
  RUN_NAME="${EXPERIMENT}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

  CMD=(
    uv run ./src/evaluations/fisher_evolution_pipeline.py
    --experiment_name "$EXPERIMENT"
    --seeds "$SEED"
    --batch-size "$BATCH_SIZE"
    --dataset-dir "$DATASET_DIR"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --out-dir "$OUT_DIR"
    --log-file "$LOG_FILE"
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  )

  echo "=== ${RUN_NAME} ==="
  echo "  experiment:   ${EXPERIMENT}"
  echo "  seed:         ${SEED}"
  echo "  batch_size:   ${BATCH_SIZE}"
  echo "  datasets:     ${DATASETS:-<all>}"
  echo "  checkpoints:  ${CHECKPOINT_DIR}"
  echo "  dataset:      ${DATASET_DIR}"
  echo "  out_dir:      ${OUT_DIR}"
  echo "  gpus:         ${CUDA_VISIBLE_DEVICES}"
  echo "  git_commit:   ${GIT_COMMIT}"
  echo "  log:          ${LOG_FILE}"

  # The pipeline writes LOG_FILE itself (--log-file) and echoes to stdout.
  env GIT_COMMIT="$GIT_COMMIT" "${CMD[@]}"
done
