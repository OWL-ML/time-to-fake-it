#!/usr/bin/env bash
#
# run_training.sh — launch continual-learning training on a single GPU.
#
# Usage:
#   ./run_training.sh naive|accumulated|ewc|derpp|grad_match   (src/train_cl.py)
#   ./run_training.sh sprompts|soyo|pina                       (src/train_prompts_cl.py)
#
# Multi-seed:
#   SEEDS="42 123 777" ./run_training.sh naive
#
# Optional env (all methods):
#   BATCH_SIZE=128 MODEL_NAME=<method>
# Optional env (naive/accumulated/ewc/derpp/grad_match):
#   START_TASK=0
#   LAMBDA_EWC=1 ALPHA=0.8 BETA=0.5 REPLAY_SIZE=64 CAPACITY=1500
# Optional env (sprompts/soyo/pina):
#   EVAL_ONLY=true
#
# Demographic annotations are always read from metadata/{train,val,test}.csv;
# they are built from the per-task CSVs (src/utils/join_metadata.py) if missing.

set -euo pipefail

ENV_PATH="./.env"

# Preserve command-line / exported env overrides before loading .env.
OVERRIDE_VARS=(
  SEEDS DATASET_DIR CUDA_VISIBLE_DEVICES
  BATCH_SIZE MODEL_NAME GIT_COMMIT
  START_TASK
  LAMBDA_EWC ALPHA BETA REPLAY_SIZE CAPACITY
  EVAL_ONLY
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

METHOD="${1:?Usage: $0 naive|accumulated|ewc|derpp|grad_match|sprompts|soyo|pina}"

export DATASET_DIR="${DATASET_DIR:-datasets/ai_face}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BATCH_SIZE="${BATCH_SIZE:-128}" # FIXED across all methods (protocol)
MODEL_NAME="${MODEL_NAME:-$METHOD}"

LOG_ROOT="logs/runs"
mkdir -p "$LOG_ROOT"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

METHOD_ARGS=()
RUN_INFO=()

case "$METHOD" in
  naive|accumulated|ewc|derpp|grad_match)
    # ------------------------------------------------------------------
    # Fine-tuning / regularization / replay baselines (src/train_cl.py)
    # ------------------------------------------------------------------
    SCRIPT="./src/train_cl.py"
    SEED_FLAG="--base_seed"
    SEEDS="${SEEDS:-777}"

    START_TASK="${START_TASK:-0}"

    LAMBDA_EWC="${LAMBDA_EWC:-1}"
    ALPHA="${ALPHA:-0.8}"
    BETA="${BETA:-0.5}"
    REPLAY_SIZE="${REPLAY_SIZE:-64}"
    CAPACITY="${CAPACITY:-1500}"

    case "$METHOD" in
      accumulated)
        METHOD_ARGS+=(--accumulated true)
        ;;
      ewc)
        METHOD_ARGS+=(--ewc true --lambda_ewc "$LAMBDA_EWC")
        ;;
      derpp)
        METHOD_ARGS+=(
          --derpp true
          --alpha "$ALPHA"
          --beta "$BETA"
          --replay_size "$REPLAY_SIZE"
          --capacity "$CAPACITY"
        )
        ;;
      grad_match)
        METHOD_ARGS+=(
          --grad_match true
          --beta "$BETA"
          --replay_size "$REPLAY_SIZE"
          --capacity "$CAPACITY"
        )
        ;;
    esac

    METHOD_ARGS+=(--start_task "$START_TASK")

    RUN_INFO+=("start_task:   ${START_TASK}")
    ;;

  sprompts|soyo|pina)
    # ------------------------------------------------------------------
    # Prompt-based methods (src/train_prompts_cl.py)
    # ------------------------------------------------------------------
    SCRIPT="./src/train_prompts_cl.py"
    SEED_FLAG="--seed"
    SEEDS="${SEEDS:-42}"
    EVAL_ONLY="${EVAL_ONLY:-false}"

    # Shared protocol hyperparameters (FIXED across prompt methods).
    METHOD_ARGS+=(
      --method        "$METHOD"
      --evaluate_only "$EVAL_ONLY"
      # Epoch budget from your SOYO config (init_epoch: 20, epochs: 25).
      --init_epochs 25
      --epochs      25
    )

    # Method-specific hyperparameters (TUNED per method, then frozen & reported)
    #
    # !! The lr/epoch values below are sensible starting points in the style of
    # !! the reference repos. Before final runs, diff them against the actual
    # !! JSON configs in the S-Prompts and SOYO repos (exps/cddb_*.json) and
    # !! adopt those values — cite the config file in your setup table.
    # weight_decay is per method: SOYO's config says 2e-4; verify the
    # S-Prompts / PINA configs independently before assuming the same value.
    case "$METHOD" in
      sprompts)
        # !! Placeholder values in the SOYO style — verify against your actual
        # !! S-Prompts config before final runs.
        METHOD_ARGS+=(
          --prompt_length 10
          --init_lr       5e-4
          --lr            5e-4
          --weight_decay  2e-4
        )
        ;;
      soyo)
        # Values from your SOYO config (cddb yaml) — verified ground truth.
        METHOD_ARGS+=(
          --prompt_length 10
          --hidden_dim    16        # adapter bottleneck width (config: 16)
          --init_lr       5e-4      # config: init_lr 0.0005
          --lr            5e-4      # config: lr 0.0005
          --weight_decay  2e-4      # config: 2.0e-4 (init and main)
          # Phase 2: domain-classifier (soyo_network) training
          --soyo_epoch        20    # config: 20
          --soyo_lr           1e-3  # config: 0.001
          --soyo_weight_decay 2e-4  # config: 2.0e-4
        )
        ;;
      pina)
        # !! Placeholder values mirroring SOYO — verify against your actual
        # !! PINA config (esp. ca_mode and hidden_dim) before final runs.
        METHOD_ARGS+=(
          --prompt_length 10
          --hidden_dim    16
          --ca_mode       deep      # or "shallow"; report which
          --init_lr       5e-4
          --lr            5e-4
          --weight_decay  2e-4
          --soyo_epoch        20    # PINA shares the soyo_network routing phase
          --soyo_lr           1e-3
          --soyo_weight_decay 2e-4
        )
        ;;
    esac

    RUN_INFO+=("eval_only:    ${EVAL_ONLY}")
    ;;

  *)
    echo "[ERROR] Unknown method: $METHOD" >&2
    exit 1
    ;;
esac

# ----------------------------------------------------------------------
# Demographic annotations: build metadata/<split>.csv from the per-task
# CSVs (metadata/<split>/<task>.csv) if missing.
# ----------------------------------------------------------------------
MISSING_SPLITS=()
for _split in train val test; do
  [[ -f "metadata/${_split}.csv" ]] || MISSING_SPLITS+=("$_split")
done
if [[ ${#MISSING_SPLITS[@]} -gt 0 ]]; then
  echo "Building metadata/<split>.csv from the per-task CSVs for: ${MISSING_SPLITS[*]}"
  uv run ./src/utils/join_metadata.py \
    --metadata-root metadata --output-dir metadata --splits "${MISSING_SPLITS[@]}"
fi

# ----------------------------------------------------------------------
# Launch (one run per seed, sequential; logs tee'd per run)
# ----------------------------------------------------------------------
SEEDS_NORMALIZED="$(printf '%s' "$SEEDS" | tr ',' ' ')" # "42 123" or "42,123"
for SEED in ${SEEDS_NORMALIZED}; do
  RUN_NAME="${MODEL_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

  CMD=(
    uv run "$SCRIPT"
    --model_name "$MODEL_NAME"
    --batch_size "$BATCH_SIZE"
    "$SEED_FLAG" "$SEED"
    "${METHOD_ARGS[@]}"
  )

  echo "=== ${RUN_NAME} ==="
  echo "  method:       ${METHOD}"
  echo "  model_name:   ${MODEL_NAME}"
  echo "  script:       ${SCRIPT}"
  echo "  seed:         ${SEED}"
  echo "  batch_size:   ${BATCH_SIZE}"
  for line in "${RUN_INFO[@]}"; do
    echo "  ${line}"
  done
  echo "  dataset:      ${DATASET_DIR}"
  echo "  gpus:         ${CUDA_VISIBLE_DEVICES}"
  echo "  git_commit:   ${GIT_COMMIT}"
  echo "  log:          ${LOG_FILE}"

  env RUN_NAME="$RUN_NAME" GIT_COMMIT="$GIT_COMMIT" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
done
