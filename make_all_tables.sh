#!/usr/bin/env bash
#
# Generate every table/figure variant from the per-run fairness CSVs.
#
#   ./make_all_tables.sh                 # all figure types, all datasets
#   ./make_all_tables.sh --dry-run       # print the commands without running them
#   ./make_all_tables.sh --per-seed      # also one figure set per seed, for a
#                                        # seed-sensitivity check
#   ./make_all_tables.sh --figures algs  # only that type (diff|algs|bars)
#   ./make_all_tables.sh --ds 1 3        # only those datasets for the algs tables
#
# Figures land in <FIG_DIR>/main, and the per-seed sets in <FIG_DIR>/seed_<seed>.

set -euo pipefail

# ----------------------------------------------------------------- config

# Order matters: the first method supplies the "Init. F" column and is the
# colour reference of the algorithm tables.
METHODS=(accumulated naive ewc derpp grad_match soyo sprompts pina)
METRICS=(FEO FDP FOAE FEOP)
FINAL_TASK=11

FAIRNESS_DIR=fairness_metrics     # <method>/seed_run_<seed>/*.csv
MERGED_DIR=fairness_merged
FIG_DIR=figures
PYTHON=${PYTHON:-python}

# ----------------------------------------------------------------- args

DRY_RUN=0
PER_SEED=0
FIGURES=(all)
DATASETS=(all)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --per-seed) PER_SEED=1; shift ;;
    --figures)  shift; FIGURES=(); while [[ $# -gt 0 && "$1" != --* ]]; do FIGURES+=("$1"); shift; done ;;
    --ds)       shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --methods)  shift; METHODS=();  while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    *)          echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------- helpers

run() {
  echo "+ $*"
  [[ $DRY_RUN -eq 1 ]] || "$@"
}

# Only the methods that actually have prediction-derived CSVs; a missing one
# would otherwise abort the whole run inside src/visualization/plot_fairness_table.py.
available_methods() {
  local found=()
  for m in "${METHODS[@]}"; do
    if compgen -G "$FAIRNESS_DIR/$m/seed_run_*/fairness_group.csv" > /dev/null; then
      found+=("$m")
    else
      echo "  skipping '$m': no $FAIRNESS_DIR/$m/seed_run_*/fairness_group.csv" >&2
    fi
  done
  printf '%s\n' "${found[@]}"
}

discover_seeds() {
  find "$FAIRNESS_DIR" -mindepth 2 -maxdepth 2 -type d -name 'seed_run_*' \
    | sed 's/.*seed_run_//' | sort -u
}

# ----------------------------------------------------------------- run

[[ -d "$FAIRNESS_DIR" ]] || { echo "No such directory: $FAIRNESS_DIR" >&2; exit 1; }

mapfile -t USE_METHODS < <(available_methods)
[[ ${#USE_METHODS[@]} -gt 0 ]] || { echo "No methods found under $FAIRNESS_DIR" >&2; exit 1; }

echo "Methods:  ${USE_METHODS[*]}"
echo "Figures:  ${FIGURES[*]}   Datasets: ${DATASETS[*]}"

common=(
  --fairness-dir "$FAIRNESS_DIR"
  --methods "${USE_METHODS[@]}"
  --metrics "${METRICS[@]}"
  --final-task "$FINAL_TASK"
  --figures "${FIGURES[@]}"
  --datasets "${DATASETS[@]}"
)

# 1) The main set: every method, every seed.
run "$PYTHON" src/visualization/plot_fairness_table.py \
  "${common[@]}" \
  --merged-dir "$MERGED_DIR/main" \
  --fig-dir "$FIG_DIR/main"

# 2) One set per seed: same figures with the other seeds excluded, so a claim
#    that rests on a single seed is easy to spot. Each gets its own merged dir,
#    because --merged-dir is rebuilt on every run.
if [[ $PER_SEED -eq 1 ]]; then
  mapfile -t SEEDS < <(discover_seeds)
  [[ ${#SEEDS[@]} -gt 0 ]] || { echo "No seed_run_* directories found" >&2; exit 1; }
  echo "Seeds:    ${SEEDS[*]}"

  for keep in "${SEEDS[@]}"; do
    drop=()
    for s in "${SEEDS[@]}"; do [[ "$s" == "$keep" ]] || drop+=("$s"); done

    args=("${common[@]}" --merged-dir "$MERGED_DIR/seed_$keep" --fig-dir "$FIG_DIR/seed_$keep")
    [[ ${#drop[@]} -eq 0 ]] || args+=(--exclude-seeds "${drop[@]}")

    run "$PYTHON" src/visualization/plot_fairness_table.py "${args[@]}"
  done
fi

echo "Done. Figures under $FIG_DIR/"
