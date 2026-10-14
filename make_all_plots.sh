#!/usr/bin/env bash
#
# Generate the analysis plot suites from their prepared CSV inputs.
# Fairness table/figure variants are generated separately by make_all_tables.sh.
#
#   ./make_all_plots.sh                         # every plot suite
#   ./make_all_plots.sh --dry-run               # print commands only
#   ./make_all_plots.sh --plots buffer routing   # selected suites only
#   ./make_all_plots.sh --plots trajectories-grid # just the trajectory grid
#   ./make_all_plots.sh --out-dir figures_paper # choose an output root
#
# Outputs are written under <OUT_DIR>/{buffer,routing,trajectories}. The legacy
# average-accuracy and task-heatmap scripts write to plots/ themselves.

set -euo pipefail

# ----------------------------------------------------------------- config

PYTHON=${PYTHON:-python}
BUFFER_DIR=buffer_analysis
ROUTING_DIR=routing_analysis
METADATA=metadata/train.csv
OUT_DIR=figures_plots
PLOTS=(all)
DRY_RUN=0

# ----------------------------------------------------------------- args

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1; shift ;;
    --plots)      shift; PLOTS=(); while [[ $# -gt 0 && "$1" != --* ]]; do PLOTS+=("$1"); shift; done ;;
    --out-dir)    OUT_DIR=$2; shift 2 ;;
    --buffer-dir) BUFFER_DIR=$2; shift 2 ;;
    --routing-dir) ROUTING_DIR=$2; shift 2 ;;
    --metadata)   METADATA=$2; shift 2 ;;
    -h|--help)    sed -n '2,11p' "$0"; exit 0 ;;
    *)            echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

for plot in "${PLOTS[@]}"; do
  case "$plot" in
    all|buffer|routing|trajectories|trajectories-grid|average-accuracy|heatmap) ;;
    *) echo "Unknown plot suite: $plot (expected buffer, routing, trajectories, trajectories-grid, average-accuracy, heatmap, or all)" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------- helpers

run() {
  echo "+ $*"
  [[ $DRY_RUN -eq 1 ]] || "$@"
}

selected() {
  local name=$1
  [[ " ${PLOTS[*]} " == *" all " || " ${PLOTS[*]} " == *" $name " ]]
}

require_file() {
  local path=$1
  [[ -f "$path" ]] || { echo "Required input not found: $path" >&2; exit 1; }
}

require_dir() {
  local path=$1
  [[ -d "$path" ]] || { echo "Required input directory not found: $path" >&2; exit 1; }
}

# ----------------------------------------------------------------- run

if selected buffer; then
  require_file "$BUFFER_DIR/csv/representation.csv"
  require_file "$BUFFER_DIR/csv/replay_and_eviction.csv"
  run "$PYTHON" src/visualization/plot_buffer_heatmaps.py \
    --analysis-dir "$BUFFER_DIR" \
    --out-dir "$OUT_DIR/buffer" \
    --all
fi

if selected routing; then
  require_file "$ROUTING_DIR/csv/routing_aggregated.csv"
  require_file "$ROUTING_DIR/csv/routing_per_checkpoint.csv"
  require_file "$ROUTING_DIR/csv/routing_destinations_aggregated.csv"
  run "$PYTHON" src/visualization/plot_routing_heatmaps.py \
    --analysis-dir "$ROUTING_DIR" \
    --out-dir "$OUT_DIR/routing" \
    --all
fi

if selected trajectories; then
  require_file "$METADATA"
  run "$PYTHON" src/visualization/plot_trajectories.py \
    --buffer-dir "$BUFFER_DIR" \
    --routing-dir "$ROUTING_DIR" \
    --metadata "$METADATA" \
    --out-dir "$OUT_DIR/trajectories"
fi

if [[ " ${PLOTS[*]} " == *" trajectories-grid " ]]; then
  require_file "$METADATA"
  run "$PYTHON" src/visualization/plot_trajectories.py \
    --buffer-dir "$BUFFER_DIR" \
    --routing-dir "$ROUTING_DIR" \
    --metadata "$METADATA" \
    --out-dir "$OUT_DIR/trajectories" \
    --grid-only
fi

if selected average-accuracy; then
  require_dir "metrics"
  run "$PYTHON" src/visualization/plot_average_accuracy.py
fi

if selected heatmap; then
  require_dir "metrics"
  run "$PYTHON" src/visualization/plot_heatmap.py
fi

echo "Done. Plots under $OUT_DIR/ and plots/"
