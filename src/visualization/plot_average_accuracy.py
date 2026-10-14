import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE_DICT = {
    "accumulated": {
        "label": "Accumulated",
        "color": "black",
        "linestyle": "-",
        "marker": "o",
    },
    "naive": {"label": "Naive", "color": "tab:red", "linestyle": "-", "marker": "o"},
    "derpp": {"label": "DER++", "color": "tab:green", "linestyle": "--", "marker": "o"},
    "soyo": {"label": "SOYO", "color": "tab:olive", "linestyle": "--", "marker": "s"},
    "pina": {"label": "PINA", "color": "tab:purple", "linestyle": "-.", "marker": "o"},
    "grad_match": {
        "label": "GradMatch",
        "color": "tab:brown",
        "linestyle": "-.",
        "marker": "s",
    },
    "ewc": {"label": "EWC", "color": "tab:pink", "linestyle": ":", "marker": "s"},
    "sprompts": {
        "label": "S-Prompts",
        "color": "tab:gray",
        "linestyle": ":",
        "marker": "D",
    },
}


def plot_avg_acc_comparison(metric_details):
    """
    Plots mean accuracies and standard deviation bands using your original style rules.

    Parameters:
    -----------
    metric_details : dict
        A nested dictionary structured like this:
        {
            "naive": {"mean": [90.2, 85.1, ...], "std": [1.2, 2.3, ...]},
            "soyo":  {"mean": [92.4, 89.0, ...], "std": [0.8, 1.5, ...]}
        }
    """
    file_name = os.path.join("plots", "comparison_plot.pdf")
    plt.figure(figsize=(10, 8))
    tasks_len = 0

    # 1. Plot each method's mean line and deviation band
    for method_key, metrics in metric_details.items():
        means = np.array(metrics["mean"])
        stds = np.array(metrics["std"])
        tasks_len = max(tasks_len, len(means))

        # Safely copy style configuration so we don't break STYLE_DICT across repeated runs
        style = STYLE_DICT.get(method_key, {}).copy()
        legend_label = style.pop("label", method_key)
        color = style.get("color", "black")

        x_positions = np.arange(len(means))

        # Plot the standard deviation band first (renders behind the line)
        plt.fill_between(
            x_positions,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,  # Soft opacity so overlapping bands stay readable
        )

        # Plot the main trendline matching your exact original style
        plt.plot(x_positions, means, label=legend_label, **style)

    # 2. Export the metrics matrix to a CSV file (saves the mean scores)
    # export_dict = {k: v["mean"] for k, v in metric_details.items()}

    # 3. Axis layout, labels, and formatting matching your original code
    plt.xlabel("Task ID", fontsize=16)
    plt.ylabel("Average Accuracy", fontsize=16)

    x_positions = range(tasks_len)
    x_labels = [f"{i + 1}" for i in range(tasks_len)]
    plt.xticks(x_positions, x_labels, ha="right")

    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Anchored at the upper center, but expanded across 100% of the plot width
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.1),  # (x_start, y_start, width, height)
        ncol=4,
        mode="expand",
        frameon=False,
        fontsize=14,  # Matches the clean original text size
        borderaxespad=0.0,
    )

    # Ensures the expanded top legend has enough padding and won't get cut off
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.yticks([40, 50, 60, 70, 80, 90, 100])
    plt.grid(axis="y", linestyle="-", alpha=0.6)
    plt.grid(axis="x", linestyle="-", alpha=0.6)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(file_name, dpi=300)
    plt.show()


def parse_seed_json(json_data):
    """Flattens a single seed's JSON metrics into a flat tabular format."""
    rows = []
    model_name = json_data["model_name"]
    performance = json_data["task_wise_performance"]

    # Iterate through each training stage (j)
    for j_str, task_data in performance.items():
        j = int(j_str)  # The task the model was just trained on

        # Look for accuracy keys like 'task_1_acc_after_task_1'
        rows.append(
            {
                "model": model_name,
                "task": j,  # Row identifier in the final heatmap
                "average_accuracy": task_data.get("average_accuracy"),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 1. Parse and compile all 5 seeds into a single large master list
    #
    metrics_dir = "gitroot/aifacedetection/metrics"
    seeds = [42, 123, 777, 2024, 3407]
    metric_files = sorted(os.listdir(metrics_dir))
    all_models = {}
    accuracy_df = pd.DataFrame(columns=["model", "seed", "task", "average_accuracy"])
    all_seeds_dfs = []
    for metric_file in metric_files:
        if metric_file.endswith(".bak"):
            continue
        model_name = metric_file.split("_")[0]
        if model_name == "grad":
            model_name = "grad_match"
        seed = int(metric_file.split("_")[-2])
        if seed not in seeds:
            continue

        if model_name not in all_models:
            all_models[model_name] = {"mean": [], "std": []}
        metric_file_path = os.path.join(metrics_dir, metric_file)
        with open(metric_file_path, "r") as f:
            data = json.load(f)
            all_seeds_dfs.append(parse_seed_json(data))
    master_raw_df = pd.concat(all_seeds_dfs, ignore_index=True)
    # 2. Group by model, evaluation task, and training stage to compute metrics across seeds
    summary_df = (
        master_raw_df.groupby(["model", "task"])["average_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    formatted_metric_details = {}
    for model, group in summary_df.groupby("model"):
        sorted_group = group.sort_values("task")
        formatted_metric_details[model] = {
            "mean": sorted_group["mean"].values,
            "std": sorted_group["std"].values,
        }
    plot_avg_acc_comparison(formatted_metric_details)
