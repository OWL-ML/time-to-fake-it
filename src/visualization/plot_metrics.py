import colorsys
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

STYLE_DICT = {
    "accumulated": {
        "label": "Accumulated",
        "color": "black",
        "linestyle": "-",
        "marker": "o",
    },
    "naive": {
        "label": "Naive",
        "color": "tab:red",
        "linestyle": "-",
        "marker": "o",
    },
    "derpp": {
        "label": "DER++",
        "color": "tab:green",
        "linestyle": "--",
        "marker": "o",
    },
    "soyo": {
        "label": "SOYO",
        "color": "tab:olive",
        "linestyle": "--",
        "marker": "s",
    },
    "pina": {
        "label": "PINA",
        "color": "tab:purple",
        "linestyle": "-.",
        "marker": "o",
    },
    "grad_match": {
        "label": "GradMatch",
        "color": "tab:brown",
        "linestyle": "-.",
        "marker": "s",
    },
    "ewc": {
        "label": "EWC",
        "color": "tab:pink",
        "linestyle": ":",
        "marker": "s",
    },
    "sprompts": {
        "label": "S-Prompts",
        "color": "tab:gray",
        "linestyle": ":",
        "marker": "D",
    },
}


def plot_accuracy_diff_heatmap(
    source_acc, target_acc, model_name, title="Accuracy Change Heatmap"
):
    """
    Plot a pastel red-green heatmap of accuracy differences between target and source matrices.

    Parameters:
        source_acc (np.ndarray): Baseline accuracy matrix (e.g., naive sequential)
        target_acc (np.ndarray): Target accuracy matrix (e.g., EWC)
        title (str): Title of the heatmap
    """
    # Ensure both are numpy arrays
    file_name = os.path.join("plots", f"task_accuracy_change_{model_name}.png")
    source_acc = np.array(source_acc)
    target_acc = np.array(target_acc)

    # Check if target_acc is larger than source_acc
    if (
        target_acc.shape[0] > source_acc.shape[0]
        or target_acc.shape[1] > source_acc.shape[1]
    ):
        raise ValueError("target_acc must be smaller or equal to source_acc")
    # Trim to the number of tasks in target
    min_rows = min(source_acc.shape[0], target_acc.shape[0])
    min_cols = min(source_acc.shape[1], target_acc.shape[1])

    diff = target_acc[:min_rows, :min_cols] - source_acc[:min_rows, :min_cols]

    # Define a smooth pastel red-green colormap
    colors = [
        (1.0, 0.85, 0.85),  # light red
        (1.0, 1.0, 1.0),  # white
        (0.85, 1.0, 0.85),  # light green
    ]
    cmap = LinearSegmentedColormap.from_list("pastel_red_green", colors)

    # Plot heatmap
    plt.figure(figsize=(min_cols * 1.2, min_rows * 1.2))
    sns.heatmap(
        diff,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Δ Accuracy"},
    )
    plt.title(title)
    plt.xlabel("Evaluated Task")
    plt.ylabel("Trained Task")
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def plot_heatmap(matrix, tasks, model_name):
    sns.set(font_scale=1.5)
    # FIXED HUE (pure red)
    H = 0.0

    # REDUCED SATURATION for pastel effect
    # 1.0 = vivid red, 0.0 = gray; pastel = 0.2–0.5 range
    S = 0.5  # tweak this to adjust pastel softness

    # Lightness curve (smooth, nonlinear)
    x = np.linspace(0, 1, 256)

    # Pastel-friendly lightness curve:
    # - very light at the top end
    # - slow drop until 80
    # - steeper drop below 50
    L = (
        0.25  # baseline lightness (pastel softness)
        + 0.75 * (x**2.5)  # gamma curve to control darkness
    )

    # Normalize max lightness to ~1.0
    L = L / L.max()

    # Convert to RGB
    rgb = np.array([colorsys.hls_to_rgb(H, lr, S) for lr in L])

    # Build colormap
    cmap_pastel_red = LinearSegmentedColormap.from_list("pastel_red_smooth", rgb)

    file_name = os.path.join("plots", f"task_accuracy_heatmap_{model_name}.pdf")
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap_pastel_red,
        vmin=0,
        vmax=100,
        # norm=norm,
        xticklabels=[f"{t}" for t in tasks],
        yticklabels=[f"{t}" for t in tasks],
        annot_kws={"size": 14},
    )
    plt.yticks(rotation=0)
    plt.xlabel("Evaluated Task")
    plt.ylabel("Trained Task")
    # plt.title("Task-wise Accuracy Heatmap")

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Matches the line plot's rect

    plt.savefig(file_name, dpi=300)
    plt.show()


def build_accuracy_matrix(log_dict):
    # Sort by task_id key
    tasks = sorted(log_dict.keys(), key=lambda x: int(x))
    num_tasks = len(tasks)

    matrix = np.zeros((num_tasks, num_tasks))

    for row_idx, t in enumerate(tasks):  # after learning task t
        entry = log_dict[t]

        # Extract accuracies task_1_acc_after_task_t ... task_N_acc_after_task_t
        for col_idx in range(num_tasks):  # evaluation on task j
            key = f"task_{col_idx + 1}_acc_after_task_{t}"
            matrix[row_idx, col_idx] = entry.get(key, np.nan)

    return matrix, tasks


def plot_avg_acc_comparison(metric_details):
    file_name = os.path.join("plots", "comparison_plot.pdf")
    plt.figure(figsize=(10, 8))
    tasks_len = 0
    # Plot each list
    for method_key, metrics in metric_details.items():
        style = STYLE_DICT.get(method_key, {})
        legend_label = style.pop("label", method_key)
        plt.plot(metrics, label=legend_label, **style)
        tasks_len = max(tasks_len, len(metrics))
        # plt.plot(metrics, label=label)
        # print(label)

    metrics_df = pd.DataFrame(metric_details)
    metrics_df.to_csv("metrics.csv")
    # Axis labels
    plt.xlabel("Task ID", fontsize=16)
    plt.ylabel("Average Accuracy", fontsize=16)
    x_positions = range(tasks_len)
    x_labels = [f"{i + 1}" for i in range(tasks_len)]

    plt.xticks(x_positions, x_labels, ha="right")

    # plt.legend(loc="upper center", bbox_to_anchor=(0.48, 1.25), ncol=4, frameon=False)

    # plt.subplots_adjust(bottom=0.25, top=0.78)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)

    # plt.subplots_adjust(top=0.82)
    #
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    # plt.tight_layout(rect=[0, 0.18, 1, 0.82])
    # x_tick_labels = [f"{i + 1}" for i in range(tasks_len)]
    # plt.xticks(range(tasks_len), x_tick_labels, fontsize=16)
    # plt.xticks(rotation=45)

    # # plt.title("Comparison of Average Accuracy")
    # # plt.legend()
    # plt.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(1, 1.15),
    #     ncol=4,
    #     frameon=False
    # )
    plt.yticks([40, 50, 60, 70, 80, 90, 100])
    plt.grid(axis="y", linestyle="-", alpha=0.6)
    plt.grid(axis="x", linestyle="-", alpha=0.6)
    plt.savefig(file_name, dpi=300)
    plt.show()


def plot_metrics():
    metrics_details = {}
    accuracy_metrics = {}
    for mid, metric in enumerate(os.listdir("metrics")):
        if metric.endswith("_metrics.json"):
            model_name = Path(metric).stem.replace("_metrics", "")
            print(f"Processing {model_name}")
            metrics_file = os.path.join("metrics", metric)
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

                average_acc = []
                matrix, tasks = build_accuracy_matrix(metrics["task_wise_performance"])
                plot_heatmap(matrix, tasks, model_name)
                accuracy_metrics[model_name] = matrix
                for task in metrics["task_wise_performance"].values():
                    average_acc.append(task["average_accuracy"])
                metrics_details[model_name] = average_acc

    plot_avg_acc_comparison(metrics_details)
    plot_accuracy_diff_heatmap(
        accuracy_metrics["naive"],
        accuracy_metrics["ewc"],
        "ewc",
    )


def main():
    # model_names = ["naive", "accumulated"]

    plot_metrics()


if __name__ == "__main__":
    main()
