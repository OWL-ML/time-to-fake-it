import colorsys
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_heatmap_custom_text(mean_matrix, std_matrix, tasks, model_name):
    sns.set(font_scale=1.5)

    # FIXED HUE (pure red)
    H = 0.0
    # REDUCED SATURATION for pastel effect
    S = 0.5
    # Lightness curve (smooth, nonlinear)
    x = np.linspace(0, 1, 256)
    L = 0.25 + 0.75 * (x**2.5)
    L = L / L.max()

    # Convert to RGB
    rgb = np.array([colorsys.hls_to_rgb(H, lr, S) for lr in L])
    cmap_pastel_red = LinearSegmentedColormap.from_list("pastel_red_smooth", rgb)

    file_name = os.path.join("plots", f"task_accuracy_heatmap_{model_name}.pdf")
    fig, ax = plt.subplots(
        figsize=(11, 9), dpi=300
    )  # Slightly wider for 2-decimal strings

    vmin, vmax = 0, 100

    # 1. Plot the base heatmap
    sns.heatmap(
        mean_matrix,
        annot=False,
        cmap=cmap_pastel_red,
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"{t}" for t in tasks],
        yticklabels=[f"{t}" for t in tasks],
        ax=ax,
    )

    # 2. Loop over cells to place text with dynamic color contrast
    num_tasks = len(tasks)
    for row in range(num_tasks):
        for col in range(num_tasks):
            # Extract cell values cleanly whether input is DataFrame or NumPy array
            mean_val = (
                mean_matrix.iloc[row, col]
                if isinstance(mean_matrix, pd.DataFrame)
                else mean_matrix[row, col]
            )
            std_val = (
                std_matrix.iloc[row, col]
                if isinstance(std_matrix, pd.DataFrame)
                else std_matrix[row, col]
            )

            if pd.isna(mean_val):
                continue

            # Dynamic Contrast Calculation: Find the cell color to choose light/dark font
            # Normalize the mean value between 0.0 and 1.0 based on vmin/vmax
            norm_val = (mean_val - vmin) / (vmax - vmin)
            norm_val = np.clip(norm_val, 0.0, 1.0)

            # Fetch the RGB color of this specific cell from our colormap
            cell_rgb = cmap_pastel_red(norm_val)[:3]

            # W3C Formula for Relative Luminance (Perceived Brightness)
            luminance = (
                0.2126 * cell_rgb[0] + 0.7152 * cell_rgb[1] + 0.0722 * cell_rgb[2]
            )

            # Choose text color based on background brightness threshold
            text_color = "#1A1A1A" if luminance > 0.45 else "#FFFFFF"
            std_alpha = (
                0.8 if text_color == "#1A1A1A" else 0.85
            )  # Tweak readability for dark cells

            # Center coordinates for current grid cell
            x_pos = col + 0.5
            y_pos = row + 0.5

            # Write Mean Accuracy (Two-decimal precision)
            ax.text(
                x_pos,
                y_pos - 0.12,
                f"{mean_val:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
                color=text_color,
            )

            # Write Standard Deviation (Two-decimal precision, smaller font)
            ax.text(
                x_pos,
                y_pos + 0.20,
                f"±{std_val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                alpha=std_alpha,
            )

    plt.yticks(rotation=0)
    plt.xlabel("Evaluated Task")
    plt.ylabel("Trained Task")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.tight_layout(rect=[0, 0, 1, 0.9])

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
        for key, accuracy in task_data.items():
            if "acc_after_task" in key:
                # Extract the evaluation task ID (i) from the key string
                parts = key.split("_")
                i = int(parts[1])  # Gets '1' out of 'task_1_acc_after_task_1'

                rows.append(
                    {
                        "model": model_name,
                        "task": i,  # Row identifier in the final heatmap
                        "after_task": j,  # Column identifier in the final heatmap
                        "accuracy": accuracy,
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
    filter_model = ["pina"]
    accuracy_df = pd.DataFrame(columns=["model", "seed", "task", "average_accuracy"])
    all_seeds_dfs = []
    for metric_file in metric_files:
        if metric_file.endswith(".bak"):
            continue
        model_name = metric_file.split("_")[0]
        if model_name == "grad":
            model_name = "grad_match"
        if model_name not in filter_model:
            continue
        seed = int(metric_file.split("_")[-2])
        if seed not in seeds:
            continue

        print(f"Processing {model_name} with seed {seed}")

        if model_name not in all_models:
            all_models[model_name] = {"mean": [], "std": []}
        metric_file_path = os.path.join(metrics_dir, metric_file)
        with open(metric_file_path, "r") as f:
            data = json.load(f)
            all_seeds_dfs.append(parse_seed_json(data))
    master_raw_df = pd.concat(all_seeds_dfs, ignore_index=True)

    # 2. Group by model, evaluation task, and training stage to compute metrics across seeds
    summary_df = (
        master_raw_df.groupby(["model", "task", "after_task"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # 3. Rename columns to exactly match your requested naming structure
    summary_df = summary_df.rename(
        columns={"mean": "average_accuracy", "std": "standard_deviation"}
    )

    # Get a list of unique models found in your dataframe
    unique_models = summary_df["model"].unique()
    print(f"Found {len(unique_models)} unique models.")

    for model in unique_models:
        if model != filter_model:
            continue
        print(f"Generating custom text heatmap for {model}...")

        # Filter for a single model
        model_df = summary_df[summary_df["model"] == model]

        # Pivot into standard rows (Trained Task) and columns (Evaluated Task)
        # Note: Your initial script maps trained tasks to rows and evaluation tasks to columns
        mean_matrix = model_df.pivot(
            index="after_task", columns="task", values="average_accuracy"
        )
        std_matrix = model_df.pivot(
            index="after_task", columns="task", values="standard_deviation"
        )

        # Extract the ordered task sequence labels
        tasks_list = sorted(model_df["task"].unique())

        # Generate the chart
        plot_heatmap_custom_text(mean_matrix, std_matrix, tasks_list, model_name=model)
