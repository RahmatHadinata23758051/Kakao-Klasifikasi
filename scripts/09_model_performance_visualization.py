import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent
outputs_dir = base_dir / "outputs"
results_path = outputs_dir / "08_model_benchmark_results.json"


MODEL_LABELS = {
    "hog_linear_svm": "HSV + HOG\nLinear SVM",
    "mobilenetv2_embeddings": "MobileNetV2\nEmbeddings + LR",
    "efficientnetb0_embeddings": "EfficientNetB0\nEmbeddings + LR",
    "mobilenetv2_saved_keras": "MobileNetV2\nSaved Keras",
}

MODEL_COLORS = {
    "hog_linear_svm": "#d95f5f",
    "mobilenetv2_embeddings": "#4c78a8",
    "efficientnetb0_embeddings": "#1aaf5d",
    "mobilenetv2_saved_keras": "#f2a541",
}

METRIC_CONFIG = {
    "test_accuracy": ("Accuracy", "Test Accuracy"),
    "test_precision_macro": ("Precision", "Test Precision (Macro)"),
    "test_recall_macro": ("Recall", "Test Recall (Macro)"),
    "test_f1_macro": ("F1-Score", "Test F1-Score (Macro)"),
}


def load_results():
    with results_path.open("r", encoding="utf-8") as handle:
        raw_results = json.load(handle)

    rows = []
    for item in raw_results:
        rows.append(
            {
                "model_slug": item["model_slug"],
                "model_label": MODEL_LABELS.get(item["model_slug"], item["display_name"]),
                "family": item["family"],
                "runtime_sec": item["runtime_sec"],
                "val_accuracy": item["val_metrics"]["accuracy"],
                "val_f1_macro": item["val_metrics"]["f1_macro"],
                "test_accuracy": item["test_metrics"]["accuracy"],
                "test_precision_macro": item["test_metrics"]["precision_macro"],
                "test_recall_macro": item["test_metrics"]["recall_macro"],
                "test_f1_macro": item["test_metrics"]["f1_macro"],
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe["sort_rank"] = dataframe["test_f1_macro"].rank(method="dense", ascending=False)
    dataframe = dataframe.sort_values(["sort_rank", "test_accuracy"], ascending=[True, False]).reset_index(drop=True)
    return dataframe


def annotate_bars(axis, bars):
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(height + 0.015, 1.03),
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def style_metric_axis(axis, metric_title, baseline_value):
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_xlabel("Algorithm")
    axis.set_title(metric_title, fontsize=12, fontweight="bold")
    axis.axhline(
        baseline_value,
        color="#d62728",
        linestyle=(0, (3, 3)),
        linewidth=1.5,
        alpha=0.9,
    )
    axis.grid(axis="y", linestyle="--", alpha=0.25)


def build_dashboard(dataframe: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    figure = plt.figure(figsize=(16, 12))
    grid = GridSpec(3, 2, figure=figure, height_ratios=[1, 1, 0.95], hspace=0.32, wspace=0.18)

    baseline_row = dataframe[dataframe["model_slug"] == "mobilenetv2_saved_keras"].iloc[0]
    x_labels = dataframe["model_label"].tolist()
    colors = [MODEL_COLORS[slug] for slug in dataframe["model_slug"]]

    for index, (metric_key, (_, metric_title)) in enumerate(METRIC_CONFIG.items()):
        row, col = divmod(index, 2)
        axis = figure.add_subplot(grid[row, col])
        bars = axis.bar(x_labels, dataframe[metric_key], color=colors, edgecolor="#1f2937", linewidth=0.8)
        style_metric_axis(axis, metric_title, baseline_row[metric_key])
        annotate_bars(axis, bars)

        best_index = dataframe[metric_key].idxmax()
        best_bar = bars[best_index]
        best_bar.set_linewidth(2.5)
        best_bar.set_edgecolor("#111111")
        axis.text(
            best_bar.get_x() + best_bar.get_width() / 2,
            min(best_bar.get_height() + 0.07, 1.04),
            "Best",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111111",
            fontweight="bold",
        )

        for label in axis.get_xticklabels():
            label.set_rotation(0)
            label.set_fontsize(9)

    heatmap_axis = figure.add_subplot(grid[2, :])
    heatmap_frame = dataframe[
        [
            "model_label",
            "val_accuracy",
            "val_f1_macro",
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
        ]
    ].copy()
    heatmap_frame = heatmap_frame.set_index("model_label")
    heatmap_frame.columns = [
        "Val Acc",
        "Val F1",
        "Test Acc",
        "Test Prec",
        "Test Recall",
        "Test F1",
    ]
    sns.heatmap(
        heatmap_frame,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Score"},
        ax=heatmap_axis,
    )
    heatmap_axis.set_title("Validation and Test Metric Summary", fontsize=12, fontweight="bold")
    heatmap_axis.set_xlabel("Metric")
    heatmap_axis.set_ylabel("Algorithm")

    figure.suptitle(
        "Model Performance Comparison Across Benchmark Algorithms\nDashed red line = current deployable MobileNetV2 baseline",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.02,
        "Metrics sourced from outputs/08_model_benchmark_results.json. Higher is better for all displayed metrics.",
        ha="center",
        fontsize=10,
        color="#374151",
    )

    png_path = outputs_dir / "09_model_performance_dashboard.png"
    svg_path = outputs_dir / "09_model_performance_dashboard.svg"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    return png_path, svg_path


def main():
    if not results_path.exists():
        raise FileNotFoundError(f"Benchmark result file not found: {results_path}")

    dataframe = load_results()
    png_path, svg_path = build_dashboard(dataframe)
    print(f"Saved dashboard PNG: {png_path}")
    print(f"Saved dashboard SVG: {svg_path}")


if __name__ == "__main__":
    main()
