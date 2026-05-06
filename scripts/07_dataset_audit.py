import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RGB_SAMPLE_SIZE = (64, 64)


current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent
master_dir = base_dir / "Master_Dataset"
split_dir = base_dir / "Dataset_Split"
outputs_dir = base_dir / "outputs"

outputs_dir.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


def iter_image_paths(root_dir: Path):
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def hash_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_master_dataset_rows():
    rows = []
    for class_dir in sorted([path for path in master_dir.iterdir() if path.is_dir()]):
        for image_path in iter_image_paths(class_dir):
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                sampled = rgb_image.resize(RGB_SAMPLE_SIZE)
                pixels = np.asarray(sampled, dtype=np.float32) / 255.0
                rgb_mean = pixels.mean(axis=(0, 1))

                rows.append(
                    {
                        "class_name": class_dir.name,
                        "relative_path": str(image_path.relative_to(base_dir)),
                        "filename": image_path.name,
                        "format": image.format,
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(width / height, 4) if height else None,
                        "mean_r": float(rgb_mean[0]),
                        "mean_g": float(rgb_mean[1]),
                        "mean_b": float(rgb_mean[2]),
                        "md5": hash_file(image_path),
                    }
                )
    return pd.DataFrame(rows)


def collect_split_duplicate_rows():
    if not split_dir.exists():
        return pd.DataFrame()

    rows = []
    for split_name in sorted([path for path in split_dir.iterdir() if path.is_dir()]):
        for class_dir in sorted([path for path in split_name.iterdir() if path.is_dir()]):
            for image_path in iter_image_paths(class_dir):
                rows.append(
                    {
                        "split": split_name.name,
                        "class_name": class_dir.name,
                        "relative_path": str(image_path.relative_to(base_dir)),
                        "md5": hash_file(image_path),
                    }
                )
    return pd.DataFrame(rows)


def save_class_distribution_chart(dataframe: pd.DataFrame):
    class_counts = (
        dataframe.groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    class_counts["percentage"] = class_counts["count"] / class_counts["count"].sum() * 100

    plt.figure(figsize=(9, 6))
    chart = sns.barplot(data=class_counts, x="class_name", y="count", hue="class_name", palette="viridis", legend=False)
    chart.set_title("Dataset Audit: Class Distribution", fontsize=14, fontweight="bold")
    chart.set_xlabel("Class")
    chart.set_ylabel("Image Count")

    for patch, percentage in zip(chart.patches, class_counts["percentage"]):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        chart.text(x, y + 20, f"{int(y)}\n({percentage:.1f}%)", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(outputs_dir / "07_class_distribution_audit.png", dpi=200)
    plt.close()


def save_resolution_chart(dataframe: pd.DataFrame):
    plt.figure(figsize=(10, 7))
    scatter = sns.scatterplot(
        data=dataframe,
        x="width",
        y="height",
        hue="class_name",
        alpha=0.7,
        s=50,
        palette="Set2",
    )
    scatter.set_title("Dataset Audit: Image Resolution Spread", fontsize=14, fontweight="bold")
    scatter.set_xlabel("Width (px)")
    scatter.set_ylabel("Height (px)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "07_image_resolution_scatter.png", dpi=200)
    plt.close()


def save_aspect_ratio_chart(dataframe: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    chart = sns.boxplot(data=dataframe, x="class_name", y="aspect_ratio", hue="class_name", palette="pastel", legend=False)
    chart.set_title("Dataset Audit: Aspect Ratio by Class", fontsize=14, fontweight="bold")
    chart.set_xlabel("Class")
    chart.set_ylabel("Aspect Ratio (width / height)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "07_aspect_ratio_boxplot.png", dpi=200)
    plt.close()


def save_rgb_mean_chart(dataframe: pd.DataFrame):
    rgb_summary = (
        dataframe.groupby("class_name")[["mean_r", "mean_g", "mean_b"]]
        .mean()
        .reset_index()
        .melt(id_vars="class_name", var_name="channel", value_name="value")
    )

    plt.figure(figsize=(10, 6))
    chart = sns.barplot(data=rgb_summary, x="class_name", y="value", hue="channel", palette=["#d73027", "#1a9850", "#4575b4"])
    chart.set_title("Dataset Audit: Mean RGB Intensity by Class", fontsize=14, fontweight="bold")
    chart.set_xlabel("Class")
    chart.set_ylabel("Normalized Mean Intensity")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outputs_dir / "07_rgb_channel_means.png", dpi=200)
    plt.close()


def summarize_dataset(dataframe: pd.DataFrame, split_duplicates: pd.DataFrame):
    class_counts = dataframe.groupby("class_name").size().sort_values(ascending=False)
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    total_images = int(class_counts.sum())
    class_weights = {
        class_name: round(total_images / (len(class_counts) * count), 4)
        for class_name, count in class_counts.items()
    }

    duplicate_rows = dataframe[dataframe["md5"].duplicated(keep=False)].sort_values(["md5", "relative_path"])
    duplicate_groups = int(duplicate_rows["md5"].nunique())

    split_leakage_rows = pd.DataFrame()
    cross_split_groups = 0
    if not split_duplicates.empty:
        grouped = split_duplicates.groupby("md5").agg(
            duplicate_count=("md5", "size"),
            split_count=("split", "nunique"),
        )
        leak_hashes = grouped[(grouped["duplicate_count"] > 1) & (grouped["split_count"] > 1)].index.tolist()
        split_leakage_rows = split_duplicates[split_duplicates["md5"].isin(leak_hashes)].sort_values(["md5", "split", "relative_path"])
        cross_split_groups = len(leak_hashes)

    summary = {
        "total_images": total_images,
        "classes": class_counts.to_dict(),
        "class_percentages": {key: round(value / total_images * 100, 2) for key, value in class_counts.items()},
        "imbalance_ratio_max_to_min": round(imbalance_ratio, 4),
        "class_weights_balanced": class_weights,
        "image_format_distribution": dataframe["format"].value_counts().to_dict(),
        "resolution": {
            "width_min": int(dataframe["width"].min()),
            "width_max": int(dataframe["width"].max()),
            "width_median": float(dataframe["width"].median()),
            "height_min": int(dataframe["height"].min()),
            "height_max": int(dataframe["height"].max()),
            "height_median": float(dataframe["height"].median()),
        },
        "aspect_ratio": {
            "min": round(float(dataframe["aspect_ratio"].min()), 4),
            "max": round(float(dataframe["aspect_ratio"].max()), 4),
            "median": round(float(dataframe["aspect_ratio"].median()), 4),
        },
        "mean_rgb_by_class": (
            dataframe.groupby("class_name")[["mean_r", "mean_g", "mean_b"]]
            .mean()
            .round(4)
            .to_dict(orient="index")
        ),
        "exact_duplicate_groups_in_master_dataset": duplicate_groups,
        "exact_duplicate_extra_files_in_master_dataset": int(len(duplicate_rows) - duplicate_groups),
        "cross_split_duplicate_groups": cross_split_groups,
        "cross_split_duplicate_extra_files": int(len(split_leakage_rows) - cross_split_groups if cross_split_groups else 0),
    }

    return summary, duplicate_rows, split_leakage_rows


def main():
    if not master_dir.exists():
        raise FileNotFoundError(f"Master dataset folder not found: {master_dir}")

    print("=" * 60)
    print("PHASE 7: DATASET AUDIT & READINESS REVIEW")
    print("=" * 60)

    master_df = collect_master_dataset_rows()
    split_df = collect_split_duplicate_rows()

    master_df.to_csv(outputs_dir / "07_dataset_inventory.csv", index=False)

    summary, duplicate_rows, split_leakage_rows = summarize_dataset(master_df, split_df)

    duplicate_rows.to_csv(outputs_dir / "07_exact_duplicates.csv", index=False)
    split_leakage_rows.to_csv(outputs_dir / "07_cross_split_duplicates.csv", index=False)

    with (outputs_dir / "07_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    save_class_distribution_chart(master_df)
    save_resolution_chart(master_df)
    save_aspect_ratio_chart(master_df)
    save_rgb_mean_chart(master_df)

    print(f"Total images audited : {summary['total_images']}")
    print(f"Class distribution   : {summary['classes']}")
    print(f"Imbalance ratio      : {summary['imbalance_ratio_max_to_min']:.4f}")
    print(f"Exact duplicate sets : {summary['exact_duplicate_groups_in_master_dataset']}")
    print(f"Cross-split leaks    : {summary['cross_split_duplicate_groups']}")
    print(f"Summary saved to     : {outputs_dir / '07_dataset_summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
