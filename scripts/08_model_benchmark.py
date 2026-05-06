import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from model_layout import baseline_model_paths, benchmark_model_paths
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSICAL_IMAGE_SIZE = (128, 128)
DEEP_IMAGE_SIZE = (224, 224)
DEEP_BATCH_SIZE = 32


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
sns.set_theme(style="whitegrid")


current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent
split_dir = base_dir / "Dataset_Split"
outputs_dir = base_dir / "outputs"
cache_dir = outputs_dir / "cache"

outputs_dir.mkdir(exist_ok=True)
cache_dir.mkdir(exist_ok=True)


@dataclass
class SplitRecords:
    name: str
    dataframe: pd.DataFrame


def list_split_records(split_name: str, class_names):
    split_path = split_dir / split_name
    rows = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = split_path / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "path": image_path,
                        "label_idx": label_idx,
                        "label_name": class_name,
                    }
                )
    return SplitRecords(name=split_name, dataframe=pd.DataFrame(rows))


def discover_class_names():
    train_path = split_dir / "train"
    return sorted([path.name for path in train_path.iterdir() if path.is_dir()])


def compute_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_confusion_matrix(y_true, y_pred, class_names, output_path: Path, title: str):
    matrix = confusion_matrix(y_true, y_pred)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_path.with_suffix(".csv"))
    plt.figure(figsize=(8, 6))
    chart = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    chart.set_title(title, fontsize=14, fontweight="bold")
    chart.set_xlabel("Predicted")
    chart.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_image_rgb(path: Path, target_size):
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(target_size)
        return np.asarray(resized)


def extract_hist_hog_features(records: SplitRecords):
    feature_rows = []
    for image_path in records.dataframe["path"]:
        image = load_image_rgb(image_path, CLASSICAL_IMAGE_SIZE)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        hist_features = []
        for channel_idx, bins in enumerate((16, 16, 16)):
            histogram = cv2.calcHist([hsv_image], [channel_idx], None, [bins], [0, 256]).flatten()
            histogram = histogram / (histogram.sum() + 1e-8)
            hist_features.append(histogram)
        hist_features = np.concatenate(hist_features)

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feature_rows.append(np.concatenate([hist_features, hog_features]))
    return np.vstack(feature_rows)


def cache_feature_path(model_slug: str, split_name: str):
    return cache_dir / f"{model_slug}_{split_name}.npz"


def cache_classical_feature_path(model_slug: str, split_name: str):
    return cache_dir / f"{model_slug}_{split_name}_classical.npz"


def extract_deep_features(records: SplitRecords, model_slug: str, feature_model, preprocess_fn):
    cache_path = cache_feature_path(model_slug, records.name)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["features"]

    paths = records.dataframe["path"].tolist()
    all_features = []

    for batch_start in range(0, len(paths), DEEP_BATCH_SIZE):
        batch_paths = paths[batch_start : batch_start + DEEP_BATCH_SIZE]
        batch_images = np.stack([load_image_rgb(path, DEEP_IMAGE_SIZE) for path in batch_paths]).astype(np.float32)
        batch_images = preprocess_fn(batch_images)
        batch_features = feature_model.predict(batch_images, verbose=0)
        all_features.append(batch_features)

    features = np.vstack(all_features)
    np.savez_compressed(cache_path, features=features)
    return features


def extract_classical_features(records: SplitRecords, model_slug: str):
    cache_path = cache_classical_feature_path(model_slug, records.name)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["features"]

    features = extract_hist_hog_features(records)
    np.savez_compressed(cache_path, features=features)
    return features


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def save_classical_artifacts(model_slug, classifier, class_names, selected_hyperparameters):
    artifact_paths = benchmark_model_paths(model_slug)
    joblib.dump(classifier, artifact_paths["classifier"])
    metadata = {
        "model_slug": model_slug,
        "algorithm_family": "Classical ML",
        "classifier_type": type(classifier.named_steps["classifier"]).__name__,
        "class_names": class_names,
        "selected_hyperparameters": selected_hyperparameters,
        "preprocessing": {
            "image_size": list(CLASSICAL_IMAGE_SIZE),
            "color_space": "HSV histogram",
            "histogram_bins": [16, 16, 16],
            "hog": {
                "orientations": 9,
                "pixels_per_cell": [16, 16],
                "cells_per_block": [2, 2],
                "block_norm": "L2-Hys",
            },
        },
    }
    save_json(artifact_paths["metadata"], metadata)


def save_embedding_artifacts(model_slug, feature_model, classifier, class_names, selected_hyperparameters, backbone_name, preprocess_name):
    artifact_paths = benchmark_model_paths(model_slug)
    if not artifact_paths["feature_extractor_weights"].exists():
        feature_model.save_weights(artifact_paths["feature_extractor_weights"])
    joblib.dump(classifier, artifact_paths["classifier"])
    metadata = {
        "model_slug": model_slug,
        "algorithm_family": "Transfer Learning + Linear Classifier",
        "backbone": backbone_name,
        "preprocess_function": preprocess_name,
        "class_names": class_names,
        "input_size": list(DEEP_IMAGE_SIZE),
        "selected_hyperparameters": selected_hyperparameters,
        "artifacts": {
            "feature_extractor_weights": str(artifact_paths["feature_extractor_weights"].relative_to(base_dir)),
            "classifier": str(artifact_paths["classifier"].relative_to(base_dir)),
        },
    }
    save_json(artifact_paths["metadata"], metadata)


def save_baseline_manifest(class_names):
    baseline_paths = baseline_model_paths()
    manifest = {
        "model_slug": "mobilenetv2_saved_keras",
        "algorithm_family": "End-to-End Transfer Learning",
        "backbone": "MobileNetV2",
        "class_names": class_names,
        "artifacts": {
            "keras_model": str(baseline_paths["keras_model"].relative_to(base_dir)),
            "tflite_model": str(baseline_paths["tflite_model"].relative_to(base_dir)),
            "class_indices": str(baseline_paths["class_indices"].relative_to(base_dir)),
        },
    }
    save_json(baseline_paths["manifest"], manifest)


def evaluate_predictions(model_name, split_name, y_true, y_pred, class_names):
    metrics = compute_metrics(y_true, y_pred)
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        outputs_dir / f"08_{model_name}_{split_name}_confusion_matrix.png",
        f"{model_name.replace('_', ' ').title()} - {split_name.title()} Confusion Matrix",
    )
    return metrics


def run_hog_svm(train_records, val_records, test_records, class_names):
    print("[1/4] Benchmarking HOG + Linear SVM")
    start_time = perf_counter()

    x_train = extract_classical_features(train_records, "hog_linear_svm")
    x_val = extract_classical_features(val_records, "hog_linear_svm")
    x_test = extract_classical_features(test_records, "hog_linear_svm")

    y_train = train_records.dataframe["label_idx"].to_numpy()
    y_val = val_records.dataframe["label_idx"].to_numpy()
    y_test = test_records.dataframe["label_idx"].to_numpy()

    best_model = None
    best_c = None
    best_val_score = -1

    for candidate_c in (0.1, 1.0, 3.0):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(C=candidate_c, class_weight="balanced", random_state=SEED, max_iter=15000)),
            ]
        )
        pipeline.fit(x_train, y_train)
        val_pred = pipeline.predict(x_val)
        val_metrics = compute_metrics(y_val, val_pred)
        if val_metrics["f1_macro"] > best_val_score:
            best_val_score = val_metrics["f1_macro"]
            best_model = pipeline
            best_c = candidate_c

    val_pred = best_model.predict(x_val)
    test_pred = best_model.predict(x_test)
    save_classical_artifacts("hog_linear_svm", best_model, class_names, {"C": best_c})

    runtime_sec = perf_counter() - start_time
    return {
        "model_slug": "hog_linear_svm",
        "display_name": "HSV Histogram + HOG + Linear SVM",
        "family": "Classical ML",
        "selected_hyperparameters": {"C": best_c},
        "runtime_sec": round(runtime_sec, 2),
        "val_metrics": evaluate_predictions("hog_linear_svm", "val", y_val, val_pred, class_names),
        "test_metrics": evaluate_predictions("hog_linear_svm", "test", y_test, test_pred, class_names),
    }


def build_feature_extractor(model_slug: str):
    if model_slug == "mobilenetv2_embeddings":
        return MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)), mobilenet_preprocess
    if model_slug == "efficientnetb0_embeddings":
        return EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)), efficientnet_preprocess
    raise ValueError(f"Unsupported model slug: {model_slug}")


def run_embedding_logreg(model_slug, display_name, train_records, val_records, test_records, class_names):
    print(f"Benchmarking {display_name}")
    start_time = perf_counter()

    feature_model, preprocess_fn = build_feature_extractor(model_slug)

    x_train = extract_deep_features(train_records, model_slug, feature_model, preprocess_fn)
    x_val = extract_deep_features(val_records, model_slug, feature_model, preprocess_fn)
    x_test = extract_deep_features(test_records, model_slug, feature_model, preprocess_fn)

    y_train = train_records.dataframe["label_idx"].to_numpy()
    y_val = val_records.dataframe["label_idx"].to_numpy()
    y_test = test_records.dataframe["label_idx"].to_numpy()

    best_model = None
    best_c = None
    best_val_score = -1

    for candidate_c in (0.1, 1.0, 3.0, 10.0):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=candidate_c,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
        val_pred = pipeline.predict(x_val)
        val_metrics = compute_metrics(y_val, val_pred)
        if val_metrics["f1_macro"] > best_val_score:
            best_val_score = val_metrics["f1_macro"]
            best_model = pipeline
            best_c = candidate_c

    val_pred = best_model.predict(x_val)
    test_pred = best_model.predict(x_test)
    save_embedding_artifacts(
        model_slug,
        feature_model,
        best_model,
        class_names,
        {"C": best_c},
        feature_model.name,
        preprocess_fn.__name__,
    )

    runtime_sec = perf_counter() - start_time
    return {
        "model_slug": model_slug,
        "display_name": display_name,
        "family": "Transfer Learning + Linear Classifier",
        "selected_hyperparameters": {"C": best_c},
        "runtime_sec": round(runtime_sec, 2),
        "val_metrics": evaluate_predictions(model_slug, "val", y_val, val_pred, class_names),
        "test_metrics": evaluate_predictions(model_slug, "test", y_test, test_pred, class_names),
    }


def predict_saved_keras(model, records: SplitRecords):
    paths = records.dataframe["path"].tolist()
    predictions = []
    for batch_start in range(0, len(paths), DEEP_BATCH_SIZE):
        batch_paths = paths[batch_start : batch_start + DEEP_BATCH_SIZE]
        batch_images = np.stack([load_image_rgb(path, DEEP_IMAGE_SIZE) for path in batch_paths]).astype(np.float32)
        batch_images = mobilenet_preprocess(batch_images)
        batch_predictions = model.predict(batch_images, verbose=0)
        predictions.append(np.argmax(batch_predictions, axis=1))
    return np.concatenate(predictions)


def run_saved_mobilenet(train_records, val_records, test_records, class_names):
    print("Benchmarking saved MobileNetV2 Keras model")
    start_time = perf_counter()

    baseline_paths = baseline_model_paths()
    model_path = baseline_paths["keras_model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    y_val = val_records.dataframe["label_idx"].to_numpy()
    y_test = test_records.dataframe["label_idx"].to_numpy()

    val_pred = predict_saved_keras(model, val_records)
    test_pred = predict_saved_keras(model, test_records)
    save_baseline_manifest(class_names)

    runtime_sec = perf_counter() - start_time
    baseline_source = str(model_path.relative_to(base_dir))
    return {
        "model_slug": "mobilenetv2_saved_keras",
        "display_name": "Saved MobileNetV2 Keras Head",
        "family": "End-to-End Transfer Learning",
        "selected_hyperparameters": {"source": baseline_source},
        "runtime_sec": round(runtime_sec, 2),
        "val_metrics": evaluate_predictions("mobilenetv2_saved_keras", "val", y_val, val_pred, class_names),
        "test_metrics": evaluate_predictions("mobilenetv2_saved_keras", "test", y_test, test_pred, class_names),
    }


def save_comparison_table(results):
    rows = []
    for result in results:
        row = {
            "model": result["display_name"],
            "family": result["family"],
            "runtime_sec": result["runtime_sec"],
            "val_accuracy": result["val_metrics"]["accuracy"],
            "val_f1_macro": result["val_metrics"]["f1_macro"],
            "test_accuracy": result["test_metrics"]["accuracy"],
            "test_precision_macro": result["test_metrics"]["precision_macro"],
            "test_recall_macro": result["test_metrics"]["recall_macro"],
            "test_f1_macro": result["test_metrics"]["f1_macro"],
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows).sort_values(["test_f1_macro", "test_accuracy"], ascending=False)
    comparison_df.to_csv(outputs_dir / "08_model_comparison.csv", index=False)

    markdown_table = comparison_df.copy()
    for column in [
        "val_accuracy",
        "val_f1_macro",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
    ]:
        markdown_table[column] = markdown_table[column].map(lambda value: f"{value:.4f}")
    markdown_table["runtime_sec"] = markdown_table["runtime_sec"].map(lambda value: f"{value:.2f}")

    markdown_lines = [
        "| Model | Family | Runtime (s) | Val Accuracy | Val F1 Macro | Test Accuracy | Test Precision Macro | Test Recall Macro | Test F1 Macro |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in markdown_table.iterrows():
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["family"]),
                    str(row["runtime_sec"]),
                    str(row["val_accuracy"]),
                    str(row["val_f1_macro"]),
                    str(row["test_accuracy"]),
                    str(row["test_precision_macro"]),
                    str(row["test_recall_macro"]),
                    str(row["test_f1_macro"]),
                ]
            )
            + " |"
        )
    with (outputs_dir / "08_model_comparison.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines))

    plt.figure(figsize=(12, 6))
    chart_data = comparison_df.melt(
        id_vars="model",
        value_vars=["test_accuracy", "test_f1_macro"],
        var_name="metric",
        value_name="score",
    )
    chart = sns.barplot(data=chart_data, x="model", y="score", hue="metric", palette=["#355070", "#6d597a"])
    chart.set_title("Model Benchmark: Test Accuracy vs Macro F1", fontsize=14, fontweight="bold")
    chart.set_xlabel("Model")
    chart.set_ylabel("Score")
    chart.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(outputs_dir / "08_model_comparison.png", dpi=200)
    plt.close()

    return comparison_df


def main():
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dataset folder not found: {split_dir}")

    class_names = discover_class_names()
    train_records = list_split_records("train", class_names)
    val_records = list_split_records("val", class_names)
    test_records = list_split_records("test", class_names)

    print("=" * 60)
    print("PHASE 8: MULTI-MODEL BENCHMARK")
    print("=" * 60)
    print(f"Classes        : {class_names}")
    print(f"Train/Val/Test : {len(train_records.dataframe)}/{len(val_records.dataframe)}/{len(test_records.dataframe)}")

    results = [
        run_hog_svm(train_records, val_records, test_records, class_names),
        run_embedding_logreg(
            "mobilenetv2_embeddings",
            "MobileNetV2 Embeddings + Logistic Regression",
            train_records,
            val_records,
            test_records,
            class_names,
        ),
        run_embedding_logreg(
            "efficientnetb0_embeddings",
            "EfficientNetB0 Embeddings + Logistic Regression",
            train_records,
            val_records,
            test_records,
            class_names,
        ),
        run_saved_mobilenet(train_records, val_records, test_records, class_names),
    ]

    with (outputs_dir / "08_model_benchmark_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    comparison_df = save_comparison_table(results)

    print(comparison_df.to_string(index=False))
    print(f"Benchmark results saved to: {outputs_dir / '08_model_benchmark_results.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
