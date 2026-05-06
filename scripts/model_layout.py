from pathlib import Path


current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent
models_root = base_dir / "models"


MODEL_DIRS = {
    "mobilenetv2_baseline": models_root / "mobilenetv2_baseline",
    "mobilenetv2_embeddings": models_root / "mobilenetv2_embeddings",
    "efficientnetb0_embeddings": models_root / "efficientnetb0_embeddings",
    "hog_linear_svm": models_root / "hog_linear_svm",
}


def ensure_model_dir(model_key: str) -> Path:
    model_dir = MODEL_DIRS[model_key]
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def baseline_model_paths():
    model_dir = ensure_model_dir("mobilenetv2_baseline")
    return {
        "dir": model_dir,
        "keras_model": model_dir / "best_kakao_model.keras",
        "tflite_model": model_dir / "model_kakao_optimized.tflite",
        "class_indices": model_dir / "class_indices.json",
        "manifest": model_dir / "artifact_manifest.json",
    }


def benchmark_model_paths(model_key: str):
    model_dir = ensure_model_dir(model_key)
    return {
        "dir": model_dir,
        "classifier": model_dir / "classifier.joblib",
        "feature_extractor_weights": model_dir / "feature_extractor.weights.h5",
        "metadata": model_dir / "metadata.json",
    }
