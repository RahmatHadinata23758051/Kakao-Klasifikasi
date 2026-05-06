import os
import sys
import json
from model_layout import baseline_model_paths

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow missing")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
baseline_paths = baseline_model_paths()

model_path = str(baseline_paths["keras_model"])

print("="*50)
print("   PHASE 6: EXPORT & DEPLOYMENT PREPARATION")
print("="*50)

print("\n--- [1] MEMBACA MODEL KERAS TERBAIK ---")
try:
    model = tf.keras.models.load_model(model_path)
    print("   [OK] Otak Artificial (Model Keras) berhasil dimuat.")
except Exception as e:
    print(f"   [ERROR] Gagal memuat model: {e}")
    sys.exit(1)

print("\n--- [2] MENGONVERSI KE FORMAT TFLITE (Siap Mobile) ---")
# Proses packing model algoritma python menjadi file silikon baku TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Mengaktifkan Opsi Optimisasi (Kuantisasi Timbangan/Weights dari float32 -> int8 dinamis)
# Membuat ukuran model menyusut sampai 4x lebih ringan & cepat bagi RAM HP
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_path = str(baseline_paths["tflite_model"])
try:
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"   [OK] Model TFLite berhasil dipadatkan dan diekstrak ke: \n        {tflite_path}")
except Exception as e:
    if os.path.exists(tflite_path):
        print(f"   [WARNING] Konversi TFLite gagal, artefak lama dipertahankan: {e}")
    else:
        print(f"   [ERROR] Konversi TFLite gagal dan belum ada artefak cadangan: {e}")
        sys.exit(1)

manifest = {
    "algorithm": "MobileNetV2 baseline",
    "artifacts": {
        "keras_model": str(baseline_paths["keras_model"].relative_to(base_dir)),
        "tflite_model": str(baseline_paths["tflite_model"].relative_to(base_dir)),
        "class_indices": str(baseline_paths["class_indices"].relative_to(base_dir)),
    },
}
with open(baseline_paths["manifest"], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

print("\n" + "="*50)
print("SEMUA FASE SELESAI! MODEL SIAP DISERAHKAN KE TIM PROGRAMMER APP.")
print("="*50)
