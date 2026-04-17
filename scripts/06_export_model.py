import os
import sys

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow missing")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
models_dir = os.path.join(base_dir, "models")

model_path = os.path.join(models_dir, "best_kakao_model.keras")

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
tflite_model = converter.convert()

tflite_path = os.path.join(models_dir, "model_kakao_optimized.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
    
print(f"   [OK] Model TFLite berhasil dipadatkan dan diekstrak ke: \n        {tflite_path}")

print("\n" + "="*50)
print("SEMUA FASE SELESAI! MODEL SIAP DISERAHKAN KE TIM PROGRAMMER APP.")
print("="*50)
