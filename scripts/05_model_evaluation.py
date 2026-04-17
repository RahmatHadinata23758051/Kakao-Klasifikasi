import os
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("Mendownload dependencies scikit-learn...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
split_dir = os.path.join(base_dir, "Dataset_Split")
models_dir = os.path.join(base_dir, "models")
outputs_dir = os.path.join(base_dir, "outputs")

print("="*50)
print("   PHASE 5: MODEL EVALUATION (CONFUSION MATRIX)")
print("="*50)

# 1. Load Model and Indices
print("\n--- [1] MEMBACA MODEL & KAMUS KELAS ---")
model_path = os.path.join(models_dir, "best_kakao_model.keras")
model = tf.keras.models.load_model(model_path)

json_path = os.path.join(models_dir, "class_indices.json")
with open(json_path, "r") as f:
    class_map = json.load(f)
    
# Map string indices to list ['Kematangan', 'Matang', 'Mentah'] corresponding to 0, 1, 2
ordered_classes = [class_map[str(i)] for i in range(len(class_map))]

# 2. Menyiapkan Test Generator
print("\n--- [2] MENGAMBIL SOAL DARI DATA UJI (TEST SET) ---")
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    os.path.join(split_dir, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False # PENTING! Agar urutan kunci jawaban (label asli) sinkron dengan output prediksi
)

# 3. Klasifikasi Data Tes (Inference Test)
print("\n--- [3] MODEL SEDANG UJIAN! MENGANALISIS GAMBAR... ---")
predictions = model.predict(test_generator, verbose=1)

# Mengkonversi persentase softmax menjadi 1 label paling dominan (0/1/2)
y_pred = np.argmax(predictions, axis=1) 
y_true = test_generator.classes

# 4. Confusion Matrix & Classification Report
print("\n--- [4] RAPOR LULUS UJIAN (CLASSIFICATION REPORT) ---")
print("-" * 55)
report = classification_report(y_true, y_pred, target_names=ordered_classes)
print(report)
print("-" * 55)

print("\n--- [5] MENGGAMBAR CONFUSION MATRIX DIAGRAM ---")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=ordered_classes, 
            yticklabels=ordered_classes,
            annot_kws={"size": 14}) # Angka di dalam kotak
plt.title('Kakao Classification - Confusion Matrix', fontsize=16, pad=15)
plt.xlabel('Tebakan Model (Predicted Label)', fontsize=12)
plt.ylabel('Kunci Jawaban Asli (True Label)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

chart_path = os.path.join(outputs_dir, "05_confusion_matrix.png")
plt.savefig(chart_path)
plt.close()

print(f"   [OK] Grafik Confusion Matrix telah diekspor ke: \n        {chart_path}")
print("\n" + "="*50)
print("PHASE 5 SELESAI DIEKSEKUSI!")
print("="*50)
