import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    print("Tensorflow missing")
    exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
split_dir = os.path.join(base_dir, "Dataset_Split")
models_dir = os.path.join(base_dir, "models")
outputs_dir = os.path.join(base_dir, "outputs")

print("="*50)
print("   ANALISIS LANJUTAN: DETEKSI BIAS (ROC-AUC CURVE)")
print("="*50)

# 1. Load Model & Class Mapping
model = tf.keras.models.load_model(os.path.join(models_dir, "best_kakao_model.keras"))
with open(os.path.join(models_dir, "class_indices.json"), "r") as f:
    class_map = json.load(f)
ordered_classes = [class_map[str(i)] for i in range(len(class_map))]
n_classes = len(ordered_classes)

# 2. Test Generator
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    os.path.join(split_dir, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 3. Predict Probabilities
print("\n[Menganalisis Probabilitas Matematika dari setiap kelas...]")
predictions = model.predict(test_generator, verbose=1)
y_true = test_generator.classes

# Binarize output untuk multi-class ROC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

# 4. Compute ROC curve and ROC Area
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 5. Plot ROC Multi-Class
plt.figure(figsize=(9, 7))
colors = cycle(['orange', 'green', 'blue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class: {ordered_classes[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Garis Tebakan Acak (50%)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Tingkat Salah Deteksi)', fontsize=12)
plt.ylabel('True Positive Rate (Tingkat Benar Deteksi)', fontsize=12)
plt.title('Kurva ROC (Receiver Operating Characteristic) - Analisis Bias', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

chart_path = os.path.join(outputs_dir, "05b_roc_curve_bias_check.png")
plt.savefig(chart_path)
plt.close()

print(f"\n[OK] Grafik Analisis Bias (ROC-AUC) telah disimpan ke:\n     {chart_path}")
print("="*50)
