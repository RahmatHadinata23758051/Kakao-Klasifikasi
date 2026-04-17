import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
split_dir = os.path.join(base_dir, "Dataset_Split")
models_dir = os.path.join(base_dir, "models")

# Re-create generator cuma untuk mengekstrak Index
datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
    os.path.join(split_dir, 'train'),
    batch_size=1
)

# Ambil penomoran aslinya
class_map = generator.class_indices

# Balik (Invert) menjadi {0: 'Kematangan', 1: 'Matang', 2: 'Mentah'}
inverted_class_map = {v: k for k, v in class_map.items()}

json_path = os.path.join(models_dir, "class_indices.json")
with open(json_path, "w") as f:
    json.dump(inverted_class_map, f, indent=4)

print("\n[VERIFIKASI BERHASIL] Indeks Pemetaan Kelas telah diamankan!")
print(f"File disimpan di: {json_path}")
print("Isi Kamus Kelas:", inverted_class_map)
