import os
import json
from model_layout import baseline_model_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
split_dir = os.path.join(base_dir, "Dataset_Split")
baseline_paths = baseline_model_paths()

# Ambil label dari nama folder train agar tidak bergantung pada runtime generator TensorFlow.
train_dir = os.path.join(split_dir, "train")
class_names = sorted(
    [
        entry
        for entry in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, entry))
    ]
)
class_map = {class_name: index for index, class_name in enumerate(class_names)}

# Balik (Invert) menjadi {0: 'Kematangan', 1: 'Matang', 2: 'Mentah'}
inverted_class_map = {v: k for k, v in class_map.items()}

json_path = str(baseline_paths["class_indices"])
with open(json_path, "w") as f:
    json.dump(inverted_class_map, f, indent=4)

print("\n[VERIFIKASI BERHASIL] Indeks Pemetaan Kelas telah diamankan!")
print(f"File disimpan di: {json_path}")
print("Isi Kamus Kelas:", inverted_class_map)
