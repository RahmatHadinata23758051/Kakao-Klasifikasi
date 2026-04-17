import os
import matplotlib.pyplot as plt
import numpy as np

# Mengecek apakah splitfolders library sudah terinstal
try:
    import splitfolders
except ImportError:
    print("Harap install split-folders terlebih dahulu dengan: pip install split-folders")
    exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
input_dir = os.path.join(base_dir, "Master_Dataset")
output_dir = os.path.join(base_dir, "Dataset_Split")
charts_dir = os.path.join(base_dir, "outputs")

# Pastikan output direcory exist
os.makedirs(charts_dir, exist_ok=True)

print("="*50)
print("   PHASE 3: PREPROCESSING & DATA SPLITTING")
print("="*50)

print("\n--- [1] MEMULAI DATA SPLITTING (80% Train, 10% Val, 10% Test) ---")
# Using splitfolders to physically partition into train, val, test folders
splitfolders.ratio(input_dir, output=output_dir, seed=42, ratio=(0.8, 0.1, 0.1), group_prefix=None, move=False)
print(f"   [OK] Folder Dataset_Split berhasil di-generate di: \n        {output_dir}")

print("\n--- [2] MENGHITUNG KEMBALI HASIL SPLIT ---")
classes = ["Mentah", "Matang", "Kematangan"]
splits = ["train", "val", "test"]

counts = {split: [] for split in splits}

for split in splits:
    print(f"\n   [{split.upper()} SET]")
    for cls in classes:
        path = os.path.join(output_dir, split, cls)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        counts[split].append(count)
        print(f"   > {cls.ljust(15)} : {count} images")

# Generate Stacked Bar Chart for the visualization output
print("\n--- [3] MEMBUAT VISUALISASI HASIL SPLIT TUMPANG (STACKED) ---")
plt.figure(figsize=(10, 6))

train_counts = np.array(counts["train"])
val_counts = np.array(counts["val"])
test_counts = np.array(counts["test"])
ind = np.arange(len(classes))
width = 0.55

# Membuat chart tumpang tindih
p1 = plt.bar(ind, train_counts, width, color='#3181bd')
p2 = plt.bar(ind, val_counts, width, bottom=train_counts, color='#9ecbe1')
p3 = plt.bar(ind, test_counts, width, bottom=train_counts+val_counts, color='#deebf7')

plt.ylabel('Total Jumlah Gambar', fontsize=12)
plt.title('Distribusi Pemotongan Data (Train 80% | Val 10% | Test 10%)', fontsize=14, fontweight='bold')
plt.xticks(ind, classes, fontsize=11)
plt.legend((p1[0], p2[0], p3[0]), ('Train (80%)', 'Validation (10%)', 'Test (10%)'), bbox_to_anchor=(1.05, 1), loc='upper left')

# Anotasi teks di dalam tiap balok tumpukan
for i in range(len(classes)):
    # Teks train (Paling Bawah)
    if train_counts[i] > 0:
        plt.text(i, train_counts[i]/2, str(train_counts[i]), ha='center', va='center', color='white', fontweight='bold')
    # Teks val (Tengah)
    if val_counts[i] > 0:
        plt.text(i, train_counts[i] + val_counts[i]/2, str(val_counts[i]), ha='center', va='center', color='black')
    # Teks test (Paling Atas)
    if test_counts[i] > 0:
        plt.text(i, train_counts[i] + val_counts[i] + test_counts[i]/2, str(test_counts[i]), ha='center', va='center', color='black')

plt.tight_layout()
chart_path = os.path.join(charts_dir, "03_distribusi_splitting_train_val_test.png")
plt.savefig(chart_path)
plt.close()

print(f"   [OK] Grafik Stacked Bar berhasil disimpan secara informatif ke: \n        {chart_path}")
print("\n" + "="*50)
print("PHASE 3 SELESAI!")
print("="*50)
