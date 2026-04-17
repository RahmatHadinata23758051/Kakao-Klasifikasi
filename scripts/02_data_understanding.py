import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
master_dir = os.path.join(base_dir, "Master_Dataset")
classes = ["Mentah", "Matang", "Kematangan"]

print("="*50)
print("   DATA UNDERSTANDING & Exploratory Data Analysis")
print("="*50)

# 1. Menghitung Jumlah Data
print("\n--- [1] MENGHITUNG DISTRIBUSI DATA KELAS ---")
data_counts = {}
for cls in classes:
    path = os.path.join(master_dir, cls)
    if os.path.exists(path):
        data_counts[cls] = len(os.listdir(path))
    else:
        data_counts[cls] = 0
    print(f"   > Kelas {cls.upper().ljust(15)} : {data_counts[cls]} images")

total_images = sum(data_counts.values())
print(f"   > TOTAL KESELURUHAN   : {total_images} images")

# 2. Membuat Bar Chart
print("\n--- [2] MEMBUAT BAR CHART (DIAGRAM BATANG) ---")
plt.figure(figsize=(8, 6))
sns.barplot(x=list(data_counts.keys()), y=list(data_counts.values()), palette="viridis")
plt.title("Distribusi Data Kakao (Mentah vs Matang vs Kematangan)")
plt.xlabel("Kategori kematangan")
plt.ylabel("Jumlah Gambar")
for i, v in enumerate(data_counts.values()):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')

chart_path = os.path.join(base_dir, "outputs", "distribusi_kelas.png")
plt.savefig(chart_path)
plt.close()
print(f"   [OK] Diagram batang telah diekspor ke: \n        {chart_path}")

# 3. Membuat Collage Random Sample (Grid 3x5)
print("\n--- [3] MEMBUAT GRID/KOLASE FOTO ACAK ---")
num_samples_per_class = 5
fig, axes = plt.subplots(3, num_samples_per_class, figsize=(15, 9))
fig.suptitle('Sampel Acak Dataset Kakao (Variasi Kondisi)', fontsize=18, fontweight='bold')

for row, cls in enumerate(classes):
    cls_path = os.path.join(master_dir, cls)
    if os.path.exists(cls_path):
        all_imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        # Pilih max 5 gambar acak
        sampled_imgs = random.sample(all_imgs, min(num_samples_per_class, len(all_imgs)))
        
        for col, img_name in enumerate(sampled_imgs):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{cls}\n{img_name[:12]}...", fontsize=10)
            except Exception as e:
                print(f"Error opening {img_name}: {e}")
            axes[row, col].axis('off')
            
        # Kosongkan sisa axis jika jumlah sampel < 5 (walaupun ini jarang terjadi pada case mu)
        for col in range(len(sampled_imgs), num_samples_per_class):
            axes[row, col].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
collage_path = os.path.join(base_dir, "outputs", "sampel_dataset.png")
plt.savefig(collage_path)
plt.close()
print(f"   [OK] Kolase foto telah diekspor ke: \n        {collage_path}")

print("\n" + "="*50)
print("PROSES EDA SELESAI!\nFile .png siap untuk dilampirkan ke presentasi.")
print("="*50)
