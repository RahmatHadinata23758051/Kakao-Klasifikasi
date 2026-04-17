import os
import shutil
import glob

# Try importing PIL for image verification, handle nicely if missing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow (PIL) is not installed. Data cleaning (verifikasi image corrupted) hanya akan cek ekstensi.")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
output_dir = os.path.join(base_dir, "Master_Dataset")

classes = ["Mentah", "Matang", "Kematangan"]
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

def verify_and_copy(src_path, dest_dir, prefix_target=""):
    """
    Copy files from Source to Target Destination.
    Verifies if image is corrupt (if PIL is installed).
    Renames the file slightly to avoid naming collisions when merging.
    """
    if not os.path.exists(src_path): return
    filename = os.path.basename(src_path)
    
    # Clean check: Ensure file ends with typical image extensions
    lower_name = filename.lower()
    if not lower_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        return
        
    if prefix_target:
        new_name = prefix_target + "_" + filename
    else:
        new_name = filename
        
    dest_path = os.path.join(dest_dir, new_name)
    
    try:
        # Verify image integrity if PIL is available
        if HAS_PIL:
            with Image.open(src_path) as img:
                img.verify()
        
        # If valid (or if PIL skips check), copy file
        shutil.copy2(src_path, dest_path)
    except Exception as e:
        print(f"Skipping corrupted/invalid file: {filename} - {e}")

print("--- PHASE 1: DATA CONSOLIDATION STARTED ---")

# 1.1 Roboflow Dataset
print("\n[OK] Processing Dataset 1 (Roboflow)...")
robo_base = os.path.join(base_dir, "Cocoa Maturity Dataset Filtered.v2i.folder", "train")
for folder, cls in [("C1", "Mentah"), ("C2", "Mentah"), ("C3", "Matang"), ("C4", "Kematangan")]:
    src_folder = os.path.join(robo_base, folder)
    if os.path.exists(src_folder):
        count = 0
        for img_file in glob.glob(os.path.join(src_folder, "*.*")):
            verify_and_copy(img_file, os.path.join(output_dir, cls), prefix_target="DS1_" + folder)
            count += 1
        print(f"    - Extracted {count} valid files from {folder} to {cls}")

# 1.2 TCS 01 Dataset
print("\n[OK] Processing Dataset 2 (TCS 01 Cocoa Ripeness)...")
tcs_base = os.path.join(base_dir, "Dataset", "Cocoa Ripeness Dataset")
map_tcs = {"I": "Mentah", "M": "Matang", "S": "Kematangan"}
if os.path.exists(tcs_base):
    count_tcs = {"Mentah": 0, "Matang": 0, "Kematangan": 0}
    for img_file in glob.glob(os.path.join(tcs_base, "*.*")):
        basename = os.path.basename(img_file)
        if basename.startswith("I"): 
            verify_and_copy(img_file, os.path.join(output_dir, "Mentah"), prefix_target="DS2")
            count_tcs["Mentah"] += 1
        elif basename.startswith("M"): 
            verify_and_copy(img_file, os.path.join(output_dir, "Matang"), prefix_target="DS2")
            count_tcs["Matang"] += 1
        elif basename.startswith("S"): 
            verify_and_copy(img_file, os.path.join(output_dir, "Kematangan"), prefix_target="DS2")
            count_tcs["Kematangan"] += 1
    print(f"    - Extracted {count_tcs['Mentah']} Mentah, {count_tcs['Matang']} Matang, {count_tcs['Kematangan']} Kematangan.")

# 1.3 CocoaFMDB
print("\n[OK] Processing Dataset 3 (CocoaFMDB)...")
fmdb_base = os.path.join(base_dir, "CocoaFMDB")
for stage, cls in [("unmature", "Mentah"), ("mature", "Matang")]:
    stage_dir = os.path.join(fmdb_base, stage)
    if os.path.exists(stage_dir):
        count_fmdb = 0
        for var_folder in os.listdir(stage_dir):
            var_path = os.path.join(stage_dir, var_folder)
            if os.path.isdir(var_path):
                for img_file in glob.glob(os.path.join(var_path, "*.*")):
                    verify_and_copy(img_file, os.path.join(output_dir, cls), prefix_target=f"DS3_{var_folder}")
                    count_fmdb += 1
        print(f"    - Extracted {count_fmdb} files from {stage} to {cls}")

print("\n--- PHASE 1 SUMMARY (Master_Dataset Counts) ---")
total_count = 0
for cls in classes:
    cnt = len(os.listdir(os.path.join(output_dir, cls)))
    total_count += cnt
    print(f"Total {cls}: {cnt} images")
print(f"GRAND TOTAL: {total_count} images consolidated successfully.")
