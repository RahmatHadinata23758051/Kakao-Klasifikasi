import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# HEADER
md_header = """# 🚀 CONTROL PANEL: KAKAO RIPENESS CLASSIFICATION 
Notebook utama ini berfungsi sebagai panel eksekusi keseluruhan tahapan (*Phase*) proyek, agar proses pengerjaan tertata di satu tempat sesuai standarisasi, namun logikanya tetap berada di modul folder `scripts/`."""
nb.cells.append(nbf.v4.new_markdown_cell(md_header))

# PHASE 1
nb.cells.append(nbf.v4.new_markdown_cell("## Phase 1: Data Consolidation\nMenyatukan dataset mentah dari 3 sumber folder yang berbeda menjadi 1 folder rapi (`Master_Dataset`)."))
nb.cells.append(nbf.v4.new_code_cell("!python scripts/01_data_consolidation.py"))

# PHASE 2
nb.cells.append(nbf.v4.new_markdown_cell("## Phase 2: Exploratory Data Analysis (EDA)\nMensimulasikan perhitungan data hasil gabungan dan mengekstrak visualisasinya ke folder `/outputs`."))
nb.cells.append(nbf.v4.new_code_cell("!python scripts/02_data_understanding.py"))

nb.cells.append(nbf.v4.new_markdown_cell("### Preview Hasil EDA (Phase 2)\nMemuat tampilan gambar (`.png`) yang telah diekspor oleh script ke dalam blok ini agar mudah ditinjau."))
nb.cells.append(nbf.v4.new_code_cell("from IPython.display import Image, display\nprint('Distribusi Kuantitas Kelas:')\ndisplay(Image(filename='outputs/distribusi_kelas.png'))\nprint('Preview Visi Algoritma:')\ndisplay(Image(filename='outputs/sampel_dataset.png'))"))

# PHASE 3
nb.cells.append(nbf.v4.new_markdown_cell("## Phase 3: Preprocessing & Data Splitting\nMembelah dataset di `Master_Dataset` secara proporsional ke dalam Train Set (80%), Val Set (10%), dan Test Set (10%). Outputnya ditaruh ke folder baru bernama `Dataset_Split`."))
nb.cells.append(nbf.v4.new_code_cell("!python scripts/03_preprocessing_split.py"))

nb.cells.append(nbf.v4.new_markdown_cell("### Preview Distribusi Splitting (Phase 3)"))
nb.cells.append(nbf.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename='outputs/03_distribusi_splitting_train_val_test.png'))"))


# Simpan ke root folder proyek
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Kakao_Model_Development.ipynb')
with open(file_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print(f"Buku kerja Jupyter (Kakao_Model_Development.ipynb) berhasil digenerate di {file_path}")
