import os
import matplotlib.pyplot as plt

# Handle TensorFlow import gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
except ImportError:
    print("TensorFlow belum diinstal. System akan menginstall library terlebih dahulu.")
    exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, ".."))
split_dir = os.path.join(base_dir, "Dataset_Split")
models_dir = os.path.join(base_dir, "models")
outputs_dir = os.path.join(base_dir, "outputs")

os.makedirs(models_dir, exist_ok=True)

print("="*50)
print("   PHASE 4: MODEL DEVELOPMENT & TRAINING")
print("="*50)

# 1. Menyiapkan Data Generator
print("\n--- [1] MENYIAPKAN DATA GENERATOR & AUGMENTASI ---")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Augmentasi dataset untuk pelatihan (mencegah overfitting dan memperkaya data Kematangan)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validasi tidak boleh di-augmentasi aneh-aneh, cukup scale inputnya!
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 2. Membangun Arsitektur Model MobileNetV2
print("\n--- [2] MEMBANGUN ARSITEKTUR MOBILENET-V2 (Pre-trained) ---")
# include_top=False membuang lapis ujung imagenet 1000-class
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze base model terlebih dahulu agar proses latih PC kamu tidak berat
base_model.trainable = False 

# Modifikasi Ujung Jaringan (Classifier Head Khusus Kakao)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x) # Mengurangi overfitting
predictions = Dense(3, activation='softmax')(x) # 3 Kelas Klasifikasi

model = Model(inputs=base_model.input, outputs=predictions)

# Kompilasi dengan Adam
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Tampilkan info struktur layer
print(f"Total layer dalam model baru: {len(model.layers)}")

# 3. Training & Callbacks
print("\n--- [3] MEMULAI TRAINING MODEL (BASELINE: 5 EPOCH) ---")
checkpoint_path = os.path.join(models_dir, "best_kakao_model.keras")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Sebagai permulaan let's go for 5 Epochs agar bisa dieksperimen segera
EPOCHS = 5
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# 4. Evaluasi & Visualisasi Training
print("\n--- [4] MENYIMPAN GRAFIK TRAINING ---")
plt.figure(figsize=(12, 5))

# Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val (Validation) Accuracy', marker='s')
plt.title('Kurva Belajar: Akurasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.ylim([0, 1.1])
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val (Validation) Loss', marker='s')
plt.title('Kurva Kesalahan: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
chart_path = os.path.join(outputs_dir, "04_training_history.png")
plt.savefig(chart_path)
plt.close()

print(f"   [OK] Laporan Kurva Training (.png) telah diterbitkan di:\n        {chart_path}")
print(f"   [OK] File Beban Timbangan (Weights) Model terbaik disimpan di:\n        {checkpoint_path}")
print("\n" + "="*50)
print("PHASE 4 SCRIPT SELESAI DIEKSEKUSI!")
print("="*50)
