import os
import pandas as pd
import numpy as np

# MLflow
import mlflow
import mlflow.sklearn

# Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =========================================================================
# 1. INISIALISASI MLFLOW AUTOLOG (Kriteria Basic)
# =========================================================================
# Mengaktifkan autolog secara global untuk scikit-learn sebelum model dilatih.
# Ini akan mencatat parameter default, metrik, dan model secara otomatis.
mlflow.sklearn.autolog()

# Set nama eksperimen lokal
mlflow.set_experiment("Mental_Health_Sentiment_Local_Basic")

# =========================================================================
# 2. LOAD DATA & PREPARATION
# =========================================================================
print("[INFO] Membaca dataset bersih untuk model dasar...")

# Solusi Dinamis: Mengambil path folder tempat file modelling.py ini berada
print("[INFO] Membaca dataset bersih...")

# Mendapatkan path root repository (satu tingkat di atas folder saat ini)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Mengarahkan langsung ke folder preprocessing dan nama file sesuai gambar Anda
data_path = os.path.join(root_dir, "preprocessing", "data_clean_mental_health_indo.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File tidak ditemukan di: {data_path}. Pastikan nama file sesuai!")

df = pd.read_csv(data_path)

# Mengatasi nilai kosong jika ada
df['text_clean_automation'] = df['text_akhir'].fillna('')

X = df['text_clean_automation']
y = df['label']

# Split data (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vektorasi TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# =========================================================================
# 3. TRAINING MODEL DENGAN MLFLOW RUN
# =========================================================================
print("[INFO] Memulai training model dengan MLflow Autolog...")

with mlflow.start_run(run_name="LogisticRegression_Local_Autolog"):
    
    # Inisialisasi model standar (tanpa tuning parameter)
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Proses fitting/training
    # Karena autolog aktif, MLflow otomatis merekam parameter C, penalty, dll.
    model.fit(X_train, y_train)
    
    # Evaluasi sederhana di terminal
    y_pred = model.predict(X_test)
    print("\n[HASIL EVALUASI MODEL DASAR]:")
    print(classification_report(y_test, y_pred))

print("[SUCCESS] Training selesai! Riwayat disimpan di folder lokal 'mlruns'.")
print("[INFO] Jalankan perintah 'mlflow ui' di terminal untuk melihat dashboard lokal Anda.")