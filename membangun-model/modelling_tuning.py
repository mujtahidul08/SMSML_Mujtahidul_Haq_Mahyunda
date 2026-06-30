import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow & DagsHub
import mlflow
import mlflow.sklearn
import dagshub

# Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =========================================================================
# 1. KONFIGURASI DAGSHUB & MLFLOW (Ubah sesuai akun Anda)
# =========================================================================
DAGSHUB_USERNAME = "mujahaq"  
REPO_NAME = "SMSML_Mujtahidul_Haq_Mahyunda"

print("[INFO] Menginisialisasi koneksi DagsHub...")
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)

# Set nama eksperimen di MLflow
mlflow.set_experiment("Mental_Health_Sentiment_Tuning")

# =========================================================================
# 2. LOAD DATA & PREPARATION
# =========================================================================
print("[INFO] Membaca dataset bersih...")

# Mendapatkan path root repository (satu tingkat di atas folder saat ini)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Mengarahkan langsung ke folder preprocessing dan nama file sesuai gambar Anda
data_path = os.path.join(root_dir, "preprocessing", "data_clean_mental_health_indo1.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File tidak ditemukan di: {data_path}. Pastikan nama file sesuai!")

df = pd.read_csv(data_path)

# Pastikan tidak ada nilai NaN di kolom teks hasil preprocessing
df['text_clean_automation'] = df['text_akhir'].fillna('')

X = df['text_clean_automation']
y = df['label']

# Split data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vektorasi TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# =========================================================================
# 3. HYPERPARAMETER TUNING & MANUAL LOGGING
# =========================================================================
print("[INFO] Memulai Hyperparameter Tuning dengan GridSearchCV...")

# Definisikan model dan parameter yang akan di-tuning
model_base = LogisticRegression(max_iter=1000, random_state=42)
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(model_base, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"[INFO] Parameter Terbaik: {best_params}")

# Mulai run MLflow untuk mencatat model terbaik
with mlflow.start_run(run_name="LogisticRegression_Best_Tuning"):
    
    # 1. Log Hyperparameters (Manual Logging)
    print("[LOG] Mencatat parameter ke MLflow...")
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
    mlflow.log_param("vectorizer_max_features", 5000)
    
    # Prediksi
    y_pred = best_model.predict(X_test)
    
    # Evaluasi Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # 2. Log Metrics (Manual Logging)
    print("[LOG] Mencatat metrik evaluasi...")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # 3. Log Artefak Tambahan Minimal 2 (Kriteria Advance)
    print("[LOG] Membuat dan mencatat artefak tambahan...")
    
    # Artefak 1: Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Best Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    # Unggah plot ke MLflow sebagai artefak
    mlflow.log_artifact(cm_path)
    
    # Artefak 2: Classification Report Text File
    report_dict = classification_report(y_test, y_pred)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_dict)
        
    # Unggah file teks ke MLflow sebagai artefak
    mlflow.log_artifact(report_path)
    
    # 4. Log Model Utama
    print("[LOG] Menyimpan artefak model ke MLflow...")
    mlflow.sklearn.log_model(best_model, "model_sentimen_terbaik")
    
    # Bersihkan file lokal setelah diunggah ke cloud dagsHub
    if os.path.exists(cm_path): os.remove(cm_path)
    if os.path.exists(report_path): os.remove(report_path)

print("[SUCCESS] Seluruh proses tuning dan manual logging ke DagsHub selesai!")