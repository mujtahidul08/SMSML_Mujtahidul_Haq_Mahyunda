import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import saka
# Pastikan resource NLTK terunduh saat file ini di-import/dijalankan
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

# Inisialisasi Sastrawi sekali saja di luar fungsi agar hemat memori
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def text_preprocessing_pipeline(text):
    """
    Fungsi untuk membersihkan satu baris teks mentah (Sama persis dengan eksperimen).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Cleaning Teks
    text = re.sub(r'[\(\)\-\,\.\?\!\\\/]', ' ', text)
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#[A-Za-z0-9]+', '', text) 
    text = re.sub(r'RT[\s]', '', text) 
    text = re.sub(r"http\S+", '', text) 
    text = re.sub(r'[0-9]+', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.replace('\n', ' ') 
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = text.strip(' ') 
    
    # 2. Case Folding
    text = text.lower()
    
    # 3. Normalisasi Slangword (Saka-NLP)
    try:
        text = saka.normalize(text)
    except:
        pass
        
    # 4. Stemming (Sastrawi)
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    text = ' '.join(stemmed_words)
    
    # 5. Tokenizing & Filtering Stopwords
    tokens = word_tokenize(text)
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(set(stopwords.words('english')))
    list_stopwords.update(['tiktok','tik','tok', 'aplikasi', 'aplikasinya', 'apk',
                          'gitu', 'udh', 'deh', 'iya', 'yaa', 'gak', 'nya', 'na',
                          'sih', 'ku', "di", "ga", "ya", "gaa", "loh", "kah",
                          'banget', 'biar', 'doang', 'mohon', 'plis', 'si', 'kan', 'kok'])
    
    filtered_tokens = [txt for txt in tokens if txt not in list_stopwords]
    
    # 6. Menggabungkan kembali menjadi kalimat utuh
    return ' '.join(filtered_tokens)


def automate_preprocessing(filepath, test_size=0.2, random_state=0):
    """
    Fungsi utama otomatisasi yang membaca file mentah, membersihkannya, 
    mengubahnya ke TF-IDF, membagi data, dan mengembalikan data siap latih.
    """
    # 1. Load Data
    print("[INFO] Membaca file dataset...")
    df = pd.read_csv(filepath)
    
    # Penanganan data kosong & duplikat
    df = df.dropna(subset=['text', 'label'])
    df = df.drop_duplicates(subset=['text'])
    
    # 2. Jalankan pipeline pembersihan teks pada kolom 'text'
    print("[INFO] Melakukan text preprocessing (Cleaning, Casefolding, Saka, Sastrawi, Stopwords)...")
    df['text_clean_automation'] = df['text'].apply(text_preprocessing_pipeline)
    
    # Pastikan tidak ada string kosong setelah dibersihkan
    df = df[df['text_clean_automation'].str.strip() != ""]
    
    X = df['text_clean_automation']
    y = df['label']
    
    # 3. Split Dataset menjadi Train dan Test sebelum Vektorasi (untuk menghindari Data Leakage)
    print("[INFO] Membagi dataset menjadi Train dan Test set...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Fit & Transform menggunakan TF-IDF Vectorizer
    print("[INFO] Melakukan ekstraksi fitur dengan TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000) # Batasi max_features agar model ringan di Docker
    
    X_train = vectorizer.fit_transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()
    
    print("[INFO] Preprocessing Selesai! Data siap dilatih.")
    
    # Mengembalikan data siap latih beserta objek vectorizer-nya untuk deployment nanti
    return X_train, X_test, y_train, y_test, vectorizer

# Script Guard untuk pengujian lokal jika file ini dieksekusi langsung
# Script Guard untuk pengujian lokal/GitHub Actions
if __name__ == "__main__":
    # 1. Sesuaikan dengan nama file dataset mentah Anda sesuai kriteria Advance
    path_sample = 'mental-health-indo-dataset-text.csv' 
    print("Menjalankan testing otomatisasi fungsi...")
    
    # 2. Aktifkan fungsi utama (hilangkan tanda komparasi/comment '#')
    X_train, X_test, y_train, y_test, vec = automate_preprocessing(path_sample)