# 📘 Data Mining — Kelompok 11
Mata Kuliah: **Data Mining**  
Semester: 5  

## 👥 Anggota Kelompok 11
- **Lidya Khairunnisa (L0123075)**
- **Mohammad Adzka Crosamer (L0123083)**
- **Muiz Afif Mirza Lindu Aji (L0123099)**

---

## 📌 Deskripsi Proyek
Repository ini berisi implementasi lima topik utama dalam **Data Mining**, yaitu:

1. **Estimasi**
2. **Prediksi**
3. **Klastering**
4. **Klasifikasi**
5. **Asosiasi**

Setiap topik menggunakan **satu algoritma utama** dan **beberapa algoritma pembanding**, lengkap dengan dataset, source code, evaluasi, dan visualisasi.

Struktur folder telah diatur berdasarkan topik agar mudah dipelajari dan direplikasi.

---

## 📚 Topik & Algoritma yang Digunakan

### 1️⃣ Estimasi
- **Algoritma Utama:** Gradient Boosting  
- **Pembanding:** Random Forest, Linear Regression  
- **Dataset:** `insurance.csv`

Hasil dan Evaluasi:
- MAE  
- RMSE  
- R² Score  

---

### 2️⃣ Prediksi
- **Algoritma Utama:** SARIMA  
- **Pembanding:** ARIMA, Holt-Winters, Prophet  
- **Dataset:** `AirPassengers.csv`

Hasil dan Evaluasi:
- MAE
- RMSE
- MAPE  

---

### 3️⃣ Klastering
- **Algoritma Utama:** K-Means  
- **Pembanding:** Hierarchical Clustering  
- **Dataset:** `dataset500.csv`

Hasil dan Evaluasi:
- Silhouette Score  
- Calinski - Harabasz Index  
- Dunn Index  

---

### 4️⃣ Klasifikasi
- **Algoritma Utama:** Naive Bayes  
- **Pembanding:** Logistic Regression, Random Forest, SVM  
- **Dataset:** `spam.csv`

Hasil dan Evaluasi:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

### 5️⃣ Asosiasi
- **Algoritma Utama:** FP-Growth  
- **Pembanding:** ECLAT, AIS  
- **Dataset:** `associationBaru.csv`

Hasil dan Evaluasi:
- Frequent itemsets  
- Association rules  
- Jumlah rules  
- Waktu eksekusi  
- Rata-rata confidence & lift  
- Grafik pop-up menggunakan Matplotlib  

---

## 🛠️ Cara Menjalankan Program

### 1. Install dependency dan jalankan script
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels prophet mlxtend

python association/association.py
python estimation/estimation.py
python prediction/prediction.py
python clustering/clustering.py
python classification/classification.py

