📘 Data Mining Paper – Kelompok 11

Repository ini berisi implementasi lengkap lima topik utama dalam Data Mining, yaitu Estimasi, Prediksi, Klastering, Klasifikasi, dan Asosiasi.
Seluruh topik telah dilengkapi dengan dataset, source code, output, dan evaluasi algoritma, sesuai ketentuan tugas kelompok.

👥 Anggota Kelompok 11

Lidya Khairunnisa (L0123075)

Mohammad Adzka Crosamer (L0123083)

Muiz Afif Mirza Lindu Aji (L0123099)

📂 Struktur Repository
data_mining_paper/
│
├── association/
│   ├── association.py
│   └── associationBaru.csv
│
├── classification/
│   ├── classification.py
│   └── spam.csv
│
├── clustering/
│   ├── clustering.py
│   └── dataset500.csv
│
├── estimation/
│   ├── estimation.py
│   └── insurance.csv
│
├── prediction/
│   ├── prediction.py
│   └── AirPassangers.csv
│
└── README.md

📑 Penjelasan Setiap Topik & Algoritma
1️⃣ Estimasi

Algoritma utama: Gradient Boosting
Pembanding: Random Forest, Linear Regression

Digunakan untuk memperkirakan nilai kontinu menggunakan dataset insurance.csv (estimasi biaya asuransi).
Evaluasi mencakup:

MAE

MSE

RMSE

R² Score

2️⃣ Prediksi

Algoritma utama: SARIMA
Pembanding: ARIMA, Holt-Winters, Prophet

Digunakan untuk melakukan peramalan deret waktu menggunakan dataset AirPassengers.csv.
Analisis mencakup:

Decomposition

Plot hasil prediksi

Error metrics (MAPE, RMSE)

3️⃣ Klastering

Algoritma utama: K-Means
Pembanding: Hierarchical Clustering

Mengelompokkan data pada dataset500.csv ke dalam beberapa cluster.
Visualisasi & evaluasi:

Scatter plot cluster

Dendrogram

Silhouette Score

4️⃣ Klasifikasi

Algoritma utama: Naive Bayes
Pembanding: Logistic Regression, Random Forest, SVM

Mengklasifikasikan email spam pada dataset spam.csv.
Evaluasi:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

5️⃣ Asosiasi

Algoritma utama: FP-Growth
Pembanding: ECLAT, AIS

Analisis association rules dilakukan pada dataset associationBaru.csv.
Hasil mencakup:

Frequent itemsets

Association rules

Perbandingan jumlah rules

Waktu eksekusi

Rata-rata confidence dan lift

Grafik menggunakan Matplotlib (window pop-up)

🛠 Cara Menjalankan Program

Pastikan Python 3.10+ sudah terinstall

Install dependencies:

pip install pandas numpy matplotlib scikit-learn statsmodels prophet mlxtend


Jalankan program per topik:

python association/association.py
python estimation/estimation.py
python prediction/prediction.py
python clustering/clustering.py
python classification/classification.py


Semua grafik akan muncul melalui jendela pop-up Matplotlib.

📚 Dataset

Semua dataset yang digunakan berasal dari Kaggle atau sumber publik lain dan telah disertakan langsung dalam repository untuk memudahkan replikasi.

🧾 Output

Setiap script menghasilkan:

Visualisasi grafik

Tabel evaluasi

Perbandingan performa algoritma

File output (khusus asosiasi: CSV summary & rules)

📖 Referensi

Referensi lengkap terdapat pada laporan (.docx) masing-masing topik.

Jika kamu mau, aku bisa buatkan:
✅ README versi lebih estetis (emoji + banner)
✅ README versi akademik (tanpa emoji)
✅ README dengan badge GitHub (stars, issues, license)