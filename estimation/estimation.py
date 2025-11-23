import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
df = pd.read_csv("D:/College/Semester_5/Data Mining/coding/tugas paper/estimation/insurance.csv")

print("Jumlah data:", len(df))
print(df.head())

# PREPROCESSING
df = df.dropna()  # Hapus missing values

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# DEFINISI MODEL
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = {}

# TRAINING & EVALUATION
predictions_all = {}    # untuk menyimpan prediksi per model (dipakai untuk grafik)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    predictions_all[name] = preds   # simpan prediksi

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

# TAMPILKAN HASIL EVALUASI
print("\n======= HASIL EVALUASI MODEL =======")
for name, metrics in results.items():
    print(f"\n{name}")
    print(f"MAE  : {metrics['MAE']:.2f}")
    print(f"RMSE : {metrics['RMSE']:.2f}")
    print(f"R²   : {metrics['R²']:.4f}")

# ESTIMASI CONTOH DATA ASLI
sample = X_test.iloc[:5]
sample_preds = models["Gradient Boosting"].predict(sample)

print("\n======= CONTOH ESTIMASI =======")
for i in range(len(sample)):
    print(f"Estimasi: {sample_preds[i]:.2f} | Aktual: {y_test.iloc[i]:.2f}")

# GRAFIK ESTIMASI vs AKTUAL
best_model_name = "Gradient Boosting"
best_preds = predictions_all[best_model_name]

plt.figure(figsize=(8,6))
plt.scatter(y_test, best_preds)
plt.xlabel("Nilai Aktual (charges)")
plt.ylabel("Estimasi Model")
plt.title(f"Estimasi vs Aktual — {best_model_name}")
plt.grid(True)
plt.show()
