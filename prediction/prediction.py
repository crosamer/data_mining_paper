import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet

# Load Dataset
df = pd.read_csv("D:/College/Semester_5/Data Mining/coding/tugas paper/prediction/AirPassengers.csv")  # ganti sesuai nama file Anda
df.columns = ["Month", "Passengers"]
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)

# Train-test split (80/20)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fungsi Evaluasi
def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, rmse, mape


# Model SARIMA
model_sarima = SARIMAX(train["Passengers"],
                       order=(2,1,2),
                       seasonal_order=(1,1,1,12))
result_sarima = model_sarima.fit()

pred_sarima = result_sarima.forecast(steps=len(test))

mae_s, rmse_s, mape_s = evaluate_model(test["Passengers"], pred_sarima)


# Model ARIMA
model_arima = ARIMA(train["Passengers"], order=(2,1,2))
result_arima = model_arima.fit()

pred_arima = result_arima.forecast(steps=len(test))

mae_a, rmse_a, mape_a = evaluate_model(test["Passengers"], pred_arima)


# Holt-Winters
model_hw = ExponentialSmoothing(train["Passengers"],
                                trend="add",
                                seasonal="mul",
                                seasonal_periods=12)

result_hw = model_hw.fit()
pred_hw = result_hw.forecast(len(test))

mae_hw, rmse_hw, mape_hw = evaluate_model(test["Passengers"], pred_hw)


# Model Prophet
df_prophet = df.reset_index().rename(columns={"Month": "ds", "Passengers": "y"})
train_p = df_prophet.iloc[:train_size]
test_p = df_prophet.iloc[train_size:]

model_prophet = Prophet()
model_prophet.fit(train_p)

future = model_prophet.make_future_dataframe(periods=len(test_p), freq="M")
forecast = model_prophet.predict(future)

pred_prophet = forecast["yhat"].iloc[-len(test):].values

mae_p, rmse_p, mape_p = evaluate_model(test["Passengers"].values, pred_prophet)


# Hasil
print("===================================================")
print("              MODEL EVALUATION RESULT")
print("===================================================")
print(f"SARIMA          -> MAE: {mae_s:.2f}, RMSE: {rmse_s:.2f}, MAPE: {mape_s:.2f}%")
print(f"ARIMA           -> MAE: {mae_a:.2f}, RMSE: {rmse_a:.2f}, MAPE: {mape_a:.2f}%")
print(f"Holt-Winters    -> MAE: {mae_hw:.2f}, RMSE: {rmse_hw:.2f}, MAPE: {mape_hw:.2f}%")
print(f"Prophet         -> MAE: {mae_p:.2f}, RMSE: {rmse_p:.2f}, MAPE: {mape_p:.2f}%")
print("===================================================")


# Visualisasi
plt.figure(figsize=(12,6))
plt.plot(train.index, train["Passengers"], label="Train")
plt.plot(test.index, test["Passengers"], label="Test")

plt.plot(test.index, pred_sarima, label="SARIMA")
plt.plot(test.index, pred_arima, label="ARIMA")
plt.plot(test.index, pred_hw, label="Holt-Winters")
plt.plot(test.index, pred_prophet, label="Prophet")

plt.legend()
plt.title("Forecasting Comparison - Airline Passengers Dataset")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()
