import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create output folders
os.makedirs("training", exist_ok=True)
os.makedirs("Data", exist_ok=True)

# =========================
# 1. LOAD DATA
# =========================
rain = pd.read_csv(
    r"C:\Users\adrit\OneDrive\Documents\UBCO\Y4T2\CMPE 401\Self-Defined Project\TURBIDITY_MODEL\Data\Mesonet_Minute_RainGauge.csv"
)
turb = pd.read_csv(
    r"C:\Users\adrit\OneDrive\Documents\UBCO\Y4T2\CMPE 401\Self-Defined Project\TURBIDITY_MODEL\Data\Northwest Branch Anacostia River at Brentwood, MD - USGS-01651003.csv"
)

rain["datetime"] = pd.to_datetime(rain["TMSTAMP_UTC"], utc=True)
turb["datetime"] = pd.to_datetime(turb["time"], utc=True)

rain = rain[["datetime", "Rain_ICA_Tot_mm"]].copy()
turb = turb[["datetime", "value"]].copy()

rain = rain.rename(columns={"Rain_ICA_Tot_mm": "rainfall_mm"})
turb = turb.rename(columns={"value": "turbidity_fnu"})

rain = rain.sort_values("datetime")
turb = turb.sort_values("datetime")

# =========================
# 2. MERGE DATA
# =========================
df = pd.merge_asof(
    turb,
    rain,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("5min")
)

# Fill missing rain with 0
df["rainfall_mm"] = df["rainfall_mm"].fillna(0)

# =========================
# 3. FEATURE ENGINEERING
# =========================
# Rain lags
df["rain_lag_1"] = df["rainfall_mm"].shift(1)
df["rain_lag_3"] = df["rainfall_mm"].shift(3)
df["rain_lag_6"] = df["rainfall_mm"].shift(6)

# Turbidity lags
df["turb_lag_1"] = df["turbidity_fnu"].shift(1)
df["turb_lag_3"] = df["turbidity_fnu"].shift(3)
df["turb_lag_6"] = df["turbidity_fnu"].shift(6)

# Rolling rainfall totals
df["rain_15min"] = df["rainfall_mm"].rolling(window=3).sum()
df["rain_30min"] = df["rainfall_mm"].rolling(window=6).sum()
df["rain_1hr"] = df["rainfall_mm"].rolling(window=12).sum()

# Rolling turbidity mean
df["turb_mean_30min"] = df["turbidity_fnu"].rolling(window=6).mean()

# Drop NaNs from lag/rolling creation
df = df.dropna().reset_index(drop=True)

print(df.head(10))
print(df.columns)
print(df.shape)

df.to_csv("Data/processed_turbidity_rainfall.csv", index=False)
print("Saved processed dataset.")

# =========================
# 4. TARGET SETUP
# =========================
df["target_turbidity"] = df["turbidity_fnu"].shift(-1)
df = df.dropna().reset_index(drop=True)

feature_cols = [
    "turbidity_fnu",
    "rainfall_mm",
    "rain_lag_1",
    "rain_lag_3",
    "rain_lag_6",
    "turb_lag_1",
    "turb_lag_3",
    "turb_lag_6",
    "rain_15min",
    "rain_30min",
    "rain_1hr",
    "turb_mean_30min"
]

X = df[feature_cols].copy()
y = df["target_turbidity"]

print(X.head())
print(y.head())

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# =========================
# 6. SCALE DATA
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_train_scaled shape:", y_train_scaled.shape)
print("y_test_scaled shape:", y_test_scaled.shape)

# =========================
# 7. CREATE SEQUENCES
# =========================
def create_sequences(X_data, y_data, window_size=12):
    X_seq, y_seq = [], []

    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i + window_size])
        y_seq.append(y_data[i + window_size])

    return np.array(X_seq), np.array(y_seq)

window_size = 12  # 12 x 5 min = 1 hour

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window_size)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

print("X_train_seq shape:", X_train_seq.shape)
print("y_train_seq shape:", y_train_seq.shape)
print("X_test_seq shape:", X_test_seq.shape)
print("y_test_seq shape:", y_test_seq.shape)

# =========================
# 8. BUILD MODEL
# =========================
model = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# =========================
# 9. TRAIN MODEL
# =========================
print("Starting training...")

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq)
)

# =========================
# 10. PLOT TRAINING LOSS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training/loss_curve.png")
plt.show()
plt.close()

# =========================
# 11. BASELINE EVALUATION
# =========================
y_pred_scaled = model.predict(X_test_seq)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_seq)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)

# Save prediction results
results_df = pd.DataFrame({
    "actual_turbidity": y_test_actual.flatten(),
    "predicted_turbidity": y_pred.flatten()
})
results_df.to_csv("training/predictions.csv", index=False)
print("Saved predictions to training/predictions.csv")

# =========================
# 12. PLOT ACTUAL VS PREDICTED
# =========================
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:500], label="Actual")
plt.plot(y_pred[:500], label="Predicted")
plt.title("Actual vs Predicted Turbidity (First 500 Test Points)")
plt.xlabel("Time Step")
plt.ylabel("Turbidity (FNU)")
plt.legend()
plt.tight_layout()
plt.savefig("training/prediction_vs_actual.png")
plt.show()
plt.close()

# =========================
# 13. MOCK RAINFALL TEST
# =========================
# Synthetic storm for comparison only
mock_rain = np.array([0.0, 0.0, 0.2, 0.5, 1.0, 2.0, 1.4, 0.8, 0.3, 0.1, 0.0, 0.0])

# Use first test window as baseline
mock_seq_unscaled = X_test.iloc[:window_size].copy().reset_index(drop=True)

# Replace rainfall pattern
mock_seq_unscaled["rainfall_mm"] = mock_rain
mock_seq_unscaled["rain_lag_1"] = mock_seq_unscaled["rainfall_mm"].shift(1).fillna(0)
mock_seq_unscaled["rain_lag_3"] = mock_seq_unscaled["rainfall_mm"].shift(3).fillna(0)
mock_seq_unscaled["rain_lag_6"] = mock_seq_unscaled["rainfall_mm"].shift(6).fillna(0)
mock_seq_unscaled["rain_15min"] = mock_seq_unscaled["rainfall_mm"].rolling(window=3, min_periods=1).sum()
mock_seq_unscaled["rain_30min"] = mock_seq_unscaled["rainfall_mm"].rolling(window=6, min_periods=1).sum()
mock_seq_unscaled["rain_1hr"] = mock_seq_unscaled["rainfall_mm"].rolling(window=12, min_periods=1).sum()

mock_seq_scaled = scaler_X.transform(mock_seq_unscaled[feature_cols])
mock_seq_scaled = mock_seq_scaled.reshape(1, window_size, len(feature_cols))

# Compare normal vs storm prediction
normal_pred_scaled = model.predict(X_test_seq[0:1])
normal_pred = scaler_y.inverse_transform(normal_pred_scaled)

mock_pred_scaled = model.predict(mock_seq_scaled)
mock_pred = scaler_y.inverse_transform(mock_pred_scaled)

print("\n--- Mock Rainfall Comparison ---")
print("Normal sequence predicted next turbidity:", normal_pred[0][0])
print("Mock storm sequence predicted next turbidity:", mock_pred[0][0])
print("Mock rainfall pattern used:", mock_rain)

# =========================
# 14. PLOT MOCK RAINFALL
# =========================
plt.figure(figsize=(8, 5))
plt.plot(mock_rain, marker="o")
plt.title("Synthetic Rainfall Event")
plt.xlabel("Time Step (5-min intervals)")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.savefig("training/mock_rainfall.png")
plt.show()
plt.close()

# =========================
# 15. SAVE MOCK COMPARISON TEXT
# =========================
with open("training/mock_comparison.txt", "w") as f:
    f.write("--- Mock Rainfall Comparison ---\n")
    f.write(f"Normal sequence predicted next turbidity: {normal_pred[0][0]}\n")
    f.write(f"Mock storm sequence predicted next turbidity: {mock_pred[0][0]}\n")
    f.write(f"Mock rainfall pattern used: {mock_rain.tolist()}\n")

print("Saved mock comparison to training/mock_comparison.txt")

# =========================
# 16. SAVE MODEL
# =========================
model.save("lstm_model.keras")
print("Model saved.")