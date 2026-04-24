import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Create folder for V2 results
os.makedirs("training_v2", exist_ok=True)

# =========================
# 1. LOAD PROCESSED DATA
# =========================
df = pd.read_csv("Data/processed_turbidity_rainfall.csv")

# =========================
# 2. ADD EXTRA FEATURES
# =========================
df["rain_change"] = df["rainfall_mm"].diff().fillna(0)
df["is_raining"] = (df["rainfall_mm"] > 0).astype(int)

# =========================
# 3. TARGET
# =========================
df["target_turbidity"] = df["turbidity_fnu"].shift(-1)
df = df.dropna().reset_index(drop=True)

# =========================
# 4. FEATURES
# =========================
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
    "turb_mean_30min",
    "rain_change",
    "is_raining"
]

X = df[feature_cols]
y = df["target_turbidity"]

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# =========================
# 6. SCALE
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# =========================
# 7. CREATE SEQUENCES
# =========================
def create_sequences(X_data, y_data, window_size=24):
    X_seq, y_seq = [], []

    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i + window_size])
        y_seq.append(y_data[i + window_size])

    return np.array(X_seq), np.array(y_seq)

window_size = 24

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window_size)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

print("V2 X_train_seq:", X_train_seq.shape)
print("V2 X_test_seq:", X_test_seq.shape)

# =========================
# 8. V2 MODEL
# =========================
model = Sequential([
    LSTM(64, input_shape=(window_size, len(feature_cols))),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# =========================
# 9. TRAIN
# =========================
history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=12,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq)
)

# =========================
# 10. PLOT LOSS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("V2 Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_v2/loss_curve_v2.png")
plt.show()
plt.close()

# =========================
# 11. EVALUATE
# =========================
y_pred_scaled = model.predict(X_test_seq)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_seq)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)

print("V2 RMSE:", rmse)
print("V2 MAE:", mae)

# Save predictions
results_df = pd.DataFrame({
    "actual_turbidity": y_test_actual.flatten(),
    "predicted_turbidity": y_pred.flatten()
})
results_df.to_csv("training_v2/predictions_v2.csv", index=False)

# =========================
# 12. PLOT ACTUAL VS PREDICTED
# =========================
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:500], label="Actual")
plt.plot(y_pred[:500], label="Predicted")
plt.title("V2 Actual vs Predicted Turbidity")
plt.xlabel("Time Step")
plt.ylabel("Turbidity (FNU)")
plt.legend()
plt.tight_layout()
plt.savefig("training_v2/prediction_vs_actual_v2.png")
plt.show()
plt.close()

# =========================
# 13. SAVE MODEL
# =========================
model.save("improved_lstm_model_v2.keras")
print("V2 model saved.")