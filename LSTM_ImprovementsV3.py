import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# SETUP
# =========================
os.makedirs("training_v3_fixed", exist_ok=True)

# =========================
# 1. LOAD PROCESSED DATA
# =========================
df = pd.read_csv("Data/processed_turbidity_rainfall.csv")

# =========================
# 2. ADD TURBIDITY-ONLY FEATURES
# =========================
df["turb_diff_1"] = df["turbidity_fnu"].diff().fillna(0)
df["turb_diff_3"] = df["turbidity_fnu"] - df["turbidity_fnu"].shift(3)
df["turb_diff_3"] = df["turb_diff_3"].fillna(0)

df["turb_std_30min"] = df["turbidity_fnu"].rolling(window=6).std()
df["turb_std_30min"] = df["turb_std_30min"].fillna(0)

df["turb_max_30min"] = df["turbidity_fnu"].rolling(window=6).max()
df["turb_max_30min"] = df["turb_max_30min"].fillna(df["turbidity_fnu"])

# =========================
# 3. CREATE SPIKE TARGET
# =========================
df["target_delta"] = df["turbidity_fnu"].shift(-1) - df["turbidity_fnu"]
df = df.dropna().reset_index(drop=True)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# Slightly stricter spike definition
positive_deltas = train_df.loc[train_df["target_delta"] > 0, "target_delta"]

if len(positive_deltas) == 0:
    raise ValueError("No positive turbidity changes found in training data.")

spike_threshold = positive_deltas.quantile(0.97)
print("Spike threshold:", spike_threshold)

train_df["spike_flag"] = (train_df["target_delta"] >= spike_threshold).astype(int)
test_df["spike_flag"] = (test_df["target_delta"] >= spike_threshold).astype(int)

print("Train spike count:", train_df["spike_flag"].sum())
print("Test spike count:", test_df["spike_flag"].sum())

# =========================
# 4. FEATURES / TARGET
# =========================
feature_cols = [
    "turbidity_fnu",
    "turb_lag_1",
    "turb_lag_3",
    "turb_lag_6",
    "turb_mean_30min",
    "turb_diff_1",
    "turb_diff_3",
    "turb_std_30min",
    "turb_max_30min"
]

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

y_train = train_df["spike_flag"].values
y_test = test_df["spike_flag"].values

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =========================
# 5. SCALE FEATURES
# =========================
scaler_X = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# =========================
# 6. CREATE SEQUENCES
# =========================
def create_classification_sequences(X_data, y_data, window_size=24):
    X_seq, y_seq = [], []

    for i in range(len(X_data) - window_size):
        X_seq.append(X_data[i:i + window_size])
        y_seq.append(y_data[i + window_size])

    return np.array(X_seq), np.array(y_seq)

window_size = 24

X_train_seq, y_train_seq = create_classification_sequences(X_train_scaled, y_train, window_size)
X_test_seq, y_test_seq = create_classification_sequences(X_test_scaled, y_test, window_size)

print("X_train_seq shape:", X_train_seq.shape)
print("X_test_seq shape:", X_test_seq.shape)
print("Train spike rate:", y_train_seq.mean())
print("Test spike rate:", y_test_seq.mean())

# =========================
# 7. CLASS WEIGHTS
# =========================
num_neg = np.sum(y_train_seq == 0)
num_pos = np.sum(y_train_seq == 1)

if num_pos == 0:
    raise ValueError("No spike samples found after sequence creation. Lower the threshold.")

positive_weight = (num_neg / num_pos) * 0.4
class_weight = {
    0: 1.0,
    1: positive_weight
}

print("Class weights:", class_weight)

# =========================
# 8. BUILD CNN MODEL
# =========================
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(window_size, len(feature_cols))),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=64, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 9. TRAIN
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=15,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    class_weight=class_weight,
    callbacks=[early_stop]
)

# =========================
# 10. PLOT TRAINING CURVES
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("V3 Fixed Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy")
plt.legend()
plt.tight_layout()
plt.savefig("training_v3_fixed/loss_curve_v3_fixed.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("V3 Fixed Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_v3_fixed/accuracy_curve_v3_fixed.png")
plt.show()
plt.close()

# =========================
# 11. PREDICT
# =========================
y_prob = model.predict(X_test_seq).flatten()

# Higher threshold to reduce false positives
decision_threshold = 0.80
y_pred = (y_prob >= decision_threshold).astype(int)

# =========================
# 12. EVALUATE
# =========================
precision = precision_score(y_test_seq, y_pred, zero_division=0)
recall = recall_score(y_test_seq, y_pred, zero_division=0)
f1 = f1_score(y_test_seq, y_pred, zero_division=0)
cm = confusion_matrix(y_test_seq, y_pred)

print("Decision threshold:", decision_threshold)
print("V3 Fixed Precision:", precision)
print("V3 Fixed Recall:", recall)
print("V3 Fixed F1 Score:", f1)
print("Confusion Matrix:\n", cm)

report = classification_report(y_test_seq, y_pred, zero_division=0)
print(report)

# Save report
with open("training_v3_fixed/classification_report_v3_fixed.txt", "w") as f:
    f.write(f"Spike threshold: {spike_threshold}\n")
    f.write(f"Decision threshold: {decision_threshold}\n\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# =========================
# 13. CONFUSION MATRIX PLOT
# =========================
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("V3 Fixed Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["No Spike", "Spike"])
plt.yticks([0, 1], ["No Spike", "Spike"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig("training_v3_fixed/confusion_matrix_v3_fixed.png")
plt.show()
plt.close()

# =========================
# 14. SAVE PREDICTIONS
# =========================
pred_df = pd.DataFrame({
    "actual_spike": y_test_seq,
    "predicted_spike": y_pred,
    "predicted_probability": y_prob
})
pred_df.to_csv("training_v3_fixed/spike_predictions_v3_fixed.csv", index=False)

# =========================
# 15. SAVE MODEL
# =========================
model.save("spike_classifier_v3_fixed.keras")
print("V3 fixed spike classifier saved.")