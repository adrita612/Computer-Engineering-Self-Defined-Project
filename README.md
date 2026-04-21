# Turbidity Spike Prediction using LSTM

## Overview
This project develops a Long Short-Term Memory (LSTM) model to predict turbidity levels in a watershed using historical turbidity and rainfall data. The goal is to analyze how rainfall influences turbidity spikes and evaluate different model configurations.

---

## Dataset

The dataset combines:

- **Turbidity data**: USGS Northwest Branch Anacostia River (Brentwood, MD)
- **Rainfall data**: Mesonet minute-level rain gauge

### Features Engineered
- Lag features (rainfall and turbidity)
- Rolling rainfall totals (15 min, 30 min, 1 hr)
- Rolling turbidity mean (30 min)
- Rainfall change (V2 improvement)
- Binary rain indicator (V2 improvement)

---

## Methodology

### Data Processing
- Timestamp alignment using nearest merge
- Handling missing rainfall values
- Feature engineering for temporal relationships

### Model Development
Three models were implemented:

- **Baseline Model**
- **Improved Model V1 (High Complexity)**
- **Improved Model V2 (Targeted Improvement)**

---

## Model Configurations

| Model | Window Size | Features | Architecture |
|------|------------|----------|--------------|
| Baseline LSTM | 12 | 12 | 1 LSTM + Dense |
| Improved V1 | 24 | 14 | Stacked LSTM + Dropout |
| Improved V2 | 24 | 14 | 1 LSTM + Dropout |

---

## Results Comparison

| Model | RMSE | MAE | Observations |
|------|------|-----|-------------|
| Baseline LSTM | **4.184** | **0.636** | Best overall performance |
| Improved V1 | 4.265 | 1.149 | Overly complex, degraded results |
| Improved V2 | 4.161 | 0.706 | Slight RMSE improvement, higher MAE |

---

## Key Findings

- The **baseline model performed best overall**, achieving the lowest MAE.
- Increasing model complexity (V1) **worsened performance**, likely due to overfitting and weak rainfall signals.
- The V2 model showed a **small improvement in RMSE**, suggesting better handling of larger deviations.
- However, the dataset is dominated by dry periods, limiting the model’s ability to learn rainfall-driven turbidity spikes.

---

## Rainfall Sensitivity Analysis

A synthetic rainfall event was introduced to evaluate model behavior.

### Observations:
- The model shows **limited response to rainfall inputs**
- Predictions rely heavily on **past turbidity trends**
- Indicates **weak rainfall signal** in dataset

---

## Visualizations

Saved in:
