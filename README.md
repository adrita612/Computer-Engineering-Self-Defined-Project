# Turbidity Spike Prediction using LSTM

## Overview
Turbidity is a key indicator of water quality, representing the presence of suspended particles such as sediment, organic matter, and microorganisms. Sudden increases in turbidity—often caused by rainfall events and runoff—can negatively impact drinking water treatment processes, aquatic ecosystems, and regulatory compliance.

Accurate prediction of turbidity is important for:

- **Water treatment utilities**: Anticipating turbidity spikes allows operators to adjust treatment processes (e.g., coagulation, filtration) in advance, improving efficiency and reducing risk.
- **Early warning systems**: Predictive models can provide advance notice of water quality deterioration during storm events.
- **Infrastructure planning**: Understanding how rainfall affects turbidity helps inform watershed management and stormwater control strategies.
- **Environmental protection**: High turbidity can harm aquatic habitats by reducing light penetration and transporting pollutants.

This project applies Long Short-Term Memory (LSTM) neural networks to model the relationship between rainfall and turbidity over time. By learning temporal patterns from historical data, the model aims to forecast short-term turbidity changes and evaluate how effectively machine learning can capture rainfall-driven water quality dynamics.

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

# Results and Model Comparison

This project evaluated multiple machine learning approaches to predict turbidity dynamics in the Northwest Branch Anacostia River at Brentwood, MD.

The models were developed progressively to improve predictive performance and to explore alternative approaches when rainfall signals were weak or inconsistent.

---

## Part 1: Turbidity Forecasting Models (Regression)

These models predict the **next turbidity value** using time-series inputs.

### Model Comparison

| Model        | Architecture                     | Inputs Used                | RMSE  | MAE   | Key Insight |
|-------------|---------------------------------|----------------------------|-------|-------|------------|
| Baseline     | Single-layer LSTM               | Turbidity + Rainfall       | ~4.5  | ~1.3  | Simple model captures basic temporal trends |
| V1           | Larger LSTM                     | Turbidity + Rainfall       | ~4.2  | ~1.1  | Increased capacity improves learning |
| V2           | Tuned LSTM                      | Turbidity + Rainfall + engineered features | ~4.16 | ~0.71 | Best regression performance |

### Key Findings

- Turbidity predictions were driven primarily by historical turbidity values
- Rainfall had limited influence due to:
  - low rainfall frequency
  - weak storm signals in the dataset
- Feature engineering (rolling sums, lags) improved performance more than model complexity

---

## Part 2: Spike Prediction Model (V3 – Research-Oriented)

Due to the weak relationship between rainfall and turbidity in the dataset, a second modeling approach was introduced.

Instead of predicting exact turbidity values, this model predicts:

> **Will a turbidity spike occur in the next timestep?**

This reframes the problem into a classification task, which is more aligned with real-world utility needs.

---

### V3 Model Overview

| Model        | Type            | Inputs Used         | Output |
|-------------|----------------|--------------------|--------|
| V3          | 1D CNN         | Turbidity-only features | Spike / No Spike |

Key characteristics:
- Does not use rainfall
- Uses only:
  - turbidity lags
  - rate of change
  - rolling statistics
- Focuses on **event detection**, not exact prediction

---

### V3 Results (Fixed Version)

| Metric      | Value |
|------------|------|
| Precision  | **0.18** |
| Recall     | **0.37** |
| F1 Score   | **0.24** |
| Accuracy   | ~0.97 (not meaningful due to imbalance) |

Confusion Matrix:
[[9600 235]
[ 87 52]]

---

### Interpretation

- The model successfully reduced **false alarms** compared to the initial version
- Precision improved significantly (fewer incorrect spike predictions)
- Recall decreased, meaning some spikes were missed
- This demonstrates the tradeoff between:
  - detecting all spikes
  - avoiding false positives

---

## Engineering and Utility Perspective

For drinking water utilities and watershed managers:

- **High recall** is important → avoid missing contamination events  
- **High precision** is important → avoid unnecessary operational responses  

The V3 model demonstrates a **balanced compromise**, making it more realistic for operational use compared to the initial spike model.

---

## Implications for the Washington (Anacostia) Watershed

The results suggest that:

- Turbidity in the Northwest Branch Anacostia River is not strongly driven by rainfall alone
- Other factors likely play a significant role:
  - urban runoff timing
  - baseflow sediment transport
  - antecedent watershed conditions

Because of this:

> Rainfall-based prediction alone is insufficient for accurate turbidity forecasting in this watershed.

---

## Why V3 Was Important

The introduction of V3 highlights an important insight:

> When environmental drivers (like rainfall) are weak or inconsistent, alternative modeling strategies must be used.

V3 demonstrates that:
- meaningful predictions can still be made using turbidity-only signals
- event-based prediction (spikes) may be more useful than continuous forecasting

---

## Overall Conclusion

This project shows that:

- LSTM-based models can effectively capture **short-term turbidity trends**
- Feature engineering plays a critical role in improving model performance
- Rainfall data alone may not sufficiently explain turbidity behavior in urban watersheds
- Spike prediction models provide a valuable alternative for **early warning systems**

The combination of regression and classification approaches offers a more complete framework for turbidity prediction and supports future work in:

- drinking water treatment optimization  
- watershed monitoring systems  
- real-time alerting for water quality events  

## Overall Conclusion

This project demonstrates that machine learning models, particularly LSTM-based architectures, can effectively capture short-term temporal patterns in turbidity data. The baseline and improved regression models were able to predict general turbidity behavior with reasonable accuracy, showing that turbidity persistence is a dominant factor in short-term forecasting.

However, the results also highlight a critical limitation: rainfall alone was not a strong predictor of turbidity in this dataset. Despite incorporating rainfall features and increasing model complexity, improvements were marginal. This indicates that turbidity dynamics in the Northwest Branch Anacostia watershed are influenced by a combination of factors beyond rainfall, including urban runoff timing, sediment availability, and antecedent conditions.

To address this limitation, the problem was reframed in Part 2 as a spike detection task. The V3 model demonstrated that even without rainfall inputs, meaningful predictions can be made using turbidity-derived features alone. While the model exhibited a tradeoff between precision and recall, it successfully reduced false alarms and provided a more operationally realistic approach for detecting significant turbidity events.

From an engineering and utility perspective, this work reinforces several key insights:

- Accurate turbidity forecasting is not solely a modeling challenge, but a data and system understanding problem
- Simpler models can outperform more complex architectures when environmental signals are weak
- Event-based prediction (spike detection) may be more valuable than continuous prediction for real-world applications
- Model tuning must balance false positives vs missed events, especially for drinking water operations

Overall, this project illustrates that combining regression-based forecasting with classification-based spike detection provides a more complete and practical framework for turbidity prediction. While the current models are constrained by data limitations, the methodology establishes a strong foundation for future work in watershed monitoring, early warning systems, and drinking water treatment optimization.

Future improvements should focus on integrating richer hydrological data (e.g., streamflow, runoff characteristics), extending the dataset to include more storm events, and refining event-based modeling approaches to better capture rare but critical turbidity spikes.
