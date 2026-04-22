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

## Implications for Rainfall and Turbidity in the Northwest Branch Anacostia Watershed

The modeling results provide insight into how rainfall influences turbidity within the Northwest Branch Anacostia River at Brentwood, MD.

The dataset used in this study showed that rainfall events were relatively **infrequent and low in intensity** compared to the overall observation period. As a result, most time steps were dominated by dry conditions, with limited instances of significant runoff-driven disturbances.

### Key Observations

- **Weak rainfall–turbidity relationship**:  
  The model relied primarily on past turbidity values rather than rainfall inputs, indicating that rainfall was not a strong or consistent driver of turbidity changes in the dataset.

- **Dominance of baseline conditions**:  
  Turbidity values remained relatively stable over time, suggesting that under typical conditions, the watershed does not experience frequent or extreme sediment mobilization.

- **Limited spike events**:  
  The scarcity of rainfall events reduced the model’s ability to learn clear cause–effect relationships between rainfall and turbidity spikes.

- **Minimal response to synthetic storms**:  
  When a mock rainfall event was introduced, the model showed little change in predicted turbidity, reinforcing the idea that rainfall-driven spikes were underrepresented in the data.

### Interpretation for the Watershed

These findings suggest that, during the study period:

- The watershed likely experienced **moderate hydrological conditions**, with few intense storm events.
- Turbidity behavior was influenced more by **background conditions** (e.g., baseflow, sediment already in suspension) than by rainfall-driven runoff.
- Rainfall alone may not be sufficient to explain turbidity spikes without considering additional factors such as:
  - streamflow or discharge
  - land use and urban runoff
  - soil conditions and antecedent moisture

### Implications for Water Utilities

For water utilities monitoring the Northwest Branch Anacostia River:

- Short-term turbidity forecasting may be more reliable when based on **recent turbidity trends** rather than rainfall alone.
- Rainfall-based early warning systems may require:
  - higher-resolution rainfall data
  - additional hydrological inputs (e.g., flow rate)
- During periods of low rainfall variability, predictive models may **underestimate sudden turbidity events**, particularly if they are driven by localized or unobserved conditions.

### Overall Conclusion

The results indicate that, for this specific dataset and time period, rainfall was not a dominant predictor of turbidity dynamics in the Northwest Branch Anacostia watershed. This highlights the importance of incorporating richer environmental data and longer observation periods when modeling water quality responses in urban watersheds.
