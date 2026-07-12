
# Adaptive Ensemble Forecasting of Industrial-Scale Biogas Production

This repository contains the official dataset and source code for the paper: **"Adaptive ensemble forecasting of industrial-scale biogas production under fluctuating operating conditions"**. 

It provides a complete, fully reproducible pipeline including data preprocessing, baseline model evaluation, the proposed volatility-aware dynamic ensemble framework (LSTM-XGB-Volatility), and SHAP-based interpretability analysis.

## 📂 Repository Structure

* `Data of biogas plant.xlsx`: The raw operational dataset collected from a full-scale wet anaerobic digestion (AD) plant (395 days).
* `data_preprocessing.py`: Handles $2\sigma$ outlier removal, missing value imputation, normalization, and feature correlation analysis.
* `models.py`: Contains the PyTorch implementations of the deep learning architectures (LSTM and CNN-LSTM).
* `utils.py`: Contains global random seed configurations to ensure **absolute reproducibility**, along with evaluation metrics (RMSE, MAE, R², MAPE) and inverse transformation tools.
* `main.py`: The main execution script. It trains all baselines, implements the dynamic trailing-window weighting strategy (ensuring zero data leakage), and generates SHAP mechanism explanations.

## ⚙️ Environment Requirements

To ensure reproducibility, please install the following dependencies:
* Python 3.8+
* torch >= 1.10.0
* xgboost >= 1.5.0
* scikit-learn >= 1.0.2
* shap >= 0.40.0
* pandas, numpy, matplotlib, seaborn

## 🚀 How to Run

Simply execute the main script. All random seeds are fixed globally to guarantee that the results exactly match those reported in the manuscript.

```bash
python main.py
```bash
python main.py
