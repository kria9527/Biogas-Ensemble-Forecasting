import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from utils import set_global_seed, inverse_transform_target, evaluate_performance
from data_preprocessing import load_and_preprocess_data
from models import LSTMModel


def main():
    set_global_seed(42)
    filepath = 'Data of biogas plant.xlsx'
    X_train, X_test, y_train_norm, y_test_norm, scaler, target_idx, feature_cols = load_and_preprocess_data(filepath)

    y_train_true = inverse_transform_target(y_train_norm, scaler, target_idx)
    y_test_true = inverse_transform_target(y_test_norm, scaler, target_idx)

    print("\n--- Training BPNN ---")
    bp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
    bp_model.fit(X_train, y_train_norm)
    bp_test_pred = inverse_transform_target(bp_model.predict(X_test), scaler, target_idx)
    evaluate_performance(y_test_true, bp_test_pred, "BPNN")

    print("\n--- Training XGBoost ---")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
    xgb_model.fit(X_train, y_train_norm)
    xgb_test_pred = inverse_transform_target(xgb_model.predict(X_test), scaler, target_idx)
    evaluate_performance(y_test_true, xgb_test_pred, "XGBoost")

    print("\n--- Training LSTM ---")
    lstm_model = LSTMModel(hidden_size=64, dropout=0.3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.003)

    train_x_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    train_y_t = torch.tensor(y_train_norm, dtype=torch.float32)
    test_x_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(train_x_t, train_y_t), batch_size=8, shuffle=True)

    for epoch in range(50):
        lstm_model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(lstm_model(x_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_test_pred_norm = np.squeeze(lstm_model(test_x_t).numpy())
    lstm_test_pred = inverse_transform_target(lstm_test_pred_norm, scaler, target_idx)
    evaluate_performance(y_test_true, lstm_test_pred, "LSTM")

    print("\n--- Ensemble Models Evaluation ---")
    static_fusion_pred = 0.65 * lstm_test_pred + 0.35 * xgb_test_pred
    evaluate_performance(y_test_true, static_fusion_pred, "LSTM-XGB-Static")

    def dynamic_fusion_safe(lstm_pred, xgb_pred, true_values, window_size=5):
        volatility = []
        for i in range(len(true_values)):
            if i < window_size:
                window_data = true_values[0:i + 1]
            else:
                window_data = true_values[i - window_size:i]
            volatility.append(np.std(window_data))

        volatility = np.array(volatility)
        v_min, v_max = volatility.min(), volatility.max()
        volatility_norm = (volatility - v_min) / (v_max - v_min + 1e-8)

        weight_lstm = np.where(volatility_norm < 0.5, 0.8, 0.2)
        weight_xgb = 1 - weight_lstm
        return weight_lstm * lstm_pred + weight_xgb * xgb_pred

    dynamic_fusion_pred = dynamic_fusion_safe(lstm_test_pred, xgb_test_pred, y_test_true)
    evaluate_performance(y_test_true, dynamic_fusion_pred, "LSTM-XGB-Volatility")

    print("\n--- Generating SHAP Explanations ---")
    background_data = X_test[:50]
    explainer = shap.KernelExplainer(bp_model.predict, background_data)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    print("SHAP plot saved as 'shap_summary.png'.")


if __name__ == "__main__":
    main()