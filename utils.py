import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def inverse_transform_target(y_scaled, scaler, target_col_idx, total_cols=10):

    dummy_array = np.zeros((len(y_scaled), total_cols))
    dummy_array[:, target_col_idx] = y_scaled
    y_original = scaler.inverse_transform(dummy_array)[:, target_col_idx]
    return y_original


def evaluate_performance(y_true, y_pred, model_name="Model"):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE calculation (avoiding division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

    print(f"--- {model_name} Performance ---")
    print(f"RMSE: {rmse:.2f} m3")
    print(f"MAE:  {mae:.2f} m3")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%\n")
    return rmse, mae, r2, mape