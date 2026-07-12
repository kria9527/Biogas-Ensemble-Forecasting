import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp


def load_and_preprocess_data(filepath, seed=42):
    print("Loading data...")
    df = pd.read_excel(filepath)
    a_df = df.drop(df.columns[[0, 1, 2]], axis=1)

    biogas_col = 'Biogas (m3)'
    target_idx = a_df.columns.get_loc(biogas_col)

    biogas_mean = a_df[biogas_col].mean()
    biogas_std = a_df[biogas_col].std()
    lower_bound = biogas_mean - 2 * biogas_std
    upper_bound = biogas_mean + 2 * biogas_std

    outlier_indices = a_df[(a_df[biogas_col] < lower_bound) | (a_df[biogas_col] > upper_bound)].index
    filtered_df = a_df.drop(outlier_indices).reset_index(drop=True)

    print(f"Original samples: {len(a_df)}, Cleaned samples: {len(filtered_df)}")

    columns_to_analyze = ['Temperature', 'Material (m3)', 'Water (kt)',
                          'Electricity (kWh)', 'TAC (mg·L-1)',
                          'VFA (mg·L-1)', 'CH4 (%)', 'Residue (kg)',
                          'Slurry (m3)', 'Biogas (m3)']

    correlation_matrix = filtered_df[columns_to_analyze].corr(method='spearman')
    print('\nSpearman correlation matrix generated.')

    mms = sp.MinMaxScaler(feature_range=(-10, 10))
    scaled_data = mms.fit_transform(filtered_df)
    b_df = pd.DataFrame(scaled_data, columns=filtered_df.columns, index=filtered_df.index)

    feature_cols = [col for col in b_df.columns if col != biogas_col]
    X = b_df[feature_cols].values
    y = b_df[biogas_col].values

    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    train_indices = indices[:300]
    test_indices = indices[300:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test, mms, target_idx, feature_cols