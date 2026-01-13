# backend_forecast.py

import joblib
import numpy as np
import pandas as pd
from math import sqrt

# ------------ CONFIG ------------
TARGET_COL = "ALLSKY_SFC_SW_DWN"   # same as training
N_HISTORY_HOURS = 24               # how many past hours to keep for lags/rolling

# ------------ 1. Helper: build timestamp + time features ------------

def add_time_features(df, timestamp_col=None):
    """
    Ensure DataFrame has a proper datetime index and add time-based features:
    - hour, dayofyear
    - hour_sin, hour_cos
    - day_sin, day_cos
    """
    if timestamp_col is not None:
        df["timestamp"] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values("timestamp").set_index("timestamp")
    else:
        # assume YEAR, MO, DY, HR are present
        df["timestamp"] = pd.to_datetime(
            {
                "year": df["YEAR"],
                "month": df["MO"],
                "day": df["DY"],
                "hour": df["HR"],
            }
        )
        df = df.sort_values("timestamp").set_index("timestamp")

    df["hour"] = df.index.hour
    df["dayofyear"] = df.index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df


# ------------ 2. Helper: load last history from training data ------------

def load_last_history(training_csv_path):
    """
    Load training data and return the last N_HISTORY_HOURS of true irradiance
    as a list to seed lag features for future predictions.
    Assumes training CSV has the same structure as before:
    YEAR, MO, DY, HR, ALLSKY_SFC_SW_DWN, T2M, RH2M, PS, WS10M, T2MDEW, SZA
    """
    df_train = pd.read_csv(training_csv_path)

    # Build timestamp and sort
    df_train["timestamp"] = pd.to_datetime(
        {
            "year": df_train["YEAR"],
            "month": df_train["MO"],
            "day": df_train["DY"],
            "hour": df_train["HR"],
        }
    )
    df_train = df_train.sort_values("timestamp").set_index("timestamp")

    # Ensure target is numeric
    df_train[TARGET_COL] = pd.to_numeric(df_train[TARGET_COL], errors="coerce")
    df_train = df_train.dropna(subset=[TARGET_COL])

    # Take last N_HISTORY_HOURS values
    history = df_train[TARGET_COL].iloc[-N_HISTORY_HOURS:].tolist()
    if len(history) < N_HISTORY_HOURS:
        raise ValueError("Not enough history in training data to build lags.")

    return history


# ------------ 3. Helper: load RF model package ------------

def load_model(model_path):
    """
    Load the RF model package from pkl.
    The package is expected to be a dict with:
    - 'model': trained RandomForestRegressor
    - 'features': list of feature names used during training
    - 'target_name': name of target column
    """
    pkg = joblib.load(model_path)
    model = pkg["model"]
    feature_names = pkg["features"]
    target_name = pkg.get("target_name", TARGET_COL)
    return model, feature_names, target_name


# ------------ 4. Main function: predict from future file + compute energy ------------

def predict_from_future_file(
    future_csv_path,
    training_csv_path,
    model_path,
    num_panels,
    panel_area,
    panel_efficiency,
    performance_ratio=1.0
):
    """
    Main backend function:
    - Load RF model from pkl
    - Load future month CSV (no irradiance column)
    - Apply the same preprocessing as training (time features)
    - Recursively predict irradiance for each hour using lag-based history
    - Compute energy per hour and total kWh

    Parameters:
    -----------
    future_csv_path : str
        Path to CSV file containing the next-month data
        (must include YEAR, MO, DY, HR, T2M, RH2M, PS, WS10M, T2MDEW, SZA)
    training_csv_path : str
        Path to original training data CSV (mydata.csv)
        used to get last N_HISTORY_HOURS of true irradiance.
    model_path : str
        Path to rf_best_model.pkl
    panel_area : float
        Total panel area in m^2
    panel_efficiency : float
        Panel efficiency (0–1)
    performance_ratio : float
        Performance ratio (0–1), default 1.0

    Returns:
    --------
    result : dict
        {
            "total_energy_kwh": float,
            "predictions": DataFrame with columns:
                [timestamp, predicted_irradiance, energy_kwh]
        }
    """

    # 1) Load model and its feature names
    model, feature_names, _ = load_model(model_path)

    # 2) Load future month file
    df_future = pd.read_csv(future_csv_path)

    # 3) Add time features (using YEAR, MO, DY, HR)
    df_future = add_time_features(df_future)

    # 4) Load last N_HISTORY_HOURS real irradiance values from training data
    history = load_last_history(training_csv_path)
    # history is a list of length N_HISTORY_HOURS

    # Prepare container for predictions
    preds = []

    # 5) Recursive forecasting over future timestamps (sorted)
    for ts, row in df_future.iterrows():
        # Extract known exogenous inputs at this timestamp
        feature_values = {}

        # Weather features
        for col in ["T2M", "RH2M", "PS", "WS10M", "T2MDEW", "SZA"]:
            if col in df_future.columns:
                feature_values[col] = float(row[col])

        # Time encodings
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            if col in df_future.columns:
                feature_values[col] = float(row[col])

        # Lag features based on history of predicted/actual irradiance
        # history[-1] = last hour, history[-2] = 2 hours ago, etc.
        feature_values["lag_1h"] = history[-1]
        feature_values["lag_2h"] = history[-2]
        feature_values["lag_3h"] = history[-3]
        feature_values["lag_6h"] = history[-6]
        feature_values["lag_24h"] = history[-24]

        # Rolling mean of last 24 hours
        feature_values["rolling_24h_mean"] = float(np.mean(history[-24:]))

        # Build feature row as a DataFrame with the same feature names as during training
        x_df = pd.DataFrame([feature_values])[feature_names]

        # Predict irradiance for this timestamp
        y_hat = float(model.predict(x_df)[0])

        # Append prediction to history so that next step can use it as lag
        history.append(y_hat)

        # Compute energy for this hour (kWh)
        # E_hour = A * η * PR * G(t) / 1000
        # Total panel area = number of panels * area per panel
        total_area = num_panels * panel_area

        # E_hour = A_total * η * PR * G(t) / 1000
        e_hour = (total_area * panel_efficiency * performance_ratio * y_hat) / 1000.0

        preds.append(
            {
                "timestamp": ts,
                "predicted_irradiance": y_hat,
                "energy_kwh": e_hour,
            }
        )

    # 6) Build predictions DataFrame
    preds_df = pd.DataFrame(preds).set_index("timestamp")

    # 7) Compute total energy over the whole future period
    total_energy_kwh = preds_df["energy_kwh"].sum()

    result = {
        "total_energy_kwh": float(total_energy_kwh),
        "predictions": preds_df,
    }

    return result


# ------------ 5. Example usage (for debugging or CLI) ------------

if __name__ == "__main__":
    FUTURE_CSV = "future_month.csv"
    TRAIN_CSV = "mydata.csv"
    MODEL_PKL = "rf_best_model.pkl"

    NUM_PANELS = 12
    PANEL_AREA = 1.7
    PANEL_EFF  = 0.18
    PR         = 0.9

    result = predict_from_future_file(
        future_csv_path=FUTURE_CSV,
        training_csv_path=TRAIN_CSV,
        model_path=MODEL_PKL,
        num_panels=NUM_PANELS,
        panel_area=PANEL_AREA,
        panel_efficiency=PANEL_EFF,
        performance_ratio=PR
    )
    print(f"Total predicted energy (kWh): {result['total_energy_kwh']:.2f}")

