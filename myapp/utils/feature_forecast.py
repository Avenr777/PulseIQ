import pandas as pd
import numpy as np


def build_forecast_features(history_df, future_time, asset_encoded, reading_encoded):
    """
    history_df must contain SCALED values.
    """

    latest = history_df.tail(6)

    features = {}

    # ---- LAGS (1–6) ----
    for i in range(1, 7):
        features[f"lag_{i}"] = latest.iloc[-i]["value"]

    # ---- Rolling stats ----
    features["rolling_mean_6"] = latest["value"].mean()
    features["rolling_std_6"] = latest["value"].std()

    # ---- Time features ----
    features["hour"] = future_time.hour
    features["day_of_week"] = future_time.dayofweek
    features["month"] = future_time.month
    features["is_weekend"] = int(future_time.dayofweek >= 5)

    # ---- Encoded IDs ----
    features["asset_encoded"] = asset_encoded
    features["reading_encoded"] = reading_encoded

    return pd.DataFrame([features])