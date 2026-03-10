import os
import joblib
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "myapp", "models")
READING_TYPES = (
    "power_w",
    "voltage",
    "current_a",
    "energy_kwh",
    "power_factor",
)

asset_models = {}


def parse_model_filename(filename):
    if not filename.endswith("_model.pkl"):
        return None, None

    base_name = filename.replace("_model.pkl", "")

    for reading_type in sorted(READING_TYPES, key=len, reverse=True):
        suffix = f"_{reading_type}"
        if base_name.endswith(suffix):
            asset_name = base_name[: -len(suffix)]
            return asset_name, reading_type

    return None, None


if os.path.exists(MODEL_PATH):

    for file in os.listdir(MODEL_PATH):

        if file.endswith("_model.pkl"):
            asset_name, reading_type = parse_model_filename(file)
            if not asset_name or not reading_type:
                print(f"Skipping unrecognized model filename: {file}")
                continue

            model_path = os.path.join(MODEL_PATH, file)
            scaler_filename = f"{asset_name}_{reading_type}_scaler.pkl"
            scaler_path = os.path.join(MODEL_PATH, scaler_filename)

            if not os.path.exists(scaler_path):
                print(f"⚠ Missing scaler for {file}")
                continue

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if asset_name not in asset_models:
                asset_models[asset_name] = {}

            asset_models[asset_name][reading_type] = {
                "model": model,
                "scaler": scaler
            }

print("Loaded structure:")
for asset in asset_models:
    print(f"{asset} → {list(asset_models[asset].keys())}")
