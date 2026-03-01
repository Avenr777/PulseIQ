import os
import joblib
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "myapp", "models")

forecast_model = joblib.load(
    os.path.join(MODEL_PATH, "global_forecast_model.pkl")
)

asset_encoder = joblib.load(
    os.path.join(MODEL_PATH, "asset_encoder.pkl")
)

reading_encoder = joblib.load(
    os.path.join(MODEL_PATH, "reading_encoder.pkl")
)
reading_type_scalers = joblib.load(
    os.path.join(MODEL_PATH, "reading_type_scalers.pkl")
)