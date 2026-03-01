from django.http import Http404, HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from .ml_models import  asset_models
import numpy as np
from datetime import datetime
import json
import random
from .models import Asset, PowerReading, AnomalyLog
from myapp.utils.feature_engineering import build_feature_vector
from .simulator import ASSET_CONFIG, READING_TYPES, cascade_state
from .ml_models import asset_models
from .ml_forecast import forecast_model, asset_encoder, reading_encoder, reading_type_scalers
import pandas as pd
from .utils.feature_forecast import build_forecast_features
@login_required
def home(request):
    return render(request, 'index.html')
@login_required
def water(request):
    return render(request, 'water.html')

def electricity(request):

    assets_data = []
    cascade_risk = 0

    selected_asset_name = request.GET.get("asset")
    selected_reading_type = request.GET.get("reading_type", "power_w")

    all_assets = Asset.objects.all()
    READING_TYPES = [
    "power_w",
    "voltage",
    "current_a",
    "energy_kwh",
    "power_factor",
    ]
    if not all_assets.exists():
        return render(request, "electricity.html", {"assets": []})

    # If no asset selected → default to first
    if not selected_asset_name:
        selected_asset_name = all_assets.first().name

    # -------------------------------------------------
    # LOOP THROUGH ALL ASSETS (DYNAMIC)
    # -------------------------------------------------

    for asset_obj in all_assets:

        asset_name = asset_obj.name

        latest_reading = PowerReading.objects.filter(
            asset=asset_obj,
            reading_type=selected_reading_type
        ).order_by("-timestamp").first()

        if not latest_reading:
            continue

        value = latest_reading.value

        # Last 6 readings
        recent_readings = list(
            PowerReading.objects.filter(
                asset=asset_obj,
                reading_type=selected_reading_type
            ).order_by("-timestamp")[:6]
        )

        recent_readings.reverse()
        values_list = [r.value for r in recent_readings]

        status = "Normal"

        if len(values_list) >= 6:

            previous_value = values_list[-2]

            feature_vector = build_feature_vector(
                current_value=value,
                previous_value=previous_value,
                history_values=values_list,
                reading_type=selected_reading_type
            )

            # Nested model access
            asset_model = asset_models.get(asset_name, {}).get(selected_reading_type)

            if asset_model:
                model = asset_model["model"]
                scaler = asset_model["scaler"]

                scaled = scaler.transform(feature_vector)
                prediction = model.predict(scaled)[0]

                if prediction == -1:
                    status = "Anomaly"

                    # Avoid duplicate logs
                    if not AnomalyLog.objects.filter(
                        reading=latest_reading,
                        reading_type=selected_reading_type
                    ).exists():

                        AnomalyLog.objects.create(
                            asset=asset_obj,
                            reading=latest_reading,
                            reading_type=selected_reading_type,
                            value=value,
                            severity=2
                        )

                    # Simple cascade rule
                    if asset_name == "transformer_block_a":
                        cascade_risk = 1
                else:
                    status = "Normal"
            else:
                status = "Unknown"

        assets_data.append({
            "id": asset_name,
            "value": round(value, 2),
            "reading_type": selected_reading_type,
            "status": status,
            "updated": latest_reading.timestamp.strftime("%H:%M:%S")
        })

    # -------------------------------------------------
    # CHART SECTION
    # -------------------------------------------------

    try:
        selected_asset = Asset.objects.get(name=selected_asset_name)
    except Asset.DoesNotExist:
        raise Http404("Asset not found")

    last_24 = list(
        PowerReading.objects.filter(
            asset=selected_asset,
            reading_type=selected_reading_type
        ).order_by("-timestamp")[:24]
    )

    last_24.reverse()

    labels = []
    values = []
    anomalies = []

    for reading in last_24:
        labels.append(reading.timestamp.strftime("%H:%M:%S"))
        values.append(reading.value)

        is_anomaly = AnomalyLog.objects.filter(
            reading=reading,
            reading_type=selected_reading_type
        ).exists()

        anomalies.append(-1 if is_anomaly else 1)

    anomaly_count = anomalies.count(-1)
    risk_score = min(100, anomaly_count * 15)

    context = {
        "assets": assets_data,
        "labels": json.dumps(labels),
        "values": json.dumps(values),
        "anomalies": json.dumps(anomalies),
        "anomaly_count": anomaly_count,
        "risk_score": risk_score,
        "cascade_risk": cascade_risk,
        "selected_asset": selected_asset_name,
        "selected_reading_type": selected_reading_type,
        "reading_types": READING_TYPES,
    }

    return render(request, "electricity.html", context)

def electricity_forecast(request):

    selected_asset = request.GET.get("asset")
    selected_reading_type = request.GET.get("reading_type", "power_w")
    steps = int(request.GET.get("steps", 24))

    assets = Asset.objects.all()

    labels = []
    predictions = []
    history_real_values = []
    history_labels = []

    if selected_asset:

        asset_obj = Asset.objects.get(name=selected_asset)

        # Get last 24 readings
        history_qs = PowerReading.objects.filter(
            asset=asset_obj,
            reading_type=selected_reading_type
        ).order_by("-timestamp")[:24]

        history_qs = list(history_qs)[::-1]

        if len(history_qs) < 24:
            return HttpResponse("Need at least 24 historical points.")

        history_df = pd.DataFrame([{
            "timestamp": r.timestamp,
            "value": r.value
        } for r in history_qs])

        history_real_values = history_df["value"].tolist()
        history_labels = [
            r.timestamp.strftime("%H:%M") for r in history_qs
        ]

        # Scaling
        scaler = reading_type_scalers[selected_reading_type]
        history_df["value"] = scaler.transform(
            history_df[["value"]]
        )

        history_scaled = history_df.copy()

        asset_encoded = asset_encoder.transform([selected_asset])[0]
        reading_encoded = reading_encoder.transform([selected_reading_type])[0]

        expected_features = forecast_model.feature_name_

        # Recursive forecast
        for step in range(steps):

            latest = history_scaled.tail(24)

            features = {}

            for i in range(1, 13):
                features[f"lag_{i}"] = latest.iloc[-i]["value"]

            features["rolling_mean_6"] = latest.tail(6)["value"].mean()
            features["rolling_mean_12"] = latest.tail(12)["value"].mean()
            features["rolling_mean_24"] = latest.tail(24)["value"].mean()
            features["rolling_std_6"] = latest.tail(6)["value"].std()

            future_time = history_qs[-1].timestamp + pd.Timedelta(minutes=step+1)

            features["hour"] = future_time.hour
            features["day_of_week"] = future_time.weekday()
            features["asset_encoded"] = asset_encoded
            features["reading_encoded"] = reading_encoded

            X_input = pd.DataFrame([features])
            X_input = X_input.reindex(columns=expected_features)

            # -------- Model Prediction --------
            model_pred = forecast_model.predict(X_input)[0]

            # -------- Stabilization (Solution 2) --------
            rolling_mean = latest.tail(6)["value"].mean()

            # Blend 70% model + 30% rolling mean
            next_scaled = 0.7 * model_pred + 0.3 * rolling_mean

            # Optional safety clamp (recommended)
            max_allowed = rolling_mean * 1.3
            min_allowed = rolling_mean * 0.7
            next_scaled = max(min(next_scaled, max_allowed), min_allowed) 

            # Convert back to real scale
            next_real = scaler.inverse_transform([[next_scaled]])[0][0]

            predictions.append(float(next_real))
            labels.append(future_time.strftime("%H:%M"))

            history_scaled = pd.concat([
                history_scaled,
                pd.DataFrame([{
                    "timestamp": future_time,
                    "value": next_scaled
                }])
            ], ignore_index=True)

    context = {
        "assets": assets,
        "history": json.dumps(history_real_values),
        "history_labels": json.dumps(history_labels),
        "predictions": json.dumps(predictions),
        "labels": json.dumps(labels),
        "selected_asset": selected_asset,
        "selected_reading_type": selected_reading_type,
        "steps": steps
    }

    return render(request, "e_forecast.html", context)
def land(request):
    return render(request, 'land.html')
def register(request):
    form = UserCreationForm()

    for field in form.visible_fields():
        field.field.widget.attrs['class'] = (
            'w-full p-3 rounded-lg bg-slate-700 text-white '
            'border border-slate-600 focus:outline-none '
            'focus:ring-2 focus:ring-blue-500'
        )

    if request.method == "POST":
        form = UserCreationForm(request.POST)

        for field in form.visible_fields():
            field.field.widget.attrs['class'] = (
                'w-full p-3 rounded-lg bg-slate-700 text-white '
                'border border-slate-600 focus:outline-none '
                'focus:ring-2 focus:ring-blue-500'
            )

        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect("home")

    return render(request, "register.html", {"form": form})

def login(request):
    form = AuthenticationForm()

    for field in form.visible_fields():
        field.field.widget.attrs['class'] = (
            'w-full p-3 rounded-lg bg-slate-700 text-white '
            'border border-slate-600 focus:outline-none '
            'focus:ring-2 focus:ring-blue-500'
        )

    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)

        for field in form.visible_fields():
            field.field.widget.attrs['class'] = (
                'w-full p-3 rounded-lg bg-slate-700 text-white '
                'border border-slate-600 focus:outline-none '
                'focus:ring-2 focus:ring-blue-500'
            )

        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect("home")
        else:
            return render(request, "login.html", {
                "form": form,
                "error": "Incorrect username or password!"
            })

    return render(request, "login.html", {"form": form})

def logout(request):
    auth_logout(request)
    return redirect("home")