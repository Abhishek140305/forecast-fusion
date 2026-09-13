import torch
import numpy as np
from utils.weather_fetcher import fetch_real_weather
from model import TimeSeriesTransformer

def prepare_live_input():
    raw = fetch_real_weather()
    if len(raw) < 168:
        raise ValueError("Not enough data (need 168 hours)")

    latest = raw[-168:]  # فقط ۷ روز اخیر
    features = []
    for record in latest:
        temp = record["temperature"]
        hum = record["humidity"]
        wind = record["wind"]
        pres = record["pressure"]
        features.append([temp, hum, wind, pres])

    arr = np.array(features, dtype=np.float32)
    arr = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))  # normalize
    return torch.tensor(arr).unsqueeze(0)  # [1, 168, 4]

def predict_from_live(model, device):
    try:
        x = prepare_live_input().to(device)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    with torch.no_grad():
        preds = model(x.unsqueeze(0)).cpu().numpy().flatten().tolist()
    return preds
