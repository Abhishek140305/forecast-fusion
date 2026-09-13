import torch
from model import TimeSeriesTransformer
from dataset import WeatherDataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from api.model_loader import get_forecast
from utils.history_saver import save_forecast_to_csv
import csv





# 📁 مسیرهای پروژه
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "final_transformer_model.pth")
CSV_PATH = os.path.join(BASE_DIR, "Weather_Data_1980_2024(hourly).csv")

# ⚙️ مشخصات مدل
input_dim = 4
model_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1
output_window = 72
input_window = 168

device = torch.device("cpu")

# 🧠 لود مدل
model = TimeSeriesTransformer(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    output_window=output_window
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 📊 لود داده‌ها برای denormalization
df = pd.read_csv(CSV_PATH)
min_temp = df["temperature"].min()
max_temp = df["temperature"].max()

# 📈 تابع پیش‌بینی نهایی با denormalization

def predict_temperature():
    y_true, y_pred = get_forecast()
    return {
        "status": "success",
        "prediction": y_pred,
        "actual": y_true
    }
    
    
    
    
y_true, y_pred = get_forecast()


def get_forecast_history():
    path = os.path.join(os.path.dirname(__file__), "..", "forecast_history.csv")
    if not os.path.exists(path):
        return []

    history = []
    with open(path, "r") as f:
        next(f)  # Skip header
        for line in f:
            timestamp, true_val, pred_val = line.strip().split(",")
            history.append({
                "timestamp": timestamp,
                "true": float(true_val),
                "pred": float(pred_val)
            })
    return history