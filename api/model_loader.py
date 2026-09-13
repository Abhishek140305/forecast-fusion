import torch
from model import TimeSeriesTransformer
from dataset import WeatherDataset
from torch.utils.data import DataLoader
import pandas as pd
import os






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

def get_forecast():
    dataset = WeatherDataset(CSV_PATH, input_window, output_window, split="test")
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x).cpu().numpy().flatten()

    # denormalize
    y_pred = y_pred * (max_temp - min_temp) + min_temp
    y_true = y.numpy().flatten() * (max_temp - min_temp) + min_temp

    return y_true.tolist(), y_pred.tolist()
