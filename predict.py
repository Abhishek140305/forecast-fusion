import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import TimeSeriesTransformer
from sklearn.metrics import mean_absolute_error

# ⚙️ پارامترها باید مثل train باشن
input_dim = 4
model_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1
output_window = 72
input_window = 168
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 دیتاست تست
dataset = WeatherDataset("./Weather_Data_1980_2024(hourly).csv", input_window, output_window, split="test")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 📥 بارگذاری مدل
model = TimeSeriesTransformer(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    output_window=output_window
).to(device)

model.load_state_dict(torch.load("/checkpoints/final_transformer_model.pth", map_location=device))
model.eval()

# 🔍 پیش‌بینی یک نمونه
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        y = y.numpy().flatten()
        y_pred = model(x).cpu().numpy().flatten()

        # 📊 رسم نمودار
        plt.figure(figsize=(12, 5))
        plt.plot(y, label="Real")
        plt.plot(y_pred, label="Predicted")
        plt.xlabel("Hour")
        plt.ylabel("Temperature")
        plt.title("Transformer Forecast")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("forecast.png")
        plt.show()

        # 🎯 ارزیابی
        mae = mean_absolute_error(y, y_pred)
        print(f"MAE: {mae:.4f}")
        break
