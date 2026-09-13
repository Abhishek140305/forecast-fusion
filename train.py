import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from dataset import WeatherDataset
from model import TimeSeriesTransformer

# 📌 هایپرپارامترها
input_dim = 4
model_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1
output_window = 72
input_window = 168
batch_size = 64
epochs = 5
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = WeatherDataset("./Weather_Data_1980_2024(hourly).csv", input_window, output_window, split='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 🧠 مدل
model = TimeSeriesTransformer(
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    output_window=output_window
).to(device)

# 🎯 تابع خطا و بهینه‌ساز
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 📁 مسیر ذخیره
os.makedirs("checkpoints", exist_ok=True)

# 🔁 آموزش
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)                 # [B, T_in, input_dim]
        y = y.to(device)                 # [B, output_window]

        optimizer.zero_grad()
        output = model(x)               # [B, output_window]
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # 💾 ذخیره مدل نهایی
    if epoch == epochs:
        torch.save(model.state_dict(), "checkpoints/final_transformer_model.pth")