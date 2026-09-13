import numpy as np          
import pandas as pd        
from sklearn.preprocessing import MinMaxScaler
import torch



class WeatherDataset:
    def __init__(self, csv_file, input_window=168, output_window=72, split='train'):
        self.df = pd.read_csv(csv_file)
        self.input_window = input_window
        self.output_window = output_window

        # ستون‌های موردنظر (قابل تنظیم)
        self.features = ['temperature', 'relative_humidity', 'wind_speed_10m (km/h)', 'vapour_pressure_deficit (kPa)']
        self.target_col = 'temperature'

        # حذف NaN
        self.df = self.df[self.features].dropna().reset_index(drop=True)

        # نرمال‌سازی
        self.scaler = MinMaxScaler()
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

        # تقسیم داده
        total_len = len(self.df)
        train_len = int(0.8 * total_len)

        if split == 'train':
            self.data = self.df[:train_len].values
        else:
            self.data = self.df[train_len - input_window - output_window:].values  # context حفظ بشه

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_window]
        y = self.data[idx + self.input_window: idx + self.input_window + self.output_window, 0]  # فقط دما
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )