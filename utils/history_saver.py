import csv
import os
from datetime import datetime

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "forecast_history.csv")

def save_forecast_to_csv(y_true, y_pred):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = zip(y_true, y_pred)
    
    file_exists = os.path.exists(HISTORY_PATH)

    with open(HISTORY_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "true_value", "predicted_value"])
        for true_val, pred_val in rows:
            writer.writerow([now, true_val, pred_val])
