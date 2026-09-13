# api/weather_api.py

import requests
import os
from datetime import datetime, timedelta

API_KEY = os.getenv("WEATHER_API_KEY") or "5311df6ce03d74d41c833ce71925fcf4"  # ← کلیدتو اینجا بذار

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

def fetch_weather_data(location="Tehran", days=7, include="hours"):
    """
    گرفتن داده آب‌وهوا برای یک موقعیت خاص در بازه زمانی مشخص (پیش‌فرض: 7 روز آینده)
    """
    end_date = (datetime.today() + timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"{BASE_URL}/{location}/today/{end_date}"

    params = {
        "unitGroup": "metric",
        "key": API_KEY,
        "contentType": "json",
        "include": include
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def extract_temperature_series(json_data):
    """
    تبدیل داده‌ها به دمای hourly در یک لیست ساده
    """
    temps = []
    for day in json_data.get("days", []):
        for hour in day.get("hours", []):
            temps.append(hour["temp"])
    return temps
