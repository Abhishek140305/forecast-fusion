import requests

def fetch_real_weather(latitude=36.3, longitude=59.6):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl"
        f"&past_days=1&forecast_days=1&timezone=auto"
    )
    
    response = requests.get(url)
    data = response.json()

    hourly = data["hourly"]
    
    result = []
    for i in range(len(hourly["time"])):
        record = {
            "time": hourly["time"][i],
            "temperature": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "wind": hourly["wind_speed_10m"][i],
            "pressure": hourly["pressure_msl"][i]
        }
        result.append(record)
    
    return result  # لیستی از دیکشنری‌های لحظه‌ای
