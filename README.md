<div align="center">

# 🌦️✨ Weather Transformer Forecast API

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async%20web%20API-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-red?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

<h3>🌍 Accurate, Modern, and Interactive Weather Forecasting with Deep Learning Transformers</h3>

<img src="forecast.png" alt="Forecast Example" width="80%"/>

</div>

---

## ✨ Overview

**Weather Transformer Forecast API** is a full-stack, deep learning-powered weather forecasting platform. It leverages a Transformer neural network to predict future temperature trends from historical and real-time weather data, and provides:

- 🚀 **FastAPI** backend for blazing-fast, async predictions
- 📊 **Interactive dashboard** for visualizing forecasts
- 🧠 **Transformer model** for state-of-the-art time series prediction
- 🌐 **RESTful API** for integration and automation

---

## 🗂️ Project Structure

```text
📁 time_series_transformer_project/
│
├── api/
│   ├── main.py                # FastAPI app and routes
│   ├── predictor.py           # Model inference and forecast logic
│   ├── model_loader.py        # Model loading utilities
│   ├── weather_api.py         # Weather data fetching from external API
│   ├── static/
│   │   └── css/style.css      # Dashboard styles
│   ├── templates/
│   │   ├── index.html         # Home page
│   │   └── forecast.html      # Forecast dashboard
│   └── final_transformer_model.pth # Trained model weights
│
├── dataset.py                 # WeatherDataset class for data loading
├── model.py                   # Transformer model definition
├── train.py                   # Model training script
├── predict.py                 # Batch prediction and evaluation
├── live_predictor.py          # Real-time prediction logic
├── utils/
│   ├── weather_fetcher.py     # Real-time weather data utilities
│   └── history_saver.py       # Save forecast history
├── checkpoints/
│   └── final_transformer_model.pth # Model checkpoint
├── Weather_Data_1980_2024(hourly).csv # Historical weather data
├── forecast.png               # Example forecast plot
├── run_api.py                 # API runner script
├── render.yaml                # Render.com deployment config
└── .gitignore
```

---

## 🧠 Model Highlights

| Feature                | Description                                  |
|-----------------------|----------------------------------------------|
| **Architecture**      | Transformer Encoder                          |
| **Input**             | Last 168 hours (7 days) of weather features  |
| **Output**            | Next 72 hours (3 days) of temperature        |
| **Features Used**     | Temperature, Humidity, Wind, VPD             |
| **Framework**         | PyTorch                                      |

---

## 🚦 Quickstart
```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Model
- Ensure `Weather_Data_1980_2024(hourly).csv` is in the project root.
- (Optional) Train your model:

```bash
python train.py
```

### 4. Run the API

```bash
python run_api.py
# or
uvicorn api.main:app --reload
```

---

## 🌐 API Endpoints

| Route                | Method | Description                                 |
|----------------------|--------|---------------------------------------------|
| `/`                  | GET    | Home page (HTML)                            |
| `/forecast`          | GET    | Get 72-hour forecast (JSON)                 |
| `/forecast/html`     | GET    | Forecast dashboard (HTML)                   |
| `/realtime-weather`  | GET    | Fetch current weather from API (JSON)       |
| `/history`           | GET    | Get forecast history (JSON)                 |
| `/live-forecast`     | GET    | Predict using latest real-time data (JSON)  |

---

## 📊 Dashboard Preview

- Visit [`/forecast/html`](http://localhost:8000/forecast/html) for an interactive forecast chart.
- The dashboard uses Chart.js for beautiful, responsive plots.

<img src="api/static/forecast_dashboard.png" alt="Dashboard Preview" width="90%"/>

---

## 📝 Example Usage

```bash
# Get JSON forecast
curl http://localhost:8000/forecast

# Get real-time weather
curl http://localhost:8000/realtime-weather
```

---

## 🛠️ Customization

- **Change Model/Features**: Edit `model.py` and `dataset.py`.
- **Add New Endpoints**: Edit `api/main.py`.
- **Switch Data Source**: Update `utils/weather_fetcher.py` or `api/weather_api.py`.

---

## ☁️ Deployment

- Ready for deployment on [Render.com](https://render.com/) (see `render.yaml`).
- Can be deployed on any platform supporting Python and FastAPI.

---

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome!<br>
Please open an issue or submit a PR for improvements.

---

## 📄 License

MIT License.<br>
See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Chart.js](https://www.chartjs.org/)
- [Visual Crossing Weather API](https://www.visualcrossing.com/)

---

<div align="center">

**Made with ❤️ for weather forecasting and deep learning.**

</div>
