from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from api.predictor import get_forecast
from utils.weather_fetcher import fetch_real_weather
from api.predictor import get_forecast_history
from live_predictor import predict_from_live
from model import TimeSeriesTransformer
import torch


app = FastAPI()
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/forecast", response_class=JSONResponse)
async def forecast():
    real, forecast = get_forecast()
    return JSONResponse(content={"real": real, "forecast": forecast})
@app.get("/forecast/html", response_class=HTMLResponse)
async def forecast_page(request: Request):
    return templates.TemplateResponse("forecast.html", {"request": request})


@app.get("/realtime-weather")
def get_real_weather():
    weather_data = fetch_real_weather(latitude=35.7, longitude=51.4)
    return {"status": "success", "data": weather_data}


@app.get("/history")
def history_api():
    data = get_forecast_history()
    return {"status": "success", "data": data}





@app.get("/live-forecast")
def live_forecast():
    model = TimeSeriesTransformer(
        input_dim=4,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        output_window=72
    )
    model.load_state_dict(torch.load("checkpoints/final_transformer_model.pth", map_location="cpu"))
    preds = predict_from_live(model, device=torch.device("cpu"))
    if isinstance(preds, dict) and preds.get("status") == "error":
        return preds
    return {"status": "success", "forecast": preds}