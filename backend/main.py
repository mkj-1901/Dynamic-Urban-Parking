from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import sys
import os

# Ensure backend module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pydantic_models import UserInput, PredictionResponse
from predict import predict_price

# === App Initialization ===
app = FastAPI(
    title="Dynamic Urban Parking API",
    version="1.0.0",
    description="🚗 FastAPI backend for smart parking price prediction and rerouting."
)

# === CORS Middleware (for frontend access) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:8501"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_VERSION = '1.0.0'


# === Root Endpoint ===
@app.api_route("/", methods=["GET", "HEAD"], summary="Welcome")
def root():
    return JSONResponse(content={"message": "Welcome to Dynamic Parking Price Prediction API 🚗"})


# === Health Check ===
@app.api_route("/health", methods=["GET", "HEAD"], summary="Health Check")
def health():
    return JSONResponse(content={
        "status": "ok",
        "version": MODEL_VERSION,
        "model_loaded": True
    })



# === Predict Price Endpoint ===
@app.post("/predict", response_model=PredictionResponse, summary="Predict Parking Price")
def predict(user_input: UserInput):
    try:
        result = predict_price(user_input)
        return result
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
