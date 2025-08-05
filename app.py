# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Credit Risk API",
    description="Predict loan default risk from customer data",
    version="1.0"
)

# Serve static files (in case you add CSS/JS later)
# app.mount("/static", StaticFiles(directory="."), name="static")

# Allow frontend to communicate (important for browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
MODEL_PATH = "credit_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found. Did you upload it?")
if not os.path.exists(SCALER_PATH):
    raise RuntimeError(f"Scaler file '{SCALER_PATH}' not found. Did you upload it?")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# Define input schema
class CreditData(BaseModel):
    limit_balance: float
    sex: int  # 1=Male, 2=Female
    education: int  # 1=Graduate, 2=University, 3=High School, 4=Others
    marriage: int  # 1=Married, 2=Single, 3=Others
    age: int
    pay_0: int  # Payment status: -1=pay duly, 1=1 month delay, etc.
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float

# ✅ Serve index.html at root
@app.get("/")
def serve_frontend():
    if not os.path.exists("index.html"):
        return {"error": "index.html not found. Make sure it's in the root directory."}
    return FileResponse("index.html")

# Prediction endpoint
@app.post("/predict")
def predict(data: CreditData):
    try:
        # Convert input to array in correct order (must match training!)
        input_data = np.array([[
            data.limit_balance, data.sex, data.education, data.marriage, data.age,
            data.pay_0, data.pay_2, data.pay_3, data.pay_4, data.pay_5, data.pay_6,
            data.bill_amt1, data.bill_amt2, data.bill_amt3, data.bill_amt4,
            data.bill_amt5, data.bill_amt6, data.pay_amt1, data.pay_amt2, data.pay_amt3,
            data.pay_amt4, data.pay_amt5, data.pay_amt6
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        risk_level = "High" if probability > 0.5 else "Low"
        recommendation = "Loan Denied" if risk_level == "High" else "Loan Approved"

        return {
            "prediction": int(prediction),
            "default_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
