# model_serve.py
from fastapi import FastAPI
import mlflow
import pandas as pd
import os
from pydantic import BaseModel
from typing import List
from datetime import datetime

# Define input schema
class InputData(BaseModel):
    columns: List[str]
    data: List[List[float]]

app = FastAPI()

# Load model from MLflow
model_uri = "models:/IrisClassifier/1"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/predictions.csv"

@app.get("/")
def home():
    return {"message": "IrisClassifier API is running. Use POST /predict."}

@app.post("/predict")
def predict(input: InputData):
    df = pd.DataFrame(input.data, columns=input.columns)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    preds = loaded_model.predict(df)
    mapped = [species_map[p] for p in preds]

    # --- Log to CSV ---
    log_df = df.copy()
    log_df["prediction"] = preds
    log_df["species"] = mapped
    log_df["timestamp"] = datetime.now().isoformat()

    if not os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    return {"predictions": preds.tolist(), "species": mapped}
