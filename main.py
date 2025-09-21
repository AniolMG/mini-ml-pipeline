from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
import os


print("Starting...")
# -------------------------------
# Define input schema
# -------------------------------
class TitanicInput(BaseModel):
    Age: float
    Sex: int  # 0=male, 1=female
    Pclass: int

# -------------------------------
# Load MLflow model from Model Registry
# -------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "TitanicModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")  # Default to "Staging"

print("Loaded execution variables!")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("Set tracking URI!")

# MLflow Model Registry URI format: models:/<model_name>/<stage_or_version>
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
print(model_uri)
model = mlflow.pyfunc.load_model(model_uri)

print("Loaded model!")

# Print dependencies for debugging
deps = mlflow.pyfunc.get_model_dependencies(model_uri)
print("Dependencies:", deps)

# -------------------------------
# Start FastAPI app
# -------------------------------
app = FastAPI(title="Titanic Survival Prediction API")

print("Started FastAPI server!")

@app.post("/predict")
def predict(data: TitanicInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    # Make prediction
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
