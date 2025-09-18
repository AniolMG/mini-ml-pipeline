from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
import os

# Define the input schema
class TitanicInput(BaseModel):
    Age: float
    Sex: int  # 0=male, 1=female
    Pclass: int


# Load MLflow model

run_id = os.getenv("RUN_ID", "default_run_id")
#run_id = "0f4d4b43e273443e96cab94c846530b2"
model_uri = f"./saved_model/{run_id}"

deps = mlflow.pyfunc.get_model_dependencies(model_uri)
print("Dependencies:", deps)

model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(title="Titanic Survival Prediction API")

@app.post("/predict")
def predict(data: TitanicInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    # Make prediction
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
