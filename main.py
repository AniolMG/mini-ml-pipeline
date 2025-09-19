from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel

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
mlflow.set_tracking_uri("http://127.0.0.1:5000")

REGISTERED_MODEL_NAME = "TitanicModel"  # Name in Model Registry
MODEL_STAGE = "Staging"              # Can also be "Staging"

# MLflow Model Registry URI format: models:/<model_name>/<stage_or_version>
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

# Print dependencies for debugging
deps = mlflow.pyfunc.get_model_dependencies(model_uri)
print("Dependencies:", deps)

# -------------------------------
# Start FastAPI app
# -------------------------------
app = FastAPI(title="Titanic Survival Prediction API")

@app.post("/predict")
def predict(data: TitanicInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    # Make prediction
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
