from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Student Performance Predictor")
mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI
# Load model from model registry
model_name = "StudentPerformanceModel"
model_stage = "Staging"  # or "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}")

# Define input schema
class StudentFeatures(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: int
    famsup: int
    paid: int
    activities: int
    nursery: int
    higher: int
    internet: int
    romantic: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int

@app.get("/")
def root():
    return {"message": "Student Performance API is up!"}

@app.post("/predict")
def predict(features: StudentFeatures):
    # Convert to DataFrame
    input_df = pd.DataFrame([features.dict()])
    
    # Make prediction
    pred = model.predict(input_df)
    return {"predicted_final_grade": round(float(pred[0]), 2)}
