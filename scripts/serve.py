from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import uvicorn

app = FastAPI(title="Heart Disease Prediction API",
              description="MLOps Portfolio Project 1",
              version="1.0")

# The registry lives in the SQLite store train.py wrote to — point MLflow at it,
# otherwise it looks in the empty default ./mlruns store.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model at startup.
# MLflow 3.x removed stages ("Production"), so reference the version carrying
# the "champion" alias instead.
MODEL_URI = "models:/HeartDiseaseModel@champion"
model = mlflow.sklearn.load_model(MODEL_URI)
print("Model loaded and ready.")


class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running",
            "status": "healthy"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "HeartDiseaseModel@champion"}


@app.post("/predict")
def predict(patient: PatientData):
    data = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "prediction": int(prediction),
        "result": "Disease Detected" if prediction == 1 else "No Disease",
        "confidence": round(float(probability), 4)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
