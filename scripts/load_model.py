import mlflow
import mlflow.sklearn
import pandas as pd

# The registry + tracking data live in the SQLite store train.py wrote to,
# so point MLflow at it (the default would be the empty ./mlruns file store).
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the model from the MLflow Model Registry.
# NOTE: MLflow 3.x removed stages ("Production"/"Staging"); the modern
# equivalent is a model ALIAS. We load whichever version carries the
# "champion" alias, so this line never has to change when you promote a
# newer model version later.
model_uri = "models:/HeartDiseaseModel@champion"
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully from MLflow registry.")

# Test prediction with one sample row
sample = pd.DataFrame([{
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 633, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print(f"Prediction: {'Disease' if prediction[0] == 1 else 'No Disease'}")
print(f"Probability: {probability[0][1]:.4f}")
