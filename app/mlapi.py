from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
from typing import Dict
import joblib
import uvicorn
import nest_asyncio

app = FastAPI(
    title="Sepsis Prediction API",
    description="This will predict Sepsis"
)

MODEL_PATH = {
    "random_forest": "random_forest_model.pkl",
    "bagging_classifier": "bagging_classifier_model.pkl",
    "gradient_boosting": "gradient_boosting_model.pkl",
    "extra_trees": "extra_trees_model.pkl"
}

# Load the Models
models = {}
for model, path in MODEL_PATH.items():
    try:
        models[model] = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model} from {path}. Error: {e}")

@app.get("/")
async def st_endpoint():
    return {"message": "Welcome to Sepsis Prediction App"}

@app.post("/predict")
async def predictor(model: str, file: UploadFile = File(...)):
    if model not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the file: {e}")

    try:
        required_features = models[model].n_features_in_
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="The loaded model does not have the attribute 'n_features_in_'. Ensure the model was trained and saved correctly."
        )

    if len(df.columns) != required_features:
        raise HTTPException(
            status_code=400,
            detail=f"The model expects {required_features} features, but the file has {len(df.columns)} columns"
        )

    # features = df.values
    model_instance = models[model]
    predictions = model_instance.predict(df)
    results = {
        "model_used": model,
        "predictions": predictions.tolist()
    }
    return results

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app)
