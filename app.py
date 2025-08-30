# ### app.py for vercel deployment #####

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import numpy as np
# # import pandas as pd
# import os

# app = FastAPI(title="ML Model API", version="1.0.0")

# # Load model and feature names
# try:
#     model = joblib.load('model/model.joblib')
#     feature_names = joblib.load('model/feature_names.joblib')
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     model = None
#     feature_names = []

# # Define input schema
# class PredictionInput(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# class BatchPredictionInput(BaseModel):
#     data: list[list[float]]  # List of feature arrays

# @app.get("/")
# async def root():
#     return {
#         "message": "ML Model API is running!",
#         "endpoints": {
#             "health": "/health",
#             "single_prediction": "/predict",
#             "batch_prediction": "/predict-batch",
#             "model_info": "/model-info"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "model_type": "RandomForestClassifier" if model else "None"
#     }

# @app.get("/model-info")
# async def model_info():
#     if not model:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     return {
#         "model_type": str(type(model).__name__),
#         "feature_names": feature_names,
#         "n_features": len(feature_names),
#         "n_classes": getattr(model, 'n_classes_', 'Unknown')
#     }

# @app.post("/predict")
# async def predict_single(input_data: PredictionInput):
#     if not model:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     try:
#         # Convert input to array
#         features = np.array([
#             input_data.sepal_length,
#             input_data.sepal_width,
#             input_data.petal_length,
#             input_data.petal_width
#         ]).reshape(1, -1)
        
#         # Make prediction
#         prediction = model.predict(features)
#         probabilities = model.predict_proba(features)
        
#         return {
#             "prediction": int(prediction[0]),
#             "probabilities": probabilities[0].tolist(),
#             "class_names": ["setosa", "versicolor", "virginica"],  # Replace with your class names
#             "input_features": input_data.dict()
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# @app.post("/predict-batch")
# async def predict_batch(input_data: BatchPredictionInput):
#     if not model:
#         raise HTTPException(status_code=500, detail="Model not loaded")
    
#     try:
#         # Convert to numpy array
#         features = np.array(input_data.data)
        
#         # Validate input shape
#         if features.shape[1] != len(feature_names):
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Expected {len(feature_names)} features, got {features.shape[1]}"
#             )
        
#         # Make predictions
#         predictions = model.predict(features)
#         probabilities = model.predict_proba(features)
        
#         return {
#             "predictions": predictions.tolist(),
#             "probabilities": probabilities.tolist(),
#             "batch_size": len(predictions)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

# # For local development
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


############ Via HuggingFace ################
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import os

app = FastAPI(title="ML Model API on Hugging Face", version="1.0.0")

# Hugging Face model repository
HF_REPO_ID = "SagarChhabriya/ml-model-api"  
MODEL_FILENAME = "model.joblib"
FEATURE_NAMES_FILENAME = "feature_names.joblib"

# Load model from Hugging Face Hub
def load_model_from_hf():
    try:
        print("📥 Downloading model from Hugging Face Hub...")
        
        # Download model file
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME
        )
        
        # Download feature names file
        feature_names_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=FEATURE_NAMES_FILENAME
        )
        
        # Load files
        model = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
        
        print("Model and feature names loaded successfully from Hugging Face!")
        return model, feature_names
        
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        return None, []

# Load model on startup
model, feature_names = load_model_from_hf()

# Define input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchPredictionInput(BaseModel):
    data: list[list[float]]  # List of feature arrays

@app.get("/")
async def root():
    return {
        "message": "ML Model API deployed on Hugging Face Spaces! 🚀",
        "endpoints": {
            "health": "/health",
            "single_prediction": "/predict",
            "batch_prediction": "/predict-batch",
            "model_info": "/model-info",
            "docs": "/docs"
        },
        "model_source": "Hugging Face Hub",
        "repository": HF_REPO_ID
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "RandomForestClassifier" if model else "None",
        "features_loaded": len(feature_names) > 0
    }

@app.get("/model-info")
async def model_info():
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": str(type(model).__name__),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": getattr(model, 'n_classes_', 'Unknown'),
        "source": HF_REPO_ID
    }

@app.post("/predict")
async def predict_single(input_data: PredictionInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to array
        features = np.array([
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return {
            "prediction": int(prediction[0]),
            "probabilities": probabilities[0].tolist(),
            "class_names": ["setosa", "versicolor", "virginica"],
            "input_features": input_data.dict(),
            "model_source": HF_REPO_ID
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(input_data: BatchPredictionInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        features = np.array(input_data.data)
        
        # Validate input shape
        if features.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(feature_names)} features, got {features.shape[1]}"
            )
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "batch_size": len(predictions),
            "feature_names": feature_names,
            "model_source": HF_REPO_ID
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")