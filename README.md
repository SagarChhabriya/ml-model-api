# ml-model-api



## Project Structure
```py
ml-model-api/
├── app.py                 # FastAPI application
├── train_model.py         # Script to train and save model
├── requirements.txt
├── model/                 # Folder for saved models
│   └── model.joblib
├── api/
│   └── index.py          # For Vercel deployment
└── vercel.json
```


## 1. Created GitHub Repo and Cloned it
- Create repo
- Clone it
- **Create the venv**

```py
# Create
python -m venv venv
```

```bash
# Activate
.\venv\Scripts\activate
```


- **Install dependencies**
```py
pip install numpy pandas scikit-learn fastapi uvicorn requests
# pip install python-multipart joblib
```

## 2. train_model.py

```py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load sample dataset (replace with your own data)
def train_and_save_model():
    print("Training model...")
    
    # Example: Iris dataset (replace with your data)
    data = load_iris()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'model/model.joblib')
    print("Model saved as 'model/model.joblib'")
    
    # Save feature names (important for API)
    feature_names = data.feature_names
    joblib.dump(feature_names, 'model/feature_names.joblib')
    
    return model, feature_names

if __name__ == "__main__":
    train_and_save_model()

```

## 3. Run train_model.py

```py
python train_model.py
```

## 4. Create RSET API Endpoint (FastAPI)

- Define **app.py**
- Execute

```py
uvicorn app:app --reload
```

## 5. Hit Locally via Swagger UI
http://127.0.0.1:8000/docs

Detailed Docs: http://127.0.0.1:8000/redoc 

## 6. Moving to Deployment: requirements.txt
Get the versions of the libraries used in the project

```py
# 1. 
pip show numpy

# 2. 
pip freeze
```

```py
uvicorn==0.35.0
scikit-learn==1.7.1
pydantic==2.11.7
pandas==2.3.2
numpy==2.3.2
fastapi==0.116.1
joblib==1.5.2
```


## 7. For Vercel Deployment

### api/index.py

```py
from app import app
```

### vercel.json

```js
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ]
}
```

### CI/CD
1. Push the code to GitHub
2. Login to vercel 
3. Projects > Add New > Project > Import repo > Deploy 😃😎

## 8. Predictions

### Single Prediction Using Curl
```bash
curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Batch Prediction Using Curl

```bash
curl -X POST https://your-app.vercel.app/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3],
      [5.8, 2.7, 5.1, 1.9]
    ]
  }'
```

### Via Python Request


##  Key Features of This API:

1. **Single predictions** (`/predict`) - For individual inputs
2. **Batch predictions** (`/predict-batch`) - For multiple inputs
3. **Health checks** (`/health`) - Monitor API status
4. **Model information** (`/model-info`) - Get model details
5. **Error handling** - Proper validation and error messages
6. **Input validation** - Using Pydantic models

##  Customization for Your Model:

1. **Replace the dataset** in `train_model.py` with your own data
2. **Update feature names** and input schema in `app.py`
3. **Modify class names** in the prediction response
4. **Add preprocessing** if your model requires it

##  Alternative Deployment Options:

### For larger models, consider:
1. **Google Cloud Run** (handles large models well)
2. **AWS Lambda** (serverless, pay-per-use)
3. **Azure Container Instances**
4. **Hugging Face Spaces** (for transformer models)
