import requests
import json
import pandas as pd

class ModelAPIClient:
    # def __init__(self, base_url="https://your-app.vercel.app"):
    def __init__(self, base_url="http://127.0.0.1:8000/"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if API is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()
    
    def predict_single(self, sepal_length, sepal_width, petal_length, petal_width):
        """Make single prediction"""
        data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        response = requests.post(f"{self.base_url}/predict", json=data)
        return response.json()
    
    def predict_batch(self, data_list):
        """Make batch predictions"""
        data = {"data": data_list}
        response = requests.post(f"{self.base_url}/predict-batch", json=data)
        return response.json()
    
    def predict_dataframe(self, df):
        """Predict from pandas DataFrame"""
        data_list = df.values.tolist()
        return self.predict_batch(data_list)

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ModelAPIClient()
    
    # Check API health
    print("🔍 Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Get model info
    print("\n📊 Model Info:")
    info = client.model_info()
    print(json.dumps(info, indent=2))
    
    # Single prediction
    print("\n🎯 Single Prediction:")
    prediction = client.predict_single(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )
    print(json.dumps(prediction, indent=2))
    
    # Batch prediction
    print("\n📦 Batch Prediction:")
    batch_data = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.2, 3.4, 5.4, 2.3],  # virginica
        [5.8, 2.7, 5.1, 1.9]   # virginica
    ]
    batch_pred = client.predict_batch(batch_data)
    print(json.dumps(batch_pred, indent=2))
    
    # Using pandas DataFrame
    print("\n🐼 DataFrame Prediction:")
    df = pd.DataFrame({
        'sepal_length': [5.1, 6.2, 5.8],
        'sepal_width': [3.5, 3.4, 2.7],
        'petal_length': [1.4, 5.4, 5.1],
        'petal_width': [0.2, 2.3, 1.9]
    })
    df_pred = client.predict_dataframe(df)
    print(json.dumps(df_pred, indent=2))