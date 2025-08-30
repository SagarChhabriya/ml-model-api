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
    